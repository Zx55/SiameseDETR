# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from UP-DETR (https://github.com/dddzg/up-detr)
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import RoIAlign
from typing import Dict, List, Optional, Union

from openselfsup.utils import print_log, parse_losses
from openselfsup.utils.nested_tensor import NestedTensor, nested_tensor_from_tensor_list
import openselfsup.utils.box_ops as box_ops
from . import builder
from .necks.helper import MLP
from .registry import MODELS
from .utils.position_encoding import build_position_encoding


@MODELS.register_module
class UPDETR(nn.Module):

    def __init__(self,
                 backbone: Dict,
                 position_embedding: Dict,
                 neck: Dict,
                 head: Dict,
                 backbone_channels: int,
                 hidden_dim: int,
                 pretrained: str = None,
                 freeze_backbone: bool = True,
                 query_shuffle: bool = False,
                 mask_ratio: float = 0.1,
                 num_queries: int = 100,
                 num_patches: int = 10,
                 multi_scale_features: bool=False,
                 multi_scale_features_backbone_strides=None,
                 multi_scale_features_backbone_num_channels=None,
                 num_feature_levels: int=4):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        self.pos_embed = build_position_encoding(position_embedding)
        self.transformer = builder.build_neck(neck)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.head = builder.build_head(head)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.patch2query = nn.Linear(backbone_channels, hidden_dim)
        self.multi_scale_features = multi_scale_features
        if multi_scale_features:
            assert neck['type'] == 'DeformableTR'
            self.backbone_strides = multi_scale_features_backbone_strides
            self.backbone_num_channels = multi_scale_features_backbone_num_channels
            self.num_feature_levels = num_feature_levels  # 3 from backbone and 1 from input_proj
            self.input_proj = self.build_multi_scale_input_proj(hidden_dim)
            for proj in self.input_proj: # from deformable
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
            assert backbone_channels == sum(self.backbone_num_channels)
        else:
            self.input_proj = nn.Conv2d(backbone_channels, hidden_dim, kernel_size=1)

        self.num_queries = num_queries
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.query_shuffle = query_shuffle

        assert num_queries % num_patches == 0  # for simplicity
        query_per_patch = num_queries // num_patches
        # the attention mask is fixed during the pre-training
        self.attention_mask = torch.ones(self.num_queries, self.num_queries) * float('-inf')
        for i in range(query_per_patch):
            self.attention_mask[i * query_per_patch:(i + 1) * query_per_patch,
                                i * query_per_patch:(i + 1) * query_per_patch] = 0

        self.init_weights(pretrained=pretrained)
        self.freeze_backbone = freeze_backbone

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load backbone from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained, strict=False)
        self.transformer.init_weights()

    def build_multi_scale_input_proj(self, hidden_dim):
        num_backbone_outs = len(self.backbone_strides)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = self.backbone_num_channels[_]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim)))

        if self.num_feature_levels > 1:
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim)))
                in_channels = hidden_dim
        return nn.ModuleList(input_proj_list)

    def input_proj_multi_scale(self, view, features, pos):
        srcs = []
        masks = []
        poses = []

        for l, (feat, pos_) in enumerate(zip(features, pos)):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            poses.append(pos_)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = view.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.pos_embed(NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        return srcs, masks, poses

    def extract_feature_maps(self, samples):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        if self.freeze_backbone:
            with torch.no_grad():
                out = [o.detach() for o in self.backbone(samples.tensors)]
        else:
            out = self.backbone(samples.tensors)

        if self.multi_scale_features:
            features = []
            pos = []
            for l, feat in enumerate(out): # 3 layer features from resnet
                mask = F.interpolate(samples.mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
                feat = NestedTensor(feat, mask)
                pos_ = self.pos_embed(feat).to(feat.tensors.dtype)
                features.append(feat)
                pos.append(pos_)
            return features, pos # Tuple[List[NestedTensor], List[Tensor]]
            
        else:
            out = out[-1]
            mask = F.interpolate(samples.mask[None].float(), size=out.shape[-2:]).to(torch.bool)[0]
            out = NestedTensor(out, mask)
            pos = self.pos_embed(out).to(out.tensors.dtype)
            return out, pos

    def align_and_proj(self, feature_map: Tensor, box: List[Tensor]) -> Tensor:
        bs = len(box)

        patch_feature = self.align(feature_map, box)
        patch_feature_gt = self.avgpool(patch_feature).flatten(1)
        patch_feature = self.patch2query(patch_feature_gt) \
            .view(bs, self.num_patches, -1) \
            .repeat_interleave(self.num_queries // self.num_patches, dim=1) \
            .permute(1, 0, 2) \
            .contiguous()  # n_queries, bs, c
        return patch_feature, patch_feature_gt

    def forward_train(self, samples: NestedTensor, patches: Tensor, targets: List[Dict]):
        # batch_num_patches = patches.shape[1]
        batch_num_patches = targets[0]['boxes'].shape[0]
        
        features, pos = self.extract_feature_maps(samples)
        if self.multi_scale_features:
            src, mask = [feat_.decompose()[0] for feat_ in features], [feat_.decompose()[1] for feat_ in features] # all len 3
        else:
            src, mask = features.decompose()

        bs = patches.size(0)
        patches = patches.flatten(0, 1)
        if self.freeze_backbone:
            with torch.no_grad():
                out = [o.detach() for o in self.backbone(patches)]
        else:
            out = self.backbone(patches)

        if self.multi_scale_features:
            patch_feature = out
            pool_feat = [self.avgpool(feat).flatten(1) for feat in  patch_feature]
            patch_feature_gt = torch.cat(pool_feat, dim=1)
            patch_feature = self.patch2query(patch_feature_gt) \
                .view(bs, batch_num_patches, -1) \
                .repeat_interleave(self.num_queries // self.num_patches, dim=1) \
                .permute(1, 0, 2) \
                .contiguous()
        else:
            patch_feature = out[-1]
            patch_feature_gt = self.avgpool(patch_feature).flatten(1)
            patch_feature = self.patch2query(patch_feature_gt) \
                .view(bs, batch_num_patches, -1) \
                .repeat_interleave(self.num_queries // self.num_patches, dim=1) \
                .permute(1, 0, 2) \
                .contiguous()

        idx = torch.randperm(self.num_queries) if self.query_shuffle \
            else torch.arange(self.num_queries)
        mask_query_patch = (torch.rand(self.num_queries, bs, 1, device=samples.tensors.device)
                            > self.mask_ratio).float()
        # mask some query patch and add query embedding
        patch_feature = patch_feature * mask_query_patch + \
            self.query_embed.weight[idx, :].unsqueeze(1).repeat(1, bs, 1)

        if not self.multi_scale_features:
            src_proj = self.input_proj(src)
        else:
            src_proj, mask, pos = self.input_proj_multi_scale(samples, features, pos)

        hs, mem, references = self.transformer(
            src_proj, mask, patch_feature, pos,
            self.attention_mask.to(patch_feature.device))
        return self.head(hs, targets, patch_feature_gt, references)[0]

    def forward(self, img: NestedTensor, patches: Tensor = None, targets: List[Dict] = None, mode: str ='train', **kwargs):      
        img = img.cuda()  # for NestedTensor class
        patches = patches.cuda() if patches is not None else None
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        if mode == 'train':
            return self.forward_train(img, patches, targets)
        elif mode == 'test':
            raise NotImplementedError
        elif mode == 'extract':
            raise NotImplementedError
        else:
            raise Exception("No such mode: {}".format(mode))

    def train_step(self, data, *args, **kwargs):
        losses = self.forward(**data, mode='train')
        loss, log_vars = parse_losses(losses)
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data['img'].data))
        return outputs
