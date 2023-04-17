# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import RoIAlign
from typing import Dict, List, Tuple, Union

from openselfsup.utils import print_log, parse_losses
from openselfsup.utils.nested_tensor import NestedTensor, nested_tensor_from_tensor_list, \
    nested_multiview_tensor_from_tensor_list
from . import builder
from .necks.helper import MLP
from .registry import MODELS
from .utils.position_encoding import build_position_encoding


def decompose_multiview_tensor(tensor: NestedTensor) -> Tuple[NestedTensor, NestedTensor]:
    tensors, mask = tensor.decompose()
    if tensors.ndim != 5:  # bs, n_view, c, h, w
        raise RuntimeError(f'invalid multiview NestedTensor shape {tensors.shape}')
    
    img_v1 = tensors[:, 0, ...].contiguous()
    mask_v1 = mask[:, 0, ...].contiguous()
    img_v2 = tensors[:, 1, ...].contiguous()
    mask_v2 = mask[:, 1, ...].contiguous()
    return NestedTensor(img_v1, mask_v1), NestedTensor(img_v2, mask_v2)


def decompose_multiview_box(box: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
    box_v1 = [b[0, ...].contiguous() for b in box]
    box_v2 = [b[1, ...].contiguous() for b in box]
    return box_v1, box_v2


def add_randomness_on_multiview_box(box_v1, box_v2, img_size, ratio=0.05):
    def _add(box, size, ratio):
        w = (box[:, 2] - box[:, 0]).view(-1, 1)
        h = (box[:, 3] - box[:, 1]).view(-1, 1)
        randomness = torch.randn(w.shape[0], 4).sigmoid()
        randomness = randomness * ratio * 2 - ratio
        addon = torch.cat([w, h, w, h], dim=-1) * randomness.to(box.device)
        new_box = box + addon
        
        max_h, max_w = size.unbind()
        new_box[:, ::2].clamp_(min=0, max=max_w)
        new_box[:, 1::2].clamp_(min=0, max=max_h)
        return new_box
    
    new_box_v1 = []
    new_box_v2 = []
    for b1, b2, sz in zip(box_v1, box_v2, img_size):
        new_box_v1.append(_add(b1, sz, ratio))
        new_box_v2.append(_add(b2, sz, ratio))
    return new_box_v1, new_box_v2


@MODELS.register_module
class SiameseDETR(nn.Module):

    def __init__(self,
                 backbone: Dict,
                 position_embedding: Dict,
                 transformer: Dict,
                 pred_head: Dict,
                 backbone_channels: int,
                 hidden_dim: int,
                 encoder_head: Union[Dict, List[Dict]] = None,
                 decoder_head: Union[Dict, List[Dict]] = None,
                 pretrained: str = None,
                 freeze_backbone: bool = False,
                 num_queries: int = 100,
                 num_patches: int = 100,
                 box_disturbance: float = 0,
                 query_shuffle: bool = False,
                 feature_recon: bool = False,
                 weight_dict: Dict = None,
                 multi_scale_features: bool = False,
                 multi_scale_features_backbone_strides: tuple = None,
                 multi_scale_features_backbone_num_channels: tuple = None,
                 num_feature_levels: int=4):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        self.patch2query = nn.Linear(backbone_channels, hidden_dim)
        self.transformer_config = transformer
        self.transformer = builder.build_neck(transformer)
        self.pred_head = builder.build_head(pred_head)  # box + cls + recon
        self.enc_head, self.dec_head = None, None
        self.multi_scale_features = multi_scale_features
        if multi_scale_features:
            assert transformer['type'] == 'DeformableTR'
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

        if encoder_head is not None:
            if not isinstance(encoder_head, List):
                encoder_head = [encoder_head]
            self.enc_head = nn.Sequential(*[builder.build_head(head) for head in encoder_head])
        if decoder_head is not None:
            if not isinstance(decoder_head, List):
                decoder_head = [decoder_head]
            self.dec_head = nn.Sequential(*[builder.build_head(head) for head in decoder_head])

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.pos_embed = build_position_encoding(position_embedding)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.multi_scale_features:
            self.align = []
            for stride in self.backbone_strides:
                roi_align = RoIAlign(output_size=(7, 7), spatial_scale=(1. / stride), sampling_ratio=-1)  # wo/ DC5
                self.align.append(roi_align)
        else:
            self.align = RoIAlign(output_size=(7, 7), spatial_scale=(1. / 32.), sampling_ratio=-1)  # wo/ DC5

        self.num_queries = num_queries
        self.num_patches = num_patches
        self.query_shuffle = query_shuffle
        self.box_disturbance = box_disturbance
        self.weight_dict = self.generate_weight_dict(weight_dict)
        self.feature_recon = feature_recon

        self.init_weights(pretrained=pretrained)
        self.freeze_backbone = freeze_backbone

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

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load backbone from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained, strict=False)
        self.transformer.init_weights()

    def generate_weight_dict(self, weight_dict):
        num_repeat = weight_dict.pop('num_repeat', 1)
        blocklist = ['loss_enc_global_contra']

        if self.pred_head.aux_loss:
            aux_weight_dict = {}
            for i in range(num_repeat - 1):
                aux_weight_dict.update({f'{k}_{i}': v for k, v in weight_dict.items()
                                        if k not in blocklist})
            weight_dict.update(aux_weight_dict)
        print_log(f'weight_dict: {json.dumps(weight_dict, indent=4)}', 'root')
        return weight_dict        

    def extract_feature_maps(self, view: NestedTensor) -> Tuple[NestedTensor, Tensor] or Tuple[List[NestedTensor], List[Tensor]]:
        if self.freeze_backbone:
            with torch.no_grad():
                out = [o.detach() for o in self.backbone(view.tensors)]
        else:
            out = self.backbone(view.tensors)

        if self.multi_scale_features:
            features = []
            pos = []
            for l, feat in enumerate(out): # 3 layer features from resnet
                mask = F.interpolate(view.mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
                feat = NestedTensor(feat, mask)
                pos_ = self.pos_embed(feat).to(feat.tensors.dtype)
                features.append(feat)
                pos.append(pos_)
            return features, pos # Tuple[List[NestedTensor], List[Tensor]]            
        else:
            out = out[-1]
            mask = F.interpolate(view.mask[None].float(), size=out.shape[-2:]).to(torch.bool)[0]
            out = NestedTensor(out, mask)
            pos = self.pos_embed(out).to(out.tensors.dtype)
            return out, pos            

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

    def extract_patch_feature(self, crop: Tensor, query_embed: Tensor = None, to_query = True) -> Tensor:
        bs, num_patch = crop.shape[:2]
        crop = crop.flatten(0, 1)

        if self.freeze_backbone:
            with torch.no_grad():
                patches = [o.detach() for o in self.backbone(crop)]
        else:
            patches = self.backbone(crop)

        if self.multi_scale_features:
            patch = [self.avgpool(o).flatten(1) for o in patches]
            patch = torch.cat(patch, dim=1)  # TODO 暂时用concat的方式结合multi scale特征, 作为recon的目标, 而query也用相似的方法得到
        else:
            patch = patches[-1]
            patch = self.avgpool(patch).flatten(1)  # bs, num_patch x c -> num_patch, bs, c

        if to_query:
            patch = self.patch2query(patch) \
                .view(bs, num_patch, -1) \
                .repeat_interleave(self.num_queries // self.num_patches, dim=1) \
                .permute(1, 0, 2) \
                .contiguous()

            if query_embed is not None:
                patch = patch + query_embed

        return patch

    def align_and_proj(self, feature_map: NestedTensor or List[NestedTensor], box: List[Tensor], query_embed: Tensor = None) -> Tensor:
        bs = len(box)
        if self.multi_scale_features:
            # for deformable, 输入到query的feature只取backbone部分的3个feature进行align和concat
            assert len(feature_map) ==  len(self.align) # and len(self.align) == self.num_feature_levels - 1  # 3
            multi_scale_patch_features = []
            for feat_, align_ in zip(feature_map, self.align):
                patch_feat_ = align_(feat_.tensors, box)
                multi_scale_patch_features.append(patch_feat_)
            patch_feature = torch.cat(multi_scale_patch_features, dim=1)  # TODO 暂时用concat的方式结合multi scale特征，作为query输入
        else:
            patch_feature = self.align(feature_map.tensors, box)

        patch_feature = self.avgpool(patch_feature).flatten(1).view(bs, self.num_patches, -1)  # bs, n_patch, c
        patch_feature = self.patch2query(patch_feature) \
            .repeat_interleave(self.num_queries // self.num_patches, dim=1) \
            .permute(1, 0, 2) \
            .contiguous()  # n_queries, bs, c

        if query_embed is not None:
            patch_feature = patch_feature + query_embed

        return patch_feature

    def forward_train(self, img: NestedTensor, box: List[Tensor], img_size: Tensor, crop: Tensor):
        # Data preprocess and box jitter.
        img_v1, img_v2 = decompose_multiview_tensor(img)
        box_v1, box_v2 = decompose_multiview_box(box)
        disturbed_box_v1, disturbed_box_v2 = None, None
        if self.box_disturbance > 0:
            disturbed_box_v1, disturbed_box_v2 = add_randomness_on_multiview_box(box_v1, box_v2, img_size, ratio=self.box_disturbance)

        # Extract global views.
        feat_v1, pos_v1 = self.extract_feature_maps(img_v1)  # in deformable, feat and pos correspond to 3 features from layer 2 3 4
        feat_v2, pos_v2 = self.extract_feature_maps(img_v2)
        if self.multi_scale_features:
            src_v1, mask_v1 = [feat_v1_.decompose()[0] for feat_v1_ in feat_v1], [feat_v1_.decompose()[1] for feat_v1_ in feat_v1] # all len 3
            src_v2, mask_v2 = [feat_v2_.decompose()[0] for feat_v2_ in feat_v2], [feat_v2_.decompose()[1] for feat_v2_ in feat_v2]
        else:
            src_v1, mask_v1 = feat_v1.decompose()
            src_v2, mask_v2 = feat_v2.decompose()

        # Extract patch views through ROIAlign.
        # Following UP-DETR, we add each patch view on every ten query embeddings.
        idx = torch.randperm(self.num_queries) if self.query_shuffle \
                else torch.arange(self.num_queries)
        query_embed = self.query_embed.weight[idx, :].unsqueeze(1).repeat(1, len(box_v1), 1)

        recon_v1_gt = recon_v2_gt = None
        if self.feature_recon:
            assert crop is not None
            patch_v1 = self.align_and_proj(feat_v1, box_v1 if disturbed_box_v1 is None else disturbed_box_v1, query_embed)
            patch_v2 = self.align_and_proj(feat_v2, box_v2 if disturbed_box_v2 is None else disturbed_box_v2, query_embed)

            crop_v1 = crop[:, 0, ...].contiguous()
            crop_v2 = crop[:, 1, ...].contiguous()
            with torch.no_grad():
                recon_v1_gt = self.extract_patch_feature(crop_v1, query_embed=None, to_query=False)
                recon_v2_gt = self.extract_patch_feature(crop_v2, query_embed=None, to_query=False)
            recon_v1_gt = recon_v1_gt.detach()
            recon_v2_gt = recon_v2_gt.detach()
        else:
            patch_v1 = self.align_and_proj(feat_v1, box_v1 if disturbed_box_v1 is None else disturbed_box_v1, query_embed)
            patch_v2 = self.align_and_proj(feat_v2, box_v2 if disturbed_box_v2 is None else disturbed_box_v2, query_embed)

        # Perform Cross View Cross Attention: patch_v1 -> feat_v2, patch_v2 -> feat_v1.
        # Note that for DeformableDETR-MS, we use a list of four features:
        # Three backbone features and an additional scale of feature from input proj.
        if self.multi_scale_features: 
            src_v2_proj, mask_v2, pos_v2 = self.input_proj_multi_scale(img_v2, feat_v2, pos_v2)
            src_v1_proj, mask_v1, pos_v1 = self.input_proj_multi_scale(img_v1, feat_v1, pos_v1)
        else:  
            src_v2_proj = self.input_proj(src_v2)
            src_v1_proj = self.input_proj(src_v1)
        hs_v1, mem_v2, ref_v1 = self.transformer(src_v2_proj, mask_v2, patch_v1, pos_v2)  # ref is (init, inter), used for DeformableDETR only
        hs_v2, mem_v1, ref_v2 = self.transformer(src_v1_proj, mask_v1, patch_v2, pos_v1)

        # Compute loss function: region detection + local/global discrimination.
        losses = {}

        # Region detection loss and local discrimination loss.
        loss_v1, indices_v1 = self.pred_head(hs_v1, ref_v1, torch.stack(box_v2), img_size, recon_v1_gt)
        loss_v2, indices_v2 = self.pred_head(hs_v2, ref_v2, torch.stack(box_v1), img_size, recon_v2_gt)
        losses.update({k: loss_v1[k] * 0.5 + loss_v2[k] * 0.5 for k in loss_v1.keys()})

        # Encoder loss, i.e., global discrimination loss. 
        if self.enc_head is not None:
            for head in self.enc_head:
                loss_enc = head(mem_v1, mem_v2, 
                                box_v1 if disturbed_box_v1 is None else disturbed_box_v1,
                                box_v2 if disturbed_box_v2 is None else disturbed_box_v2)
                losses.update(loss_enc)

        # Decoder loss, not used.
        if self.dec_head is not None:
            for head in self.dec_head:
                loss_dec = head(hs_v1, hs_v2, indices_v1, indices_v2)
                losses.update(loss_dec)
            
        losses.update({'weight_dict': self.weight_dict})
        return losses

    def forward(self, img: NestedTensor, box: List[Tensor], img_size: Tensor, crop: Tensor = None, mode='train', **kwargs):
        img = img.cuda()
        box = [b.cuda() for b in box]
        crop = crop.cuda() if crop is not None else None

        if mode == 'train':
            return self.forward_train(img, box, img_size, crop)
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
