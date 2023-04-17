# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from UP-DETR (https://github.com/dddzg/up-detr)
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from openselfsup.utils import build_from_cfg
from .builder import build_datasource
from .pipelines import ComposeWithTarget
from .registry import DATASETS, PIPELINES
from .utils import to_numpy


def get_random_patch_from_img(img, min_pixel=8):
    """
    :param img: original image
    :param min_pixel: min pixels of the query patch
    :return: query_patch,x,y,w,h
    """
    w, h = img.size
    min_w, max_w = min_pixel, w - min_pixel
    min_h, max_h = min_pixel, h - min_pixel
    sw = np.random.randint(min_w, max_w + 1)
    sh = np.random.randint(min_h, max_h + 1)
    x = np.random.randint(w - sw) if sw != w else 0
    y = np.random.randint(h - sh) if sh != h else 0
    patch = img.crop((x, y, x + sw, y + sh)) 
    return patch, x, y, sw, sh


@DATASETS.register_module
class UPDETRDataset(Dataset):

    def __init__(self, data_source, image_pipeline, patch_pipeline, num_patches=10, prefetch=False, gen_box_method='random'):
        self.data_source = build_datasource(data_source)
        image_pipeline = [build_from_cfg(p, PIPELINES) for p in image_pipeline]
        self.image_pipeline = ComposeWithTarget(image_pipeline)
        patch_pipeline = [build_from_cfg(p, PIPELINES) for p in patch_pipeline]
        self.patch_pipeline = Compose(patch_pipeline)
            
        self.num_patches = num_patches
        self.prefetch = prefetch
        self.gen_box_method = gen_box_method

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        if self.gen_box_method == 'from_datasource':
            img, unsup_boxes = self.data_source.get_sample(idx)
        else:
            img = self.data_source.get_sample(idx)
            unsup_boxes = None
    
        w, h = img.size
        if w <= 16 or h <= 16:
            return self[(idx + 1) % len(self)]

        target, patches = self.gen_patches(img, unsup_boxes=unsup_boxes)        
        img, target = self.image_pipeline(img, target)
        patches = torch.stack(patches, dim=0)

        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
            patches = torch.from_numpy(to_numpy(patches))
        return dict(img=img, targets=target, patches=patches)

    def gen_patches(self, img, unsup_boxes=None):
        w, h = img.size
        target = {
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)])}

        iscrowd, labels, boxes, area, patches = [], [], [], [], []
        if self.gen_box_method == 'random':            
            while len(area) < self.num_patches:
                patch, x, y, sw, sh = get_random_patch_from_img(img)
                boxes.append([x, y, x + sw, y + sh])
                area.append(sw * sh)
                iscrowd.append(0)
                labels.append(1)
                patches.append(self.patch_pipeline(patch))
        elif self.gen_box_method == 'from_datasource':
            assert unsup_boxes is not None and 'unsup_boxes' in unsup_boxes
            unsup_boxes_raw = unsup_boxes['unsup_boxes'].to(torch.float32)

            bound_mask = (unsup_boxes_raw >= 0).all(dim=-1)
            coord_mask = (unsup_boxes_raw[:, 2:] >= unsup_boxes_raw[:, :2]).all(dim=-1)
            mask = bound_mask & coord_mask
            unsup_boxes_raw = unsup_boxes_raw[mask]

            while len(unsup_boxes_raw) < self.num_patches:
                unsup_boxes_raw = torch.cat([unsup_boxes_raw, unsup_boxes_raw])
            unsup_boxes_raw = unsup_boxes_raw[:self.num_patches]

            unsup_boxes_raw[:, 0].clamp_(min=0, max=w)
            unsup_boxes_raw[:, 1].clamp_(min=0, max=h)
            unsup_boxes_raw[:, 2].clamp_(min=0, max=w)
            unsup_boxes_raw[:, 3].clamp_(min=0, max=h)

            for box in unsup_boxes_raw:
                x1, y1, x2, y2 = box.unbind(0)
                patch = img.crop((x1.item(), y1.item(), x2.item(), y2.item()))
                boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                area.append(((x2 - x1) * (y2 - y1)).item())
                iscrowd.append(0)
                labels.append(1)
                patches.append(self.patch_pipeline(patch))
        else:
            raise ValueError(f'gen_box_method {self.gen_box_method} not supported.')

        target['iscrowd'] = torch.tensor(iscrowd)
        target['labels'] = torch.tensor(labels)
        target['boxes'] = torch.tensor(boxes)
        target['area'] = torch.tensor(area)
        return target, patches
        
    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplementedError
