# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

import torch
from torch import Tensor
from typing import Optional, List, Tuple


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self) -> Tuple[Tensor, Tensor]:
        return self.tensors, self.mask

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

    @property
    def data(self):
        return self.tensors

    def __repr__(self):
        return str(self.tensors)


def nested_multiview_tensor_from_tensor_list(tensor_list: List[List[Tensor]]) -> NestedTensor:
    '''
    format: [[img1_view1, img1_view2], [img2_view1, img2_view2], ...]
    '''
    if tensor_list[0][0].ndim != 3:
        raise ValueError('not supported')

    max_size = _max_by_axis([list(view.shape) for img in tensor_list for view in img])
    batch_shape = [len(tensor_list), len((tensor_list[0]))] + max_size
    bs, n_view, c, h, w = batch_shape
    dtype = tensor_list[0][0].dtype
    device = tensor_list[0][0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((bs, n_view, h, w), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        for i in range(n_view):
            pad_img[i, :img[i].shape[0], :img[i].shape[1], :img[i].shape[2]].copy_(img[i])
            m[i, :img[i].shape[1], :img[i].shape[2]] = False
    return NestedTensor(tensor, mask)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    if tensor_list[0].ndim != 3:
        raise ValueError('not supported')

    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    bs, c, h, w = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((bs, h, w), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        m[: img.shape[1], :img.shape[2]] = False
    return NestedTensor(tensor, mask)    
