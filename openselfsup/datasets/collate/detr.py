# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

import torch
from torch.utils.data.dataloader import default_collate

from openselfsup.utils.nested_tensor import nested_tensor_from_tensor_list, \
    nested_multiview_tensor_from_tensor_list
from ..registry import COLLATES


@COLLATES.register_module
class DETRCollateFN(object):

    @staticmethod
    def collate_fn(batch):
        return {
            'img': nested_tensor_from_tensor_list([b['img'] for b in batch]),
            'targets': [b['targets'] for b in batch]
        }

    def get_collate(self):
        return DETRCollateFN.collate_fn


@COLLATES.register_module
class UPDETRCollateFN(object):

    @staticmethod
    def collate_fn(batch):
        data = {
            'img': nested_tensor_from_tensor_list([b['img'] for b in batch]),
            'targets': [b['targets'] for b in batch],
            'patches': default_collate([b['patches'] for b in batch]),
        }
        return data

    def get_collate(self):
        return UPDETRCollateFN.collate_fn


@COLLATES.register_module
class SiameseDETRCollateFN(object):

    @staticmethod
    def collate_fn(batch):
        data = {
            'img': nested_multiview_tensor_from_tensor_list([b['img'] for b in batch]),
            'box': [b['box'] for b in batch],
            'img_size': default_collate([b['img_size'] for b in batch])
        }

        if 'crop' in batch[0]:
            data.update({
                'crop': default_collate([b['crop'] for b in batch])
            })

        return data

    def get_collate(self):
        return SiameseDETRCollateFN.collate_fn
