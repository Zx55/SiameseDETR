# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

from ..registry import DATASOURCES
from .image_list import ImageList
from openselfsup.utils import print_log
import os
import random
import torch
import numpy as np


@DATASOURCES.register_module
class ImageNet(ImageList):

    def __init__(self, root, list_file, memcached, mclient_path, backend='mc', return_label=True, *args, **kwargs):
        super(ImageNet, self).__init__(
            root, list_file, memcached, mclient_path, backend, return_label)


@DATASOURCES.register_module
class ImageNetWithUnsupbox(ImageNet):

    def __init__(self, root, list_file, memcached, mclient_path, backend='mc', return_label=True, unsup_boxes_root=None, *args, **kwargs):
        super(ImageNetWithUnsupbox, self).__init__(
            root, list_file, memcached, mclient_path, backend, return_label)
        self.unsup_boxes_root = unsup_boxes_root
        assert return_label is False

    def __get_np_item__(self, index):
        image_rel_path = self.fns[index][len(self.root):].strip('/')
        numpy_rel_path = image_rel_path.split('.')[0] + '.npy'
        fn_np = os.path.join(self.unsup_boxes_root, numpy_rel_path)
        if self.memcached:
            self._init_memcached()
            nparray = self.mc_loader.get_np_item(fn_np)
        else:
            nparray = np.load(fn_np, allow_pickle=True)

        return nparray

    def get_sample(self, idx):
        img = super(ImageNetWithUnsupbox, self).get_sample(idx)
        nparray = self.__get_np_item__(idx)
        if nparray is None:
            next_idx = random.randint(0, self.get_length() - 1)
            print_log(f'idx -> next_idx: {idx} -> {next_idx}', 'root')
            return self.get_sample(next_idx)
            
        unsup_boxes = torch.from_numpy(nparray.tolist()['box'])
        target = dict(unsup_boxes=unsup_boxes)
        return img, target
