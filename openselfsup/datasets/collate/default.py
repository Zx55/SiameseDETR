# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

from mmcv.parallel import collate as mmcv_collate
from functools import partial
from torch.utils.data.dataloader import default_collate

from ..registry import COLLATES


@COLLATES.register_module
class DefaultCollateFN(object):

    def get_collate(self):
        return default_collate


@COLLATES.register_module
class MMCVCollateFN(object):

    def __init__(self, samples_per_gpu):
        self.samples_per_gpu = samples_per_gpu

    def get_collate(self):
        return partial(mmcv_collate, samples_per_gpu=self.samples_per_gpu)
