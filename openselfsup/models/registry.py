# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

from openselfsup.utils import Registry

MODELS = Registry('model')
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
MEMORIES = Registry('memory')
LOSSES = Registry('loss')
