# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

from openselfsup.utils import build_from_cfg

from .registry import HOOKS


def build_hook(cfg, default_args=None):
    return build_from_cfg(cfg, HOOKS, default_args)
