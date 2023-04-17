# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

from mmcv import Config

def traverse_replace(d, key, value):
    if isinstance(d, (dict, Config)):
        for k, v in d.items():
            if k == key:
                d[k] = value
            else:
                traverse_replace(v, key, value)
    elif isinstance(d, (list, tuple, set)):
        for v in d:
            traverse_replace(v, key, value)
