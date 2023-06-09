# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

import numpy as np


def to_numpy(pil_img):
    np_img = np.array(pil_img, dtype=np.uint8)
    if np_img.ndim < 3:
        np_img = np.expand_dims(np_img, axis=-1)
    np_img = np.rollaxis(np_img, 2)  # HWC to CHW
    return np_img
