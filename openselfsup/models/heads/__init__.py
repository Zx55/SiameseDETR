# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

from .contrastive_head import ContrastiveHead
from .cls_head import ClsHead
from .latent_pred_head import LatentPredictHead
from .multi_cls_head import MultiClsHead
from .detr_head import DETRHead
from .siamdetr_head import SiameseDETRPredictHead, SiameseDETREncoderGlobalLatentHead, \
    SiameseDETRDecoderLocalLatentHead
