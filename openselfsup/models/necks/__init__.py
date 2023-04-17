# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

from .linear import LinearNeck
from .relative_loc import RelativeLocNeck
from .nonlinear import NonLinearNeckV0, NonLinearNeckV1, NonLinearNeckV2, \
    NonLinearNeckSimCLR
from .conditional_tr import ConditionalTR
from .vanilla_tr import VanillaTR
from .avgpool import AvgPoolNeck
from .deformable_tr import DeformableTR
