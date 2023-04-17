# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

from .builder import build_hook
from .byol_hook import BYOLHook
from .deepcluster_hook import DeepClusterHook
from .odc_hook import ODCHook
from .optimizer_hook import DistOptimizerHook
from .extractor import Extractor
from .validate_hook import ValidateHook
from .registry import HOOKS
