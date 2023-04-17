# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

from .alias_multinomial import AliasMethod
from .collect import nondist_forward_collect, dist_forward_collect
from .collect_env import collect_env
from .config_tools import traverse_replace
from .flops_counter import get_model_complexity_info
from .logger import get_root_logger, print_log
from .registry import Registry, build_from_cfg
from . import optimizers
from .optim import build_optimizer
from .misc import set_random_seed, parse_losses
from .coco_eval import CocoEvaluator
