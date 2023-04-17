# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

from .accuracy import Accuracy, accuracy
from .conv_module import ConvModule, build_conv_layer
from .conv_ws import ConvWS2d, conv_ws_2d
from .gather_layer import GatherLayer
from .multi_pooling import MultiPooling
from .norm import build_norm_layer
from .scale import Scale
#from .weight_init import (bias_init_with_prob, kaiming_init, normal_init,
#                          uniform_init, xavier_init)
from .sobel import Sobel

#__all__ = [
#    'conv_ws_2d', 'ConvWS2d', 'build_conv_layer', 'ConvModule',
#    'build_norm_layer', 'xavier_init', 'normal_init', 'uniform_init',
#    'kaiming_init', 'bias_init_with_prob', 'Scale', 'Sobel'
#]
