# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn as nn


class Sobel(nn.Module):
    """Sobel layer."""

    def __init__(self):
        super(Sobel, self).__init__()
        grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        grayscale.weight.data.fill_(1.0 / 3.0)
        grayscale.bias.data.zero_()
        sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        sobel_filter.weight.data[0, 0].copy_(
            torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
        sobel_filter.weight.data[1, 0].copy_(
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
        sobel_filter.bias.data.zero_()
        self.sobel = nn.Sequential(grayscale, sobel_filter)
        for p in self.sobel.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.sobel(x)