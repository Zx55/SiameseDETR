// ------------------------------------------------------------------------
// Siamese DETR
// Copyright (c) 2023 SenseTime. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 [see LICENSE for details]
// ------------------------------------------------------------------------
// Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
// Copyright (c) SenseTime. All Rights Reserved
// ------------------------------------------------------------------------

#pragma once
#include <torch/extension.h>

at::Tensor
ms_deform_attn_cpu_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step);

std::vector<at::Tensor>
ms_deform_attn_cpu_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step);

