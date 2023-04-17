// ------------------------------------------------------------------------
// Siamese DETR
// Copyright (c) 2023 SenseTime. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 [see LICENSE for details]
// ------------------------------------------------------------------------
// Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
// Copyright (c) SenseTime. All Rights Reserved
// ------------------------------------------------------------------------

#include "ms_deform_attn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
}
