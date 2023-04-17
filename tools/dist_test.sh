#!/bin/bash
# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

set -x

CFG=$1
GPUS=$2
CHECKPOINT=$3
PORT=${PORT:-29500}

WORK_DIR="$(dirname $CHECKPOINT)/"

# test
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py \
    $CFG \
    $CHECKPOINT \
    --work_dir $WORK_DIR --launcher="pytorch"
