#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------
set -x

PYTHON=${PYTHON:-"python"}
CFG=$1
GPUS=$2
WORK_DIR=$3
PY_ARGS=${@:4} # "--checkpoint $CHECKPOINT --pretrained $PRETRAINED"
PORT=${PORT:-29500}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/extract.py $CFG --layer-ind "0,1,2,3,4" --work_dir $WORK_DIR \
    --launcher pytorch ${PY_ARGS}
