#!/bin/bash
# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

set -o errexit
PARTITION=$1
NUM_PROC=$2
DATA_ROOT=$3
DATA_META=$4
SAVE_DIR=$5
DATASET=$6

srun -p $PARTITION -n$NUM_PROC --ntasks-per-node=10 --job-name=hpcplayer \
    python -u task.py --root $DATA_ROOT --source $DATA_META --save_dir $SAVE_DIR --dataset $DATASET
