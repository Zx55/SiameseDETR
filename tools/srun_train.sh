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

PARTITION=$1
CFG=$2
GPUS=$3
JOB_NAME=$4
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
START_TIME=`date +%Y%m%d-%H:%M:%S`
mkdir -p logs
LOG_FILE=logs/$train_${START_TIME}.log

GLOG_vmodule=MemcachedClient=-1 \
PYTHONPATH=$PYTHONPATH:../ \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u tools/train.py ${CFG} \
        --work_dir ${WORK_DIR} --seed 0 --deterministic --launcher="slurm" \
    2>&1 | tee $LOG_FILE > /dev/null &
