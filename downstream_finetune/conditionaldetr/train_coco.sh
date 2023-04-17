# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) Microsoft. All Rights Reserved
# ------------------------------------------------------------------------

PARTITION=$1
CKPT=$2
GPUS=$3
JOB_NAME=$4

START_TIME=`date +%Y%m%d-%H:%M:%S`
mkdir -p logs
LOG_FILE=logs/$train_${START_TIME}.log

srun -p ${PARTITION} \
     --job-name=${JOB_NAME} \
     -n${GPUS} \
     --ntasks-per-node=8 \
     --cpus-per-task=4 \
     --gres=gpu:8 \
     --kill-on-bad-exit=1 \
     python -u -m main \
            --dataset_file coco \
            --lr 1e-4 \
            --lr_backbone 5e-5 \
            --root_path ../../upstream_pretrain/data/datasets/mscoco2017/ \
            --output_dir saves/${JOB_NAME}_${START_TIME} \
            --num_queries 100 \
            --pretrain ${CKPT} \
     2>&1 | tee $LOG_FILE > /dev/null &
