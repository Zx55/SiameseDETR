# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

from .builder import build_dataset, build_collate_fn
from .byol import BYOLDataset
from .data_sources import *
from .pipelines import *
from .collate import *
from .classification import ClassificationDataset
from .deepcluster import DeepClusterDataset
from .extraction import ExtractDataset
from .npid import NPIDDataset
from .rotation_pred import RotationPredDataset
from .relative_loc import RelativeLocDataset
from .contrastive import ContrastiveDataset
from .updetr import UPDETRDataset
from .siamese_detr import SiameseDETRDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .loader import DistributedGroupSampler, GroupSampler, build_dataloader
from .registry import DATASETS
