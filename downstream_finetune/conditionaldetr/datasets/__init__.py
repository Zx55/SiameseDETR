# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) Microsoft. All Rights Reserved
# ------------------------------------------------------------------------

from torch.utils.data import DataLoader, DistributedSampler, Subset, \
    RandomSampler, SequentialSampler, BatchSampler
import torchvision

from util.misc import collate_fn
from .coco import build as build_coco
from .voc import build as build_voc


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'voc':
        return build_voc(image_set, args)
        
    raise ValueError(f'dataset {args.dataset_file} not supported')


def build_train_loader(args):
    dataset = build_dataset('train', args)
    if args.distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, 
                        num_workers=args.num_workers)
    return loader, sampler


def build_val_loader(args):
    dataset = build_dataset('val', args)
    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, args.batch_size, sampler=sampler, drop_last=False,
                        collate_fn=collate_fn, num_workers=args.num_workers)

    if args.dataset_file == 'coco':
        base_ds = get_coco_api_from_dataset(dataset)
    elif args.dataset_file == 'coco_panoptic':
        coco_val = build_coco('val', args)
        base_ds = get_coco_api_from_dataset(coco_val)
    elif args.dataset_file == 'voc':
        base_ds = get_coco_api_from_dataset(dataset)
    else:
        base_ds = None
    return loader, base_ds
