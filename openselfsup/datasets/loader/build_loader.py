# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

import platform
import random
import torch

import numpy as np
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader

from .sampler import DistributedSampler, DistributedGivenIterationSampler
from torch.utils.data import RandomSampler
from ..builder import build_collate_fn


if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     replace=False,
                     seed=None,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        imgs_per_gpu (int): Number of images on each GPU, i.e., batch size of
            each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        replace (bool): Replace or not in random shuffle.
            It works on when shuffle is True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    if dist:
        rank, world_size = get_dist_info()
        sampler_type = kwargs.pop('sampler_type', ('DistributedSampler', -1))
        if sampler_type[0] == 'DistributedSampler':
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=shuffle, replace=replace)
        elif sampler_type[0] == 'DistributedGivenIterationSampler':
            sampler = DistributedGivenIterationSampler(
                dataset, total_iter=sampler_type[1], batch_size=imgs_per_gpu)
        else:
            raise ValueError(f'sampler_type {sampler_type} not supported')

        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        if replace:
            raise NotImplemented
        sampler = RandomSampler(
            dataset) if shuffle else None  # TODO: set replace
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    if kwargs.get('prefetch') is not None:
        prefetch = kwargs.pop('prefetch')
        img_norm_cfg = kwargs.pop('img_norm_cfg')
    else:
        prefetch = False

    collate_fn = kwargs.pop('collate_fn', None)
    if collate_fn is None:
        collate_fn = dict(type='MMCVCollateFN', samples_per_gpu=imgs_per_gpu)
    collate_fn = build_collate_fn(collate_fn).get_collate()

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        worker_init_fn=worker_init_fn if seed is not None else None,
        **kwargs)

    if prefetch:
        data_loader = PrefetchLoader(data_loader, img_norm_cfg['mean'], img_norm_cfg['std'])

    return data_loader


def worker_init_fn(seed):
    np.random.seed(seed)
    random.seed(seed)


class PrefetchLoader:
    """
    A data loader wrapper for prefetching data
    """
    def __init__(self, loader, mean, std):
        self.loader = loader
        self._mean = mean
        self._std = std

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True
        self.mean = torch.tensor([x * 255 for x in self._mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in self._std]).cuda().view(1, 3, 1, 1)

        for next_input_dict in self.loader:
            with torch.cuda.stream(stream):
                data = next_input_dict['img'].cuda(non_blocking=True)
                next_input_dict['img'] = data.float().sub_(self.mean).div_(self.std)

            if not first:
                yield input
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input_dict

        yield input

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
