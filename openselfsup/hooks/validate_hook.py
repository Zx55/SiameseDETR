# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

from mmcv.runner import Hook

import torch
from torch.utils.data import Dataset

from openselfsup import datasets
from openselfsup.utils import nondist_forward_collect, dist_forward_collect
from .registry import HOOKS


@HOOKS.register_module
class ValidateHook(Hook):
    """Validation hook.

    Args:
        dataset (Dataset | dict): A PyTorch dataset or dict that indicates
            the dataset.
        dist_mode (bool): Use distributed evaluation or not. Default: True.
        initial (bool): Whether to evaluate before the training starts.
            Default: True.
        interval (int): Evaluation interval (by epochs). Default: 1.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataset,
                 dist_mode=True,
                 initial=True,
                 interval=1,
                 **eval_kwargs):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))

        self.data_loader = datasets.build_dataloader(
            self.dataset,
            eval_kwargs['imgs_per_gpu'],
            eval_kwargs['workers_per_gpu'],
            dist=dist_mode,
            shuffle=False,
            prefetch=eval_kwargs.get('prefetch', False),
            img_norm_cfg=eval_kwargs.get('img_norm_cfg', dict()),
            collate_fn=eval_kwargs.get('collate_fn', None)
        )
        self.dist_mode = dist_mode
        self.initial = initial
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.data_mode = eval_kwargs.get('data_mode', 'default')

    def before_run(self, runner):
        if self.initial:
            self._run_validate(runner)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        self._run_validate(runner)

    def _run_validate(self, runner):
        runner.model.eval()
        func = lambda **x: runner.model(mode='test', **x)
        if self.dist_mode:
            results = dist_forward_collect(
                func, self.data_loader, runner.rank,
                len(self.dataset))  # dict{key: np.ndarray}
        else:
            results = nondist_forward_collect(func, self.data_loader, len(self.dataset))
        if runner.rank == 0:
            if self.data_mode == 'default':
                for name, val in results.items():
                    self._evaluate_default(runner, torch.from_numpy(val), name)
            elif self.data_mode == 'coco':
                self._evaluate_default(runner, results, None)
            elif self.data_mode == 'eval_with_runner':
                self._evaluate_with_runner(runner, results, None)
            else:
                raise ValueError(f'data_mode {self.data_mode} not supported')
        runner.model.train()

    def _evaluate_default(self, runner, results, keyword):
        eval_res = self.dataset.evaluate(
            results,
            keyword=keyword,
            logger=runner.logger,
            **self.eval_kwargs['eval_param'])
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val


        runner.log_buffer.ready = True

    def _evaluate_with_runner(self, runner, results, keyword):
        eval_res = self.dataset.evaluate(
            results,
            keyword=keyword,
            logger=runner.logger,
            runner=runner,
            **self.eval_kwargs['eval_param'])
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val


        runner.log_buffer.ready = True