# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from typing import List, Tuple

from openselfsup.utils import build_from_cfg
from .registry import DATASETS, PIPELINES
from .builder import build_datasource


@DATASETS.register_module
class SiameseDETRDataset(Dataset):

    def __init__(self, 
                 data_source,
                 base_pipeline,
                 view_pipeline,
                 prefetch=False, 
                 anchor_num=100,
                 ratio_max=8, 
                 dice_num=300, 
                 area_min_ratio=0.1, 
                 iou=0.5,
                 gen_box_method='random',
                 views_bbox_stochastic=True,
                 return_crop=False,
                 crop_pipeline=None):
        """
        Args:
            data_source:
            base_pipeline:
            view_pipeline:
            prefetch:
            anchor_num:
            ratio_max:
            dice_num:
            area_min_ratio:
            iou:
            gen_box_method:  from_datasource / random / gt
            views_bbox_stochastic:
            return_crop: 是否需要返回box所要crop的图片
            crop_pipeline: crop的图片需要经过的PA
        """
        self.data_source = build_datasource(data_source)
        self.anchor_num = anchor_num
        self.ratio_max = ratio_max
        self.dice_num = dice_num
        self.area_min_ratio = area_min_ratio
        self.gen_box_method = gen_box_method
        assert self.gen_box_method in ('from_datasource', 'random')
        self.views_bbox_stochastic = views_bbox_stochastic
        self.iou = iou
        self.return_crop = return_crop
        self.prefetch = prefetch
        
        ComposeWithTarget = PIPELINES.get('ComposeWithTarget')
        self.base_pipeline = ComposeWithTarget([build_from_cfg(p, PIPELINES) for p in base_pipeline])
        self.view_pipeline = Compose([build_from_cfg(p, PIPELINES) for p in view_pipeline])
        self.flip_transform_with_target = PIPELINES.get('SiameseRandomHorizontalFlipWithTarget')()
        self.two_view_crop_transform = PIPELINES.get('TwoViewRandomResizedCrop')(
            scale=(0.5, 1.), iou=self.iou, views_bbox_stochastic=self.views_bbox_stochastic)

        if self.return_crop:
            assert isinstance(crop_pipeline, (List, Tuple))
            self.crop_pipeline = Compose([build_from_cfg(p, PIPELINES) for p in crop_pipeline])

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, index):
        try:
            sample, target = self.data_source.get_sample(index)
            sample, target = self.base_pipeline(sample, target)
        except:
            import random 
            print('raise exception in dataset, idx:', index)
            return self.__getitem__(random.randint(0, self.__len__() - 1))


        q, k = sample, sample
        q, target, is_flip = self.flip_transform_with_target(q, target)
        k, _, _ = self.flip_transform_with_target(k, None, is_flip=is_flip)  # flip 右图

        q, (t_q, l_q, h_q, w_q), k, (t_k, l_k, h_k, w_k), view_size, intersection_v1, \
            intersection_v2, intersection_raw = self.two_view_crop_transform(q, k)   #
        b_q, r_q = t_q + h_q, l_q + w_q
        b_k, r_k = t_k + h_k, l_k + w_k
        q_coord_raw, k_coord_raw = (l_q, t_q, r_q, b_q), (l_k, t_k, r_k, b_k)
        view_height, view_width = view_size  # view size is defined by two_view_crop_transform

        anchors_q, anchors_k, anchors_labels = self.gen_boxes(
            target, 
            q_coord_raw, 
            k_coord_raw, 
            view_width, 
            view_height, 
            intersection_v1, 
            intersection_raw)
        try:
            anchors_q, anchors_k, anchors_labels = self.pick_boxes(
                anchors_q,
                anchors_k,
                anchors_labels,
                q_coord_raw,
                k_coord_raw,
                view_width,
                view_height,
                intersection_v1,
                intersection_v2)
        except ValueError:
            return self[index - 1]

        if self.return_crop:
            qcrops, kcrops = [], []
            for boxix in range(len(anchors_q)):
                boxq, boxk = anchors_q[boxix], anchors_k[boxix]
                cropq, cropk = q.crop(boxq), k.crop(boxk)
                cropq, cropk = self.crop_pipeline(cropq), self.crop_pipeline(cropk)
                qcrops.append(cropq)
                kcrops.append(cropk)
            qcrops = torch.stack(qcrops, dim=0) # M x 3 x 128 x 128
            kcrops = torch.stack(kcrops, dim=0) # M x 3 x 128 x 128
            crops = torch.stack((qcrops, kcrops), dim=0)  # 2 x nboxs x 3 x 128 x 128

        q, k = self.view_pipeline(q), self.view_pipeline(k)
        box = np.stack((anchors_q, anchors_k), axis=0)
        img = torch.stack((q, k), dim=0)
        box = torch.from_numpy(box).float()
        img_size = torch.FloatTensor(view_size)  # H, W

        result = dict(img=img, box=box, img_size=img_size)
        if self.return_crop:
            result['crop'] = crops  # 2 x nboxs x 3 x 128 x 128
        return result

    @staticmethod
    def anchor_mapping_np(anchor, loc_q, loc_k, image_height=224., image_width=224.):
        """
        Args:
            anchor: (x0, y0, x1, y1) q图中的box的坐标，是在q图的坐标系下
            loc_q: (x0, y0, x1, y1) q图在原全图的坐标
            loc_k: (x0, y0, x1, y1) k图在原全图的坐标
            img_size (float): Image size q图resize成的大小，也就是view的最终size
        Returns:
            tuple: (x0, y0, x1, y1) if transformed region falls into the boundary of image k, or (-1, -1, -1, -1) if not
        """
        x0, y0, x1, y1 = anchor
        l_q, t_q, r_q, b_q = loc_q
        l_k, t_k, r_k, b_k = loc_k
        # Step 1: transfer anchor's coordinates (relative to q) to absolute
        l_a = (r_q - l_q) * x0 / image_width + l_q
        r_a = (r_q - l_q) * x1 / image_width + l_q
        t_a = (b_q - t_q) * y0 / image_height + t_q
        b_a = (b_q - t_q) * y1 / image_height + t_q
        # Step 2: transfer anchor's absolute coordinates to relative to k
        x0 = (l_a - l_k) * image_width / (r_k - l_k)
        x1 = (r_a - l_k) * image_width / (r_k - l_k)
        y0 = (t_a - t_k) * image_height / (b_k - t_k)
        y1 = (b_a - t_k) * image_height / (b_k - t_k)
        # Step 3: check whether it's out of boundary
        mask = (x0 < 0) | (y0 < 0) | (x1 >= image_width) | (y1 >= image_height)
        x0[mask] = -1
        y0[mask] = -1
        x1[mask] = -1
        y1[mask] = -1
        return x0, y0, x1, y1

    def gen_boxes(self, 
                  target, 
                  q_coord_raw, 
                  k_coord_raw, 
                  view_width, 
                  view_height, 
                  intersection_v1, 
                  intersection_raw):
        if self.gen_box_method == 'random':
            anchors_q, anchors_k, anchors_labels = self.gen_random_boxes(
                intersection_v1=intersection_v1,
                q_coord_raw=q_coord_raw,
                k_coord_raw=k_coord_raw,
                dice_num=self.dice_num,
                ratio_max=self.ratio_max,
                image_width=view_width,
                image_height=view_height,
                area_min_ratio=self.area_min_ratio)

        elif self.gen_box_method == 'from_datasource':
            assert 'unsup_boxes' in target
            unsup_boxes_raw = target['unsup_boxes'].to(torch.float32)
            anchors_q, anchors_k, anchors_labels = self.clip_and_remove_out_box(
                unsup_boxes_raw,
                q_coord_raw,
                k_coord_raw,
                view_width,
                view_height,
                intersection_v1,
                intersection_raw)

        else:
            raise ValueError(f'gen_box_method {self.gen_box_method} not supported')

        return anchors_q, anchors_k, anchors_labels

    def gen_random_boxes(self,
                         intersection_v1,
                         q_coord_raw,
                         k_coord_raw,
                         ratio_max=8,
                         dice_num=1000,
                         area_min_ratio=0.1,
                         image_height=224,
                         image_width=224):
        rx0, ry0, rx1, ry1 = intersection_v1
        area_inter = (ry1 - ry0) * (rx1 - rx0)
        min_area = area_min_ratio * area_inter

        y0, x0 = np.random.randint(ry0, ry1, (dice_num,)), np.random.randint(rx0, rx1, (dice_num,))
        height_max = ry1 - y0
        width_max = rx1 - x0

        heights = np.random.rand(dice_num) * height_max + 1e-10
        width_mins = min_area / heights
        widths = width_mins + np.random.rand(dice_num) * (width_max - width_mins)

        y1 = y0 + heights
        x1 = x0 + widths
        mask_q_x1_in = x1 < rx1

        anchors_q = np.stack((x0, y0, x1, y1), axis=-1)
        x0, y0, x1, y1 = self.anchor_mapping_np(
            (x0, y0, x1, y1),
            q_coord_raw,
            k_coord_raw,
            image_width=image_width, 
            image_height=image_height)

        anchors_k = np.stack((x0, y0, x1, y1), axis=-1)
        anchors_labels = np.zeros((dice_num,))
        anchors_labels[x0 == -1] = -1

        wq, hq = anchors_q[:, 2] - anchors_q[:, 0], anchors_q[:, 3] - anchors_q[:, 1]
        wk, hk = anchors_k[:, 2] - anchors_k[:, 0], anchors_k[:, 3] - anchors_k[:, 1]

        mask_ratio_q = (wq / (hq + 1e-10) >= 1 / ratio_max) & (wq / (hq + 1e-10) <= ratio_max)
        mask_ratio_k = (wk / (hk + 1e-10) >= 1 / ratio_max) & (wk / (hk + 1e-10) <= ratio_max)
        mask_area_k = (wk * hk) >= min_area
        mask = mask_ratio_q & mask_ratio_k & mask_area_k & mask_q_x1_in
        anchors_labels[~mask] = -1
        return anchors_q, anchors_k, anchors_labels

    def clip_and_remove_out_box(self, boxes,
                                 q_coord_raw,
                                 k_coord_raw,
                                 image_width,
                                 image_height,
                                 intersection_v1,
                                 intersection_raw):
        (l_q, t_q, r_q, b_q), (l_k, t_k, r_k, b_k) = q_coord_raw, k_coord_raw

        # 把原图的box clamp 到 common region中来
        irx0, iry0, irx1, iry1 = intersection_raw
        boxes[:, 0].clamp_(min=irx0)
        boxes[:, 1].clamp_(min=iry0)
        boxes[:, 2].clamp_(max=irx1 - 1)
        boxes[:, 3].clamp_(max=iry1 - 1)

        boxes[:, 0::2] -= l_q  # 把坐标变换到 view1的resize之后的图片中来
        boxes[:, 1::2] -= t_q
        boxes[:, 0::2] *= (image_width / (r_q - l_q))
        boxes[:, 1::2] *= (image_height / (b_q - t_q))

        # 对面积进行筛选
        rx0, ry0, rx1, ry1 = intersection_v1
        area_inter = (ry1 - ry0) * (rx1 - rx0)
        area_boxes = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1)
        area_mask = area_boxes >= self.area_min_ratio * area_inter  # 包括面积=0的, 也就是不在公共视野的（被挤压成一个点））
        bound_mask = (boxes >= 0).all(dim=-1)
        coord_mask = (boxes[:, 2:] >= boxes[:, :2]).all(dim=-1)
        mask = area_mask & bound_mask & coord_mask
        anchors_q = boxes[mask]

        # 把这些所有的view1的框映射到另一个view下
        x0, y0, x1, y1 = anchors_q[:, 0], anchors_q[:, 1], anchors_q[:, 2], anchors_q[:, 3]
        x0, y0, x1, y1 = self.anchor_mapping_np(
            (x0, y0, x1, y1),
            q_coord_raw,
            k_coord_raw,
            image_width=image_width,
            image_height=image_height)
        anchors_k = torch.stack((x0, y0, x1, y1), dim=-1)
        anchors_labels = torch.zeros(anchors_k.shape[0])
        anchors_labels[x0 == -1] = -1
        return anchors_q, anchors_k, anchors_labels

    def pick_boxes(self,
                   anchors_q,
                   anchors_k,
                   anchors_labels,
                   q_coord_raw,
                   k_coord_raw,
                   view_width,
                   view_height,
                   intersection_v1,
                   intersection_v2):
        anchors_q = anchors_q[anchors_labels != -1]
        anchors_k = anchors_k[anchors_labels != -1]
        anchors_labels = anchors_labels[anchors_labels != -1]

        if len(anchors_labels) == 0:
            raise ValueError
        if len(anchors_labels) < self.anchor_num - 1:
            anchors_q, anchors_k, anchors_labels = self.pad_jitter_box(
                anchors_q,
                anchors_k,
                intersection_v1,
                q_coord_raw,
                k_coord_raw,
                view_width,
                view_height)
        else:
            anchors_q, anchors_k, anchors_labels = \
                anchors_q[:self.anchor_num - 1], \
                anchors_k[:self.anchor_num - 1], \
                anchors_labels[:self.anchor_num - 1]

        anchors_q_inter = np.array([intersection_v1])  # 加上intersection框
        anchors_k_inter = np.array([intersection_v2])  # 加上intersection框
        anchors_labels_inter = np.array([0])

        anchors_q = np.concatenate((anchors_q_inter, anchors_q,), axis=0)
        anchors_k = np.concatenate((anchors_k_inter, anchors_k,), axis=0)
        anchors_labels = np.concatenate((anchors_labels_inter, anchors_labels,), axis=0)

        anchors_q = anchors_q[:self.anchor_num]
        anchors_k = anchors_k[:self.anchor_num]
        anchors_labels = anchors_labels[:self.anchor_num]

        anchors_q = np.array(anchors_q, dtype=np.float)
        anchors_k = np.array(anchors_k, dtype=np.float)
        anchors_labels = np.array(anchors_labels, dtype=np.float)  # unused
        return anchors_q, anchors_k, anchors_labels

    def pad_jitter_box(self, 
                       anchors_q, 
                       anchors_k, 
                       intersection_v1, 
                       q_coord_raw, 
                       k_coord_raw, 
                       image_width, 
                       image_height):
        valid = len(anchors_q)
        need = self.anchor_num - 1 - valid 
        suffix_q = torch.cat([anchors_q.clone() for i in range(math.ceil(need / float(len(anchors_q))))], dim=0)
        suffix_q = suffix_q[:need]

        w, h = (suffix_q[:, 2] - suffix_q[:, 0]).view(-1, 1), (suffix_q[:, 3] - suffix_q[:, 1]).view(-1, 1)
        randomness = torch.randn((w.shape[0], 4)).sigmoid() 
        randomness = randomness * 2 * 0.1 - 0.1  # +-20%
        addon = torch.cat([w, h, w, h], dim=-1) * randomness
        suffix_q = suffix_q + addon
        
        suffix_q[:, 0::2].clamp_(intersection_v1[0], intersection_v1[2] - 1)  # 不出界，映射之后也不出界
        suffix_q[:, 1::2].clamp_(intersection_v1[1], intersection_v1[3] - 1)
        x0, y0, x1, y1 = suffix_q[:, 0], suffix_q[:, 1], suffix_q[:, 2], suffix_q[:, 3]
        x0, y0, x1, y1 = self.anchor_mapping_np(
            (x0, y0, x1, y1),
            q_coord_raw,
            k_coord_raw,
            image_width=image_width,
            image_height=image_height)

        suffix_k = torch.stack((x0, y0, x1, y1), dim=-1)
        anchors_q = torch.cat((anchors_q, suffix_q), dim=0)
        anchors_k = torch.cat((anchors_k, suffix_k), dim=0)
        anchors_labels = torch.zeros(anchors_k.shape[0])
        return anchors_q, anchors_k, anchors_labels

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplementedError
