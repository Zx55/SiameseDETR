# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

import math
import random
import torch
import torchvision
import torchvision.transforms.functional as F
from typing import Dict

if float(torchvision.__version__.split(".")[1]) < 7.0:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

try:
    from torchvision.transforms.transforms import _get_image_size as get_image_size
except:
    from torchvision.transforms.functional import _get_image_size as get_image_size

__all__ = [
    'interpolate', 
    'crop', 
    'resize', 
    'hflip', 
    'get_image_size', 
    'gen_two_boxes_in_outerbox_with_iou']


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners)

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image
    if not isinstance(target, Dict):
        return rescaled_image, None  # for imagenet

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if 'unsup_boxes' in target:
        boxes = target["unsup_boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["unsup_boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def gen_two_boxes_in_outerbox_with_iou(outer_box, iou=0.5, views_bbox_stochastic=True):
    """
    作用：在外接矩形内，生成两个box，他们之间的IoU满足参数要求
    流程：
          1. 在矩形中心，生成一个矩形，这个矩形的aspect ratio和外接矩形一样，它作为Intersection
          2. 以该Intersection为边界，找到外接矩形和它之间的两个box
          3. 找到一个Intersectoin的解，使得他们之间的IoU符合参数要求
          4. 如果允许抖动(views_bbox_stochastic)：两个box的union部分向内缩进，使得Union减小，等价于IoU变大
          5. 如果可能，交换两box

    :param outer_box: (x1, x2, y1, y2)
    :param iou:  float, 0 ~ 1
    :param views_bbox_stochastic: jitter two box to have higher IoU
    :return: bbox1 bbox2
    """
    assert iou >= 0 and iou <= 1
    outer_x1, outer_y1, outer_x2, outer_y2 = outer_box
    outer_yc = (outer_y1 + outer_y2) / 2
    outer_xc = (outer_x1 + outer_x2) / 2

    outer_h = outer_y2 - outer_y1
    outer_w = outer_x2 - outer_x1
    aspect_ratio = outer_w / outer_h

    # 一元二次方程
    a = (aspect_ratio * iou + 2 * aspect_ratio)
    b = -(aspect_ratio * outer_h + outer_w) * iou
    c = -(outer_h * outer_w * iou)

    inter_h = (- b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)  # 取正解
    inter_w = aspect_ratio * inter_h

    inter_top_left = (outer_xc - inter_w / 2, outer_yc - inter_h / 2)
    inter_bottom_right = (outer_xc + inter_w / 2, outer_yc + inter_h / 2)

    box1, box2 = [outer_x1, outer_y1, *inter_bottom_right], [*inter_top_left, outer_x2, outer_y2]  # \，取左上和右下

    if views_bbox_stochastic:
        # 左上角的box，左上角向内收缩， 单调降低union，等价于提升IoU
        delta_wmax = inter_top_left[0] - outer_x1
        delta_hmax = inter_top_left[1] - outer_y1

        y_delta = random.uniform(0, min(delta_wmax / aspect_ratio, delta_hmax))  # 保持 aspect ratio, 并且保证不出界
        x_delta = aspect_ratio * y_delta
        box1[0] += x_delta
        box1[1] += y_delta

        # 右下角的box，右下角向内收缩， 单调降低union，等价于提升IoU
        delta_wmax = outer_x2 - inter_bottom_right[0]
        delta_hmax = outer_y2 - inter_bottom_right[1]

        y_delta = random.uniform(0, min(delta_wmax / aspect_ratio, delta_hmax))  # 保持 aspect ratio, 并且保证不出界
        x_delta = aspect_ratio * y_delta
        box2[2] -= x_delta
        box2[3] -= y_delta

    if random.random() > 0.5:
        box1, box2 = box2, box1

    intersection = [*inter_top_left, *inter_bottom_right]

    return box1, box2, intersection
    