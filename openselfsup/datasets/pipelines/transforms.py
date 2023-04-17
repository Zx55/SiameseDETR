# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

import cv2
import inspect
import math
import numpy as np
from PIL import Image, ImageFilter
import random
import torch
from torchvision import transforms as _transforms

from openselfsup.utils import build_from_cfg
from openselfsup.utils.box_ops import box_xyxy_to_cxcywh
from ..registry import PIPELINES
from .helper import resize, hflip, crop, get_image_size, gen_two_boxes_in_outerbox_with_iou

# register all existing transforms in torchvision
_EXCLUDED_TRANSFORMS = ['GaussianBlur']
for m in inspect.getmembers(_transforms, inspect.isclass):
    if m[0] not in _EXCLUDED_TRANSFORMS:
        PIPELINES.register_module(m[1])


@PIPELINES.register_module
class RandomAppliedTrans(object):
    """Randomly applied transformations.

    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float): Probability.
    """

    def __init__(self, transforms, p=0.5):
        trans = []
        for t in transforms:
            if isinstance(t, dict):
                t = build_from_cfg(t, PIPELINES)
            else:
                pipeline_name = t.__class__.__name__
                assert pipeline_name in PIPELINES.module_dict.keys()
            trans.append(t)
        self.trans = _transforms.RandomApply(trans, p=p)

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


# custom transforms
@PIPELINES.register_module
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)."""

    _IMAGENET_PCA = {
        'eigval':
        torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec':
        torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }

    def __init__(self):
        self.alphastd = 0.1
        self.eigval = self._IMAGENET_PCA['eigval']
        self.eigvec = self._IMAGENET_PCA['eigvec']

    def __call__(self, img):
        assert isinstance(img, torch.Tensor), \
            "Expect torch.Tensor, got {}".format(type(img))
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class Solarization(object):
    """Solarization augmentation in BYOL https://arxiv.org/abs/2006.07733."""

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        img = np.array(img)
        img = np.where(img < self.threshold, img, 255 -img)
        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class ComposeWithTarget(object):
    def __init__(self, transforms):
        self.transforms = []
        for t in transforms:
            if isinstance(t, dict):
                t = build_from_cfg(t, PIPELINES)
            else:
                pipeline_name = t.__class__.__name__
                assert pipeline_name in PIPELINES.module_dict.keys()
            self.transforms.append(t)

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        repr_str = self.__class__.__name__ + "("
        for t in self.transforms:
            repr_str += "\n"
            repr_str += "    {0}".format(t)
        repr_str += "\n)"
        return repr_str


@PIPELINES.register_module
class ToTensorWithTarget(object):
    def __call__(self, img, target):
        return _transforms.functional.to_tensor(img), target

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class NormalizeWithTarget(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = _transforms.functional.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class RandomResizeWithTarget(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class RandomHorizontalFlipWithTarget(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class RandomSizeCropWithTarget(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = _transforms.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class RandomSelectWithTarget(object):
    """
    Randomly selects between transform1 and transform2,
    with probability p for transform1 and (1 - p) for transform2
    """
    def __init__(self, transform1, transform2, p=0.5):
        if isinstance(transform1, dict):
            transform1 = build_from_cfg(transform1, PIPELINES)
        else:
            pipeline_name = transform1.__class__.__name__
            assert pipeline_name in PIPELINES.module_dict.keys()

        if isinstance(transform2, dict):
            transform2 = build_from_cfg(transform2, PIPELINES)
        else:
            pipeline_name = transform2.__class__.__name__
            assert pipeline_name in PIPELINES.module_dict.keys()

        self.transform1 = transform1
        self.transform2 = transform2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transform1(img, target)
        return self.transform2(img, target)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class TwoViewRandomResizedCrop(object):
    """
    Crop the given PIL Image with random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.), 
                 interpolation=Image.BILINEAR, 
                 iou=0.8, 
                 views_bbox_stochastic=True):
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        self.iou = iou
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.views_bbox_stochastic = views_bbox_stochastic

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img1, img2):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img1, self.scale, self.ratio)
        outer_box = (j, i, j + w, i + h)
        box1, box2, intersection = gen_two_boxes_in_outerbox_with_iou(
            outer_box=outer_box, iou=self.iou, views_bbox_stochastic=self.views_bbox_stochastic)
        SIZE = (h, w)

        i1, j1, h1, w1 = box1[1], box1[0],  box1[3] - box1[1] , box1[2] - box1[0]
        i2, j2, h2, w2 = box2[1], box2[0],  box2[3] - box2[1] ,  box2[2] - box2[0]
        view1_o = (j1, i1)
        intersection_in_view1 = ((intersection[0] - view1_o[0]) * w / w1,
                                 (intersection[1] - view1_o[1]) * h / h1,
                                 (intersection[2] - view1_o[0]) * w / w1,
                                 (intersection[3] - view1_o[1]) * h / h1)
        view2_o = (j2, i2)
        intersection_in_view2 = ((intersection[0] - view2_o[0]) * w / w2,
                                 (intersection[1] - view2_o[1]) * h / h2,
                                 (intersection[2] - view2_o[0]) * w / w2,
                                 (intersection[3] - view2_o[1]) * h / h2)

        view1 = _transforms.functional.resized_crop(img1, i1, j1, h1, w1, SIZE, self.interpolation) # 目标size一样
        view2 = _transforms.functional.resized_crop(img2, i2, j2, h2, w2, SIZE, self.interpolation)

        intersection_in_origin = intersection
        return view1, (i1, j1, h1, w1), view2, (i2, j2, h2, w2), SIZE, \
            intersection_in_view1, intersection_in_view2, intersection_in_origin

    def __repr__(self):
        interpolate_str = str(self.interpolation)
        format_string = self.__class__.__name__
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


@PIPELINES.register_module
class SiameseRandomHorizontalFlipWithTarget(object):
    """
    Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target=None, is_flip=None):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if (random.random() < self.p and is_flip is None) or is_flip is True:
            if target is not None:
                w, h = img.size

                if "boxes" in target:
                    boxes = target["boxes"].clone()
                    boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
                    target["boxes"] = boxes

                if 'unsup_boxes' in target:
                    boxes = target["unsup_boxes"].clone()
                    boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
                    target["unsup_boxes"] = boxes

            return _transforms.functional.hflip(img), target, True

        return img, target, False

    def __repr__(self):
        return self.__class__.__name__ + f'(p = {self.p})'
