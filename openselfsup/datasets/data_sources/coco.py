# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

import os
from PIL import Image
from pycocotools import mask as coco_mask
import torch
import torchvision

from ..registry import DATASOURCES
from .utils import McLoader
import numpy as np


@DATASOURCES.register_module
class Coco(torchvision.datasets.CocoDetection):

    def __init__(self, root, ann_file, memcached, mclient_path, return_ann=True, 
                 return_masks=False, *args, **kwargs):
        super().__init__(root, ann_file)
        self.return_ann = return_ann
        self.preprocess = ConvertCocoPolysToMask(return_masks)

        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            assert self.mclient_path is not None
            self.mc_loader = McLoader(self.mclient_path)
            self.initialized = True

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        fn = os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])

        if self.memcached:
            self._init_memcached()
            img = self.mc_loader.get_item(fn)
        else:
            img = Image.open(fn)
        img = img.convert('RGB')
        return img, target

    def get_sample(self, idx):
        img, target = self.__getitem__(idx)

        if self.return_ann:
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            img, target = self.preprocess(img, target)
            return img, target

        return img

    def get_length(self):
        return super().__len__()


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


@DATASOURCES.register_module
class CocoWithUnsupbox(Coco):
    def __init__(self, root, ann_file, memcached, mclient_path, return_ann=True,
                 return_masks=False, unsup_boxes_root='data/datasets/edgebox/coco', *args, **kwargs):
        super(CocoWithUnsupbox, self).__init__(root, ann_file, memcached, mclient_path, return_ann=return_ann,
                 return_masks=return_masks, *args, **kwargs)
        self.unsup_boxes_root = unsup_boxes_root

    def __get_np_item__(self, index):
        img_id = self.ids[index]
        image_rel_path = self.coco.loadImgs(img_id)[0]['file_name']
        numpy_rel_path = image_rel_path.split('.')[0] + '.npy'
        fn_np = os.path.join(self.unsup_boxes_root, numpy_rel_path)
        if self.memcached:
            self._init_memcached()
            nparray = self.mc_loader.get_np_item(fn_np)
        else:
            nparray = np.load(fn_np, allow_pickle=True)

        return nparray

    def get_sample(self, idx):
        img, target = self.__getitem__(idx)
        nparray = self.__get_np_item__(idx)
        unsup_boxes = torch.from_numpy(nparray.tolist()['box'])

        if self.return_ann:
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            img, target = self.preprocess(img, target)
            target['unsup_boxes'] = unsup_boxes
            return img, target

        target = dict()
        target['unsup_boxes'] = unsup_boxes
        return img, target


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target
