# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torchvision
from pathlib import Path
import datasets.transforms as T
from datasets.coco import ConvertCocoPolysToMask


classes = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
classes_to_ind = dict(zip(classes, range(20)))


class VOCDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks=False):
        super(VOCDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(VOCDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def make_voc_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.root_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    if args.split is not None:
        train_meta = f'voc_trainval2012.cocostyle_split{args.split}.json'
    else:
        train_meta = f'voc_trainval2012.cocostyle.json'
    PATHS = {
        "train": (root / "VOCdevkit/VOC2012/JPEGImages/", root / "meta" / train_meta),
        "val": (root / "2007/VOCdevkit/VOC2007/JPEGImages/", root / "meta" / f'voc_test2007.cocostyle.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = VOCDetection(img_folder, ann_file, transforms=make_voc_transforms(image_set), return_masks=False)
    return dataset
