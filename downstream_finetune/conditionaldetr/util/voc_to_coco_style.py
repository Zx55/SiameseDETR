#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) Microsoft. All Rights Reserved
# ------------------------------------------------------------------------

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

ROOT_DIR = 'train'
IMAGE_DIR = os.path.join(ROOT_DIR, "shapes_train2018")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")
from datasets.voc import VOCDetection
INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

classes = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

classes_to_ind = dict(zip(classes, range(20)))

CATEGORIES = [{'id': v, 'name': k} for k, v in classes_to_ind.items()]

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def bbox2pmask(image, bbox):
    image_ = np.zeros(image.shape)
    image_[int(bbox[1]): int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2]), ] = 1
    binary_mask_ = image_.astype(np.int8)[:, :, 0]
    return binary_mask_

def main():

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    anno_id = 1

    test = VOCDetection('/path/to/voc', 'test', '2007', None,)  # test2007
    dataset = test
    from tqdm import tqdm, trange
    for ix in trange(len(dataset)):
        image, target, path = dataset[ix]
        relpath  = path
        image_id = int(target['image_id'])
        image_info = pycococreatortools.create_image_info(
            image_id, relpath, image.size)
        coco_output["images"].append(image_info)

        labels = target['labels'].numpy().tolist()
        difficults = target['difficult'].numpy().tolist()
        bboxs = target['boxes'].numpy().tolist()
        iscrowds = target['iscrowd'].numpy().tolist()

        for ix, bbox in enumerate(bboxs):
            category_info = {'id': labels[ix] , 'is_crowd':iscrowds[ix] , 'difficult': difficults[ix]}
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            bbox = [x1, y1, w, h]
            annotation_info = {
                "id": anno_id,
                "image_id": image_id,
                "category_id": category_info["id"],
                "iscrowd": category_info['is_crowd'],
                "area": w * h,
                "bbox": bbox,
                "segmentation": [],
                "width": image.size[0],
                "height": image.size[1],
                "difficult": category_info['difficult']
            }
            print(annotation_info)
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
            anno_id += 1

    with open(f'voc_coco_style_{dataset.image_set}_{dataset.year}.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print('dumped!')


if __name__ == "__main__":
    main()
