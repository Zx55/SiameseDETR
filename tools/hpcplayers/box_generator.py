# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import cv2
import numpy as np
import os.path as osp
import os
import math


class BoxGenerator():
    def __init__(self, save_dir, num_box=512, dataset='coco', edge_model='model.yml.gz'):
        self.edge_detection = cv2.ximgproc.createStructuredEdgeDetection(edge_model)
        self.num_box = num_box
        self.save_dir = save_dir
        self.dataset = dataset

        if self.dataset == 'coco':
            self.get_rel_path = self.get_coco_rel_path
        elif self.dataset == 'imagenet':
            self.get_rel_path = self.get_imagenet_rel_path
        else:
            raise NotImplementedError(dataset)

    @staticmethod
    def get_coco_rel_path(path):
        last_2_level = '/'.join(path.split('/')[-2:])
        npy = 'coco/' + last_2_level.split('.')[0] + '.npy'

        return npy

    @staticmethod
    def get_imagenet_rel_path(path):
        last_2_level = '/'.join(path.split('/')[-3:])
        npy = 'imagenet/' + last_2_level.split('.')[0] + '.npy'
        return npy

    @staticmethod
    def read_image(image_path):
        return cv2.imread(image_path)

    def selective_search(self, bgr_img, res_size=128):
        img_det = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        if res_size is not None:
            img_det = cv2.resize(img_det, (res_size, res_size))

        h, w = bgr_img.shape[:2]
        ss.setBaseImage(img_det)
        ss.switchToSelectiveSearchFast()
        boxes = ss.process().astype('float32')

        if res_size is not None:
            boxes /= res_size
            boxes *= np.array([w, h, w, h])

        boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
        boxes[..., 3] = boxes[..., 1] + boxes[..., 3]  # xyxy

        return boxes[:self.num_box]

    def sharpen(self, bgr_img):
        ori_shape = bgr_img.shape[:2]
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img_det = cv2.filter2D(src=bgr_img, ddepth=-1, kernel=kernel)
        alpha = 1.5  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        img_det = cv2.convertScaleAbs(img_det, alpha=alpha, beta=beta)

        factor = math.ceil(1000.0 / min(bgr_img.shape[:2]))

        img_det = cv2.resize(img_det, (ori_shape[1] * factor, ori_shape[0] * factor))

        return img_det, factor

    def _edgebox(self, bgr_img):
        rgb_im = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        edges = self.edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

        orimap = self.edge_detection.computeOrientation(edges)
        edges = self.edge_detection.edgesNms(edges, orimap)

        edge_boxes = cv2.ximgproc.createEdgeBoxes()
        edge_boxes.setMaxBoxes(self.num_box)
        boxes, scores = edge_boxes.getBoundingBoxes(edges, orimap)
        return boxes, scores

    def center(self, bgr_img):
        h, w, c = bgr_img.shape
        boxes = np.array([[w // 4, h // 4, w // 4 * 3, h // 4 * 3]])
        scores = np.array([[0.]])
        return boxes, scores

    def edgebox(self, bgr_img):

        try:
            boxes, scores = self._edgebox(bgr_img)
        except Exception as e:
            boxes, scores = (), ()

        factor = 1.0

        if len(boxes) == 0:  # 300+ images, they are all smooth image without much edges
            bgr_img_, factor = self.sharpen(bgr_img) # contrast & shape
            boxes, scores = self._edgebox(bgr_img_)

        if len(boxes) == 0:  # 1 image :train/n04347754/n04347754_18877.JPEG
            h, w, c = bgr_img.shape
            boxes = np.array([[w // 4, h // 4, w // 4 * 3, h // 4 * 3]])
            scores = np.array([[0.]])
            factor = 1

        boxes = (boxes / factor).astype(np.long)
        boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
        boxes[..., 3] = boxes[..., 1] + boxes[..., 3]  # xyxy

        # save image for checking
        # for box_ix in range(min(len(boxes), 10)):
        #     box = boxes[box_ix]
        #     start_point = (box[0], box[1])
        #     end_point = (box[2], box[3])
        #     color = (255, 0, 0)
        #     thickness = 2
        #     bgr_img = cv2.rectangle(bgr_img, start_point, end_point, color, thickness)

        # cv2.imwrite('save/' + str(random.random()) + '.jpg', bgr_img)

        return boxes, scores

    def process(self, img_path, ss=False, edge=True):
        if (ss or edge) is not True:
            raise ValueError

        bgr_image = self.read_image(img_path)
        rel_path = self.get_rel_path(img_path)
        if edge:
            boxes_edge, scores_edge = self.edgebox(bgr_img=bgr_image)
            edgebox_path = self.save_dir + '/edgebox/' + rel_path
            np_edge = dict(box=boxes_edge, score=scores_edge)
            os.makedirs(osp.dirname(edgebox_path), exist_ok=True)
            with open(edgebox_path, 'wb') as f:
                np.save(f, np_edge)

        if ss:
            boxes_ss = self.selective_search(bgr_img=bgr_image)
            selective_path = self.save_dir + '/selective_search/' +  rel_path
            np_ss = dict(box=boxes_ss)
            os.makedirs(osp.dirname(selective_path), exist_ok=True)
            with open(selective_path, 'wb') as f:
                np.save(f, np_ss)

