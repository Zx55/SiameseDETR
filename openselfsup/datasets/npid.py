# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

from PIL import Image
from .registry import DATASETS
from .base import BaseDataset


@DATASETS.register_module
class NPIDDataset(BaseDataset):
    """Dataset for NPID.
    """

    def __init__(self, data_source, pipeline):
        super(NPIDDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img = self.pipeline(img)
        return dict(img=img, idx=idx)

    def evaluate(self, scores, keyword, logger=None):

        raise NotImplemented
