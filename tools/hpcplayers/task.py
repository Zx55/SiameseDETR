# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from hpcplayer import HPCPlayer  # noqa
from box_generator import BoxGenerator
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--root', default=None, type=str)
parser.add_argument('--source', default=None, type=str)
parser.add_argument('--save_dir', default=None, type=str)
parser.add_argument('--dataset', default=None, type=str, choices=['imagenet', 'coco'])


class Uploader(HPCPlayer):
    def __init__(self, root=None, meta_file=None, save_dir=None, dataset=None):
        assert root is not None
        assert meta_file is not None
        if save_dir is None:
            save_dir = 'edge_box/'
        if dataset is None:
            dataset = 'imagenet'

        logfile = f'logs/edgebox.{dataset}.log'

        os.makedirs('logs', exist_ok=True)
        super(Uploader, self).__init__(root=root, source=meta_file, logfile=logfile)
        self.root = root
        self.box_generator = BoxGenerator(dataset=dataset, save_dir=save_dir)

    def core(self, sourceLine):
        try:
            path = os.path.join(self.root, sourceLine.split()[0])
            self.box_generator.process(path, edge=True, ss=False)
        except Exception as e:
            return sourceLine + str(e)

        return 'OK'


if __name__ == '__main__':
    args = parser.parse_args()
    uploader = Uploader(args.root, args.source, args.save_dir, args.dataset)
    uploader.run()
