# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MMSelfSup (https://github.com/open-mmlab/mmselfsup)
# Copyright (c) OpenMMLab. All Rights Reserved
# ------------------------------------------------------------------------

import io
from PIL import Image
import numpy as np
from openselfsup.utils import print_log
try:
    import mc
except ImportError as E:
    pass


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    return Image.open(buff)


def np_loader(np_str):
    buff = io.BytesIO(np_str)
    return np.load(buff, allow_pickle=True)


def trim_key(key):
    key = key[9:]
    if len(key) >= 250:
        key = str(key).encode('utf-8')
        m = hashlib.md5()
        m.update(key)
        return "md5://{}".format(m.hexdigest())
    else:
        return key


class McLoader(object):

    def __init__(self, mclient_path, backend='mc'):
        assert mclient_path is not None, \
            "Please specify 'data_mclient_path' in the config."
        self.mclient_path = mclient_path
        assert backend in ['mc']
        self.backend = backend

        if self.backend == 'mc':
            server_list_config_file = f"{mclient_path}/server_list.conf"
            client_config_file = f"{mclient_path}/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(
                server_list_config_file, client_config_file)
            print_log(f'Initializing mc client...', 'root')
            print_log(f'mc server: {server_list_config_file}', 'root')
            print_log(f'mc client: {client_config_file}', 'root')
        else:
            raise RuntimeError(f'backend {self.backend} not supported')

    def get_item(self, fn):
        try:    
            img_value = mc.pyvector()
            self.mclient.Get(fn, img_value)
            img_value_str = mc.ConvertBuffer(img_value)
            img = pil_loader(img_value_str)
            return img
        except:
            print_log('Read image failed ({})'.format(fn))
            return None

    def get_np_item(self, fn):
        try:
            np_value = mc.pyvector()
            self.mclient.Get(fn, np_value)
            np_value_str = mc.ConvertBuffer(np_value)
            nparray = np_loader(np_value_str)
            return nparray
        except:
            print_log('Read Numpy failed ({})'.format(fn))
            return None
