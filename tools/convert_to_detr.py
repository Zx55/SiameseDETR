# ------------------------------------------------------------------------
# Siamese DETR
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, type=str)
parser.add_argument('--out', type=str, default='./')
parser.add_argument('--deform', action='store_true', default=False)


def main():
    args = parser.parse_args()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError
    ckpt = torch.load(args.ckpt, 'cpu')
    state_dict = ckpt['state_dict']
  
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    new_state_dict = {}

    valid_names = ['transformer', 'query_embed', 'input_proj', 'patch2query']
    num_err = 0
    for name, p in state_dict.items():
        flag = False
        new_name = ''
        for n in valid_names:
            if name.startswith(n):
                flag = True
                new_name = name
                new_state_dict[name] = p.clone()
                break

        if name.startswith('head'):
            flag = True
            new_name = name[5:]
            if args.deform and 'box_embed' in name:
                for i in range(6):
                    new_name_ = new_name.replace('box_embed', f'box_embed.{i}')  # deformable多层box embed
                    print(name, '->', new_name_)
                    new_state_dict[new_name_] = p.clone()
            else:
                new_state_dict[new_name] = p.clone()
        if name.startswith('backbone'):
            flag = True
            new_name = f'backbone.0.body.{name[9:]}'
            new_state_dict[new_name] = p.clone()

        if name.startswith('pred_head.bbox_embed'):
            flag = True
            new_name = f'{name[10:]}'
            if args.deform:
                for i in range(6):
                    new_name_ = new_name.replace('box_embed', f'box_embed.{i}')  # deformable多层box embed
                    print(name, '->', new_name_)
                    new_state_dict[new_name_] = p.clone()
            else:
                new_state_dict[new_name] = p.clone()

        if args.deform:
            if name.startswith('pred_head.class_embed'):
                flag = True
                new_name = f'{name[10:]}'
                for i in range(6):
                    new_name_ = new_name.replace('class_embed', f'class_embed.{i}')  # 最终不会用到
                    print(name, '->', new_name_)
                    new_state_dict[new_name_] = p.clone()
        if flag:
            print(f'{name} -> {new_name}')
        else:
            num_err += 1
            print('[ERR] mismatch keys in ckpt:', name)
    
    if not os.path.exists(args.out):
        os.makedirs(args.out, exist_ok=True)
    ckpt_name = os.path.split(args.ckpt)[-1]
    ckpt_name = f'detr_{ckpt_name}'
    torch.save({'model': new_state_dict}, os.path.join(args.out, ckpt_name))
    print(f'done{", with {} error(s)".format(num_err) if num_err != 0 else ""}')


if __name__ == '__main__':
    main()
