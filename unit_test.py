#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/11/22 19:13


import numpy as np
import torch.nn as nn
import torch
import os.path as osp
import os
import pickle



def extract_boxes():


    path_list = os.listdir('./flickr30k_datasets/flickr30k_feat_nms/flickr30k_torch_nms1e4_feat')

    precomp_annos = {}
    for img_id in path_list:

        feat_path = './flickr30k_datasets/flickr30k_feat_nms/flickr30k_torch_nms1e4_feat/{}'.format(img_id)

        with open(osp.join(feat_path), 'rb') as load_f:
            res = pickle.load(load_f)

        imgs = img_id.split('.')[0]
        bbox_data = res['boxes']
        img_scale = res['img_scale']

        precomp_annos[imgs] = {'boxes': bbox_data, 'img_scale': img_scale}
        print(img_id, 'done')

    with open('./flickr30k_datasets/flickr30k_anno/precomp_annos.pkl', 'wb') as dump_f:
        pickle.dump(precomp_annos, dump_f)

if __name__ == '__main__':

    extract_boxes()












