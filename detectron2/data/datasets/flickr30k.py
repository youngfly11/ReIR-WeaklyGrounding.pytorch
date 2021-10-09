#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/10/13 16:25

from fvcore.common.file_io import PathManager
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import global_cfg as cfg
import os.path as osp
import json
import pickle

__all__ = ["register_flickr30k"]

def load_flickr30k_instances(img_root: str, anno_dir:str, txt_path: str):

    """
    Load flickr30k annotations to Detectron2 format.

    Args:
        img_root: path to load images
        anno_dir: path to load annotation
        txt_path: image split path, one of "training, val, test"
    """

    with open(txt_path, 'r') as f:
        data_ids = f.readlines()

    fileids = [i.strip() for i in data_ids]
    dicts = []

    sent_anno = json.load(open(osp.join(anno_dir, 'sent_anno.json'), 'r'))
    box_anno = json.load(open(osp.join(anno_dir, 'box_anno.json'), 'r'))
    sg_anno = json.load(open(osp.join(anno_dir, 'sg_anno_with_rel_cate_v2.json'), 'r'))
    with open(osp.join(anno_dir, 'precomp_annos.pkl'), 'rb') as load_f:
        precomp_anno = pickle.load(load_f)

    ## to load the training/val/test data
    for fileid in fileids:

        fileid_split = fileid.split('\t')
        image_id = fileid_split[0]
        sent_id = fileid_split[1]

        img_name = osp.join(img_root, image_id+'.jpg') ## full path

        r = {
            "file_name": img_name, # full path
            "image_id": image_id,
            'sent_id': sent_id
        }

        # here we load the annotations.
        box_anno_img = box_anno[image_id]
        phrase_ids, gt_boxes = merge_gt_boxes(box_anno_img)
        sent_sg = sg_anno[image_id]['relations'][int(sent_id)]
        sentence = sent_anno[image_id][int(sent_id)]

        precomp_anno_img = precomp_anno[image_id]
        pre_box = precomp_anno_img['boxes']
        precomp_bbox = pre_box[:cfg.MODEL.VG.PRECOMP_TOPK,:4]
        precomp_score = pre_box[:cfg.MODEL.VG.PRECOMP_TOPK, 4]
        precomp_det_label = pre_box[:cfg.MODEL.VG.PRECOMP_TOPK, 5] - 1 ## ignore the background label
        image_scale = precomp_anno_img['img_scale']

        r['height'] = box_anno_img['height']
        r['width'] = box_anno_img['width']
        r['phrase_ids'] = phrase_ids
        r['gt_boxes'] = gt_boxes
        r['relations'] = sent_sg
        r['sentence'] = sentence
        r['precomp_bbox'] = precomp_bbox
        r['precomp_score'] = precomp_score
        r['precomp_det_label'] = precomp_det_label
        r['image_scale'] = image_scale
        dicts.append(r)

    del sent_anno
    del sg_anno
    del box_anno
    del precomp_anno
    return dicts


def merge_gt_boxes(box_anno):
    gt_boxes = []
    phrase_ids = []
    for k, v in box_anno['boxes'].items():
        phrase_ids.append(k)
        if len(v) == 1:
            gt_boxes.append(v[0])
        else:
            # when a phrase respond to multiple regions, we take the union of them as paper given
            v = np.array(v)
            box = [v[:, 0].min(), v[:, 1].min(), v[:, 2].max(), v[:, 3].max()]
            gt_boxes.append(box)
    gt_boxes = np.array(gt_boxes)
    return phrase_ids, gt_boxes


def register_flickr30k(name, img_root, pickle_path, txt_path):

    DatasetCatalog.register(name, lambda: load_flickr30k_instances(img_root, pickle_path, txt_path))
    MetadataCatalog.get(name).set(img_root=img_root, split=txt_path, evaluator_type="flickr30k")