#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/10/13 16:25


from fvcore.common.file_io import PathManager
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
import os.path as osp
import pickle
import json
from detectron2.config import global_cfg as cfg



__all__ = ["register_refcoco"]


def load_refcoco_instances(img_root:str, pickle_path:str, txt_path:str, precomp_anns_path:str):

    """
    Load refcoco annotations to Detectron2 format.

    Args:
        img_root: path to load images
        pickle_path: path to load annotation
        txt_path: image split path, one of "training, val, test, testA, testB, testC"
    """


    with open(txt_path, 'r') as f:
        data_ids = f.readlines()

    fileids = [i.strip() for i in data_ids]

    """
    {img_id: {ref_id: {"cate": xxx,
    "sents": {sent_id: {"tokens": [xxx, xxx, xx]-> list, "sent": "xxxxxx",
                        "scenegraph":{'entities':[[xx,xx], [xxx,xxx]], 'relations:'[(0,1,[1]), (1,0,[2])]}
    }}}}}
    """
    with PathManager.open(osp.join(pickle_path), 'rb') as f:
        anno = pickle.load(f)

    with open(precomp_anns_path, 'rb') as load_f:
        precomp_anno = pickle.load(load_f)

    with open('./flickr30k_datasets/referring_annos/all_images_hw.pkl', 'rb') as load_f:
        all_image_hw = pickle.load(load_f)

    with open(cfg.MODEL.VG.ATTR_DICT_PATH, 'r') as load_f:
        ref_attr = json.load(load_f)

    with open(cfg.MODEL.VG.ATTR_VOCAB_PATH, 'r') as load_f:
        attr_vocab = json.load(load_f)['att2it']

    # attr_vocab = {ak:i for i, ak in enumerate(list(attr_vocab.keys()))}
    num_attr = len(list(attr_vocab.keys()))


    dicts = []

    for fileid in fileids:
        fileid_split = fileid.strip().split(',')
        img_name = osp.join(img_root, fileid_split[0])

        img_id = fileid_split[1]
        ref_id = fileid_split[2]
        expr_id = fileid_split[3]

        r = {
            "file_name": img_name, # full path
            'image_id': img_id,
            "sent_id": ref_id+'_'+expr_id
        }

        # Here we do not contain the image_width and height
        ## this two info can be inferred from mask
        anno_img_ref = anno[img_id][ref_id]
        expr_annos = anno_img_ref['sents'][expr_id]

        precomp_anno_img = precomp_anno[fileid_split[0].replace('COCO_train2014_', '')]
        pre_box = precomp_anno_img['boxes']
        precomp_bbox = pre_box[:cfg.MODEL.VG.PRECOMP_TOPK, :4]
        precomp_score = pre_box[:cfg.MODEL.VG.PRECOMP_TOPK, 4]
        precomp_det_label = pre_box[:cfg.MODEL.VG.PRECOMP_TOPK, 5]  ## in coco dataset, 81 cate is the background class

        ## input the attr label
        attr_rid = ref_attr.get(ref_id)
        attr_lab = generate_attr_label(attr_rid, attr_vocab, num_attr)

        r['width'] = all_image_hw[fileid_split[0].replace('COCO_train2014_', '')]['width']
        r['height'] = all_image_hw[fileid_split[0].replace('COCO_train2014_', '')]['height']
        r['sentence'] = expr_annos['tokens']
        r['cate'] = anno_img_ref['cate']
        r['nouns'] = expr_annos['nouns']
        r['noun_phr'] = expr_annos['noun_phr']
        r['image_scale'] = precomp_anno_img['img_scale']
        r['precomp_bbox'] = precomp_bbox
        r['precomp_score'] = precomp_score
        r['precomp_det_label'] = precomp_det_label
        r['attr_lab'] = attr_lab
        gt_boxes = np.array(anno_img_ref['bbox'])
        gt_boxes[2:] += gt_boxes[:2]
        r['gt_boxes'] = gt_boxes[None, :]
        dicts.append(r)

    del anno
    return dicts


def generate_attr_label(attr_ref, attr_vocab, num_attr):

    attr_lab = np.zeros(num_attr)
    if attr_ref is not None:
        if len(attr_ref) > 0:
            for att in attr_ref:
                attr_lab[attr_vocab[att]] = 1

    return attr_lab





def register_refcoco(name, img_root, pickle_path, txt_path, precomp_anno_path):

    DatasetCatalog.register(name, lambda: load_refcoco_instances(img_root, pickle_path, txt_path, precomp_anno_path))
    MetadataCatalog.get(name).set(
        img_root=img_root, split=txt_path, evaluator_type="refcoco")