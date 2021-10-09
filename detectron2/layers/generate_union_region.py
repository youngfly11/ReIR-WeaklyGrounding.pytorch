#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/5/17 10:52




import numpy as np
import torch
import os.path as osp
from detectron2.structures.boxes import phrase_boxlist_union


def generate_union_region_boxes(relation_conn, precomp_bbox, topN_boxes_ids, topN_boxes_scores):
    """
    To generate the union bbox
    :param relation_conn: list [[1,2],[1,2]]
    :param precompute_bbox: boxlist. boxes
    :param topN_boxes_ids, MxN, M is the number of phrases, N is the number of topN boxes
    :return:
    conn_map: nparray.   (num_phrases * topN, num_phrases * topN). -1 denote no connection.
    0~M, denote index of the union region sorted in phrsbj2obj_union.
    phrsbj2obj_union: the union region lab.
    phrsbj2obj_spa_config: denote pair-wise spatial terminal
    """
    ## construct the global connection map
    num_phrases, topN = topN_boxes_ids.shape
    conn_map = np.zeros((num_phrases * topN, num_phrases * topN)) - 1

    ## todo
    ## we can further consider inner relation and sym relation
    for rel in relation_conn:
        conn_map[rel[0]*topN:(rel[0]+1)*topN, rel[1]*topN:(rel[1]+1)*topN] = generate_pairwise_relationship(topN_boxes_ids, precomp_bbox, rel)

    conn_phrtnsbj, conn_phrtnobj = np.where(conn_map == 1)

    conn_phrtnsbj_1 = conn_phrtnsbj // topN
    conn_phrtnobj_1 = conn_phrtnobj // topN

    conn_phrtnobj_select = np.tile(np.arange(topN), int(conn_phrtnobj.shape[0]/topN))
    conn_phrtnsbj_bbox_id = topN_boxes_ids[conn_phrtnsbj_1, conn_phrtnsbj%topN]
    conn_phrtnobj_bbox_id = topN_boxes_ids[conn_phrtnobj_1, conn_phrtnobj_select]

    ## nms scores
    phrtnsbj_scores = topN_boxes_scores[conn_phrtnsbj_1, conn_phrtnsbj%topN]
    phrtnobj_scores = topN_boxes_scores[conn_phrtnobj_1, conn_phrtnobj_select]
    phrsbj2obj_scores = phrtnsbj_scores * phrtnobj_scores

    precomp_bbox_phrtnsbj = precomp_bbox[conn_phrtnsbj_bbox_id]
    precomp_bbox_phrtnobj = precomp_bbox[conn_phrtnobj_bbox_id]


    phrsbj2obj_scores_sort, phrsbj2obj_scores_sort_id = torch.sort(phrsbj2obj_scores, descending=True)

    ## keep indicate the merged boxes index.
    ## cluster_idx denote the all phrase
    phrsbj2obj_union_all, phrsbj2obj_union, cluster_idx, keep = phrase_boxlist_union(precomp_bbox_phrtnsbj,
                        precomp_bbox_phrtnobj, phrsbj2obj_scores_sort, phrsbj2obj_scores_sort_id, True)

    ## reverse the sort. cluster_idx here record each union belone to index of phrsbj2obj_union.
    ## we need to resort the cluster_idx to match it into conn_map
    phrsbj2obj_scores_sort_id_sort = torch.argsort(phrsbj2obj_scores_sort_id)
    keep_inds_mapping = cluster_idx[phrsbj2obj_scores_sort_id_sort]
    conn_map[conn_phrtnsbj, conn_phrtnobj] = keep_inds_mapping.detach().cpu().numpy()
    phrsbj2obj_union_all_bbox = phrsbj2obj_union_all[phrsbj2obj_scores_sort_id_sort]

    phrsbj2obj_spa_config = generate_pairwise_spatial_configuration(union_all=phrsbj2obj_union_all_bbox,
                                                                    subject_all=precomp_bbox_phrtnsbj,
                                                                    object_all=precomp_bbox_phrtnobj)

    phrsbj2obj_spa_config = torch.as_tensor(phrsbj2obj_spa_config).float().to(torch.device('cuda'))

    return conn_map, phrsbj2obj_union, phrsbj2obj_spa_config



def generate_pairwise_relationship(topN_box_ids, precomp_boxes, rel):

    """
    We follow the paper "Cross-Modal Relationship Inference for Grounding Referring Expressions" to construct the visual graph
    and ignore the relationship large than (h**2+w**2)**0.5
    """

    subject_id, object_id = rel[:2]
    topN_box_ids = topN_box_ids.long()  ## M*10

    sub_boxes = precomp_boxes[topN_box_ids[subject_id]].tensor
    obj_boxes = precomp_boxes[topN_box_ids[object_id]].tensor
    h,w = precomp_boxes.size
    scale = (h**2+w**2)**0.5
    sub_boxes[:, 0] = (sub_boxes[:, 0] + sub_boxes[:, 2]) / 2.0
    sub_boxes[:, 1] = (sub_boxes[:, 1] + sub_boxes[:, 3]) / 2.0
    sub_boxes = sub_boxes[:, :2]
    obj_boxes[:, 0] = (obj_boxes[:, 0] + obj_boxes[:, 2]) / 2.0
    obj_boxes[:, 1] = (obj_boxes[:, 1] + obj_boxes[:, 3]) / 2.0
    obj_boxes = obj_boxes[:, :2]
    h_dist = (sub_boxes[:, [0]] - obj_boxes[:, [0]].permute(1, 0))**2
    w_dist = (sub_boxes[:, [1]] - obj_boxes[:, [1]].permute(1, 0))**2
    dist = (h_dist + w_dist)**0.5
    dist = (dist<scale).float()*2-1
    dist = dist.to(torch.device('cpu')).numpy()
    return dist




def generate_pairwise_spatial_configuration(union_all, subject_all, object_all, heatmap_size=64):

    union_all = union_all.tensor.detach().cpu().numpy()
    subject_all = subject_all.tensor.detach().cpu().numpy()
    object_all = object_all.tensor.detach().cpu().numpy()

    assert union_all.shape[0] == subject_all.shape[0] == object_all.shape[0]
    number_of_union = union_all.shape[0]

    union_mask = np.zeros((number_of_union, 2, heatmap_size, heatmap_size))

    ux0 = union_all[:, 0]
    uy0 = union_all[:, 1]
    ux1 = union_all[:, 2]
    uy1 = union_all[:, 3]

    scale_x = heatmap_size / (ux1 - ux0)
    scale_y = heatmap_size / (uy1 - uy0)

    for uid in range(number_of_union):

        ox0, oy0, ox1, oy1 = subject_all[uid]
        ox0 = np.clip(np.round((ox0 - ux0[uid]) * scale_x[uid]).astype(np.int), 0, heatmap_size - 1)
        oy0 = np.clip(np.round((oy0 - uy0[uid]) * scale_y[uid]).astype(np.int), 0, heatmap_size - 1)
        ox1 = np.clip(np.round((ox1 - ux0[uid]) * scale_x[uid]).astype(np.int), 0, heatmap_size - 1)
        oy1 = np.clip(np.round((oy1 - uy0[uid]) * scale_y[uid]).astype(np.int), 0, heatmap_size - 1)

        union_mask[uid, 0, oy0:oy1, ox0:ox1] = 1.

        ox0, oy0, ox1, oy1 = object_all[uid]
        ox0 = np.clip(np.round((ox0 - ux0[uid]) * scale_x[uid]).astype(np.int), 0, heatmap_size - 1)
        oy0 = np.clip(np.round((oy0 - uy0[uid]) * scale_y[uid]).astype(np.int), 0, heatmap_size - 1)
        ox1 = np.clip(np.round((ox1 - ux0[uid]) * scale_x[uid]).astype(np.int), 0, heatmap_size - 1)
        oy1 = np.clip(np.round((oy1 - uy0[uid]) * scale_y[uid]).astype(np.int), 0, heatmap_size - 1)

        union_mask[uid, 1, oy0:oy1, ox0:ox1] = 1.

    return union_mask.astype(np.float32)