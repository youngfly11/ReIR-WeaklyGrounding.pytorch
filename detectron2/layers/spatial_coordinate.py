#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/10/14 22:20





import numpy as np
import torch



def meshgrid_generation(h, w):

    half_h = h/2
    half_w = w/2

    grid_h, grid_w = torch.meshgrid(torch.arange(h), torch.arange(w))
    grid_h = grid_h.float()
    grid_w = grid_w.float()
    grid_h = grid_h/half_h - 1
    grid_w = grid_w/half_w - 1
    spatial_coord = torch.cat((grid_h[None,None, :,:], grid_w[None, None, :, :]), 1)
    spatial_coord = spatial_coord.to(torch.device('cuda'))

    return spatial_coord



def generate_spatial_cues(featmap_H, featmap_W):

    grid_h, grid_w = torch.meshgrid(torch.arange(featmap_H), torch.arange(featmap_W))
    grid_h = grid_h.float()
    grid_w = grid_w.float()
    xmin = grid_w/featmap_W*2 -1
    xmax = (grid_w+1)/featmap_W*2 - 1
    xctr = (xmin+xmax)/2.0
    ymin = grid_h / featmap_H * 2 - 1
    ymax = (grid_h + 1) / featmap_H*2 - 1
    yctr = (ymin + ymax) / 2.0
    size = torch.ones(featmap_H, featmap_W)

    spatial_cues = torch.stack([xmin, ymin, xmax, ymax, xctr, yctr,(1.0/featmap_W)*size, (1.0/featmap_H)*size], dim=0)

    spatial_cues = spatial_cues.to(torch.device('cuda')).unsqueeze(0)

    return spatial_cues


def get_spatial_feat(precomp_boxes):

    bbox = precomp_boxes.tensor
    bbox_size = [precomp_boxes.size[0], precomp_boxes.size[1]]  ## width, height
    bbox_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    bbox_area_ratio = bbox_area / (bbox_size[0] * bbox_size[1])
    bbox_area_ratio = bbox_area_ratio.unsqueeze(1)  # 100 * 1
    device_id = precomp_boxes.tensor.get_device()
    bbox_size.extend(bbox_size)
    bbox_size = torch.FloatTensor(np.array(bbox_size).astype(np.float32)).to(device_id)
    bbox = bbox / bbox_size
    vis_spatial = torch.cat((bbox, bbox_area_ratio), 1)
    return vis_spatial


def generate_mattnet_spatial_feat(precomp_boxes, pred_det_label, topk=5):

    precomp_boxes_tensor = precomp_boxes.tensor
    h, w = precomp_boxes.size
    lfeat = precomp_boxes_tensor / torch.as_tensor(np.array([w, h])).repeat(2).to(torch.device('cuda'))

    num_boxes = precomp_boxes_tensor.shape[0]
    c_boxes_w = (precomp_boxes_tensor[:, 0] + precomp_boxes_tensor[:, 2])/2
    c_boxes_h = (precomp_boxes_tensor[:, 1] + precomp_boxes_tensor[:, 3])/2
    c_boxes = torch.stack((c_boxes_w, c_boxes_h), dim=1) ## w
    dist = ((c_boxes.unsqueeze(1) - c_boxes)**2).mean(2) ## N*N

    pred_det_label = torch.as_tensor(pred_det_label).float()
    pred_same_type = (pred_det_label.unsqueeze(1) - pred_det_label + torch.eye(num_boxes)) == 0 ## exclude the self boxes
    pred_same_type = pred_same_type.float().to(torch.device('cuda'))

    ## construct the
    rw, rh = precomp_boxes_tensor[:,2]-precomp_boxes_tensor[:, 0], precomp_boxes_tensor[:,3]-precomp_boxes_tensor[:, 1]
    # delta_xy = (precomp_boxes_tensor - torch.cat((c_boxes_w, c_boxes_h), dim=1).repeat(1, 2))/torch.cat((rw, rh), dim=1).repeat(1,2)
    delta_x1 = (precomp_boxes_tensor[:, [0]] - c_boxes_w)/rw
    delta_y1 = (precomp_boxes_tensor[:, [1]] - c_boxes_h)/rh
    delta_x2 = (precomp_boxes_tensor[:, [2]] - c_boxes_w)/rw
    delta_y2 = (precomp_boxes_tensor[:, [3]] - c_boxes_h)/rh
    delta_area = (rw*rh).unsqueeze(1)/(rw*rh)

    lfeat_area = (rw*rh)/(h*w)
    lfeat = torch.cat((lfeat, lfeat_area.unsqueeze(1)), dim=1)

    dif_lfeat_candy = torch.stack((delta_x1, delta_y1, delta_x2, delta_y2, delta_area), dim=2) ## N*N*5
    dif_lfeat = torch.zeros(precomp_boxes_tensor.shape[0], 5*topk).to(torch.device('cuda'))

    for nid in range(num_boxes):
        pred_st_nid_index = torch.where(pred_same_type[nid]>0)[0]
        if pred_st_nid_index.shape[0] > 0:
            dist_nid = dist[nid][pred_st_nid_index] ## N
            argsort = torch.argsort(dist_nid)[:topk]
            dif_lfeat_nid_argsort = dif_lfeat_candy[nid, pred_st_nid_index[argsort]].reshape(-1) ## topk*5
            dif_lfeat[nid, :dif_lfeat_nid_argsort.shape[0]] = dif_lfeat[nid, :dif_lfeat_nid_argsort.shape[0]]+dif_lfeat_nid_argsort

    spa_feat = torch.cat((lfeat, dif_lfeat), dim=1)

    return spa_feat















if __name__ == '__main__':

    # feat = torch.ones(3,1,50,50)
    # meshgrid_generation(feat=feat)

    generate_spatial_cues(40,40)

