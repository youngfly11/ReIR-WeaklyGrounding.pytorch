#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/10/14 21:14

import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
from detectron2.config import global_cfg as cfg
import torch
import torch.nn.functional as F
from fvcore.common.file_io import PathManager
from detectron2.structures.boxes import pairwise_iou
from detectron2.utils.comm import all_gather, is_main_process, synchronize
import pickle
from .evaluator import DatasetEvaluator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes
from detectron2.utils.events import get_event_storage



class FLICKR30KEvaluator(DatasetEvaluator):

    """
    Evaluate semantic segmentation
    """

    def __init__(self, dataset_name, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            num_classes (int): number of classes
            ignore_label (int): value in semantic segmentation ground truth. Predictions for the
            corresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
        """
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self.box2box_translation = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)


    def reset(self):
        self._predictions = {}
        self._losses = {}


    def record_losses(self, losses, all_img_ref_ids):
        """

        :param losses: which is an dict, contain multiple kinds of losses
        :return:
        """
        ## here we just token one img_id to be the key
        img_spec_id = all_img_ref_ids[0]
        self._losses.update({img_spec_id: losses})

    def get_avg_losses(self, ):


        if self._distributed:
            synchronize()
            self._losses = all_gather(self._losses)

            if not is_main_process():
                return

            all_losses = {}
            for p in self._losses:
                all_losses.update(p)
        else:
            all_losses = self._losses

        image_unique_ids = list(all_losses.keys())

        loss_keys = list(all_losses[image_unique_ids[0]].keys())

        losses_global_avg = {}
        for key in loss_keys:
            losses_global_avg[key] = []

        for img_spec_id in image_unique_ids:
            loss_sig = all_losses[img_spec_id]

            for key in loss_keys:
                losses_global_avg[key].append(loss_sig[key])

        for key in loss_keys:
            losses_global_avg[key] = np.array(losses_global_avg[key]).mean()

        global_loss = OrderedDict(losses_global_avg)
        return global_loss


    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """

        # batch_phrase_ids, batch_phrase_types, batch_pred_boxes, batch_pred_similarity, batch_boxes_targets, batch_precomp_boxes

        # batch_phrase_ids, batch_phrase_types, move2cpu(batch_pred_boxes), move2cpu(batch_pred_similarity),
        # move2cpu(batch_boxes_targets), move2cpu(batch_precomp_boxes), image_unique_id, move2cpu(
        #     batch_topk_pred_similarity),
        # move2cpu(batch_topk_fusion_similarity), move2cpu(batch_topk_pred_boxes), move2cpu(batch_topk_fusion_pred_boxes),
        # move2cpu(batch_topk_precomp_boxes), move2cpu(batch_topk_pred_targets), move2cpu(batch_pred_targets)

        all_phrase_ids = outputs[0]
        all_phrase_type = outputs[1]
        all_pred_boxes = outputs[2]
        all_pred_similarity = outputs[3]
        all_targets = outputs[4]
        all_precomp_boxes = outputs[5]
        all_img_sent_ids = outputs[6]

        all_topk_pred_similarity=outputs[7]
        all_topk_fusion_similarity = outputs[8]
        all_topk_pred_boxes = outputs[9]
        all_topk_fusion_pred_boxes = outputs[10]

        all_topk_precomp_boxes = outputs[11]
        all_topk_pred_targets = outputs[12]
        all_pred_targets = outputs[13]


        for phrase_ids, phrase_type, pred_boxes, pred_sim, targets, precomp_boxes, img_sent_id, topk_pred_sim, topk_fusion_pred_sim, topk_pred_box, topk_fusion_pred_boxes, topk_precomp_boxes,\
            topk_pred_targets, pred_targets in zip(all_phrase_ids, all_phrase_type, all_pred_boxes, all_pred_similarity, all_targets, all_precomp_boxes,
                       all_img_sent_ids, all_topk_pred_similarity, all_topk_fusion_similarity,all_topk_pred_boxes,all_topk_fusion_pred_boxes, all_topk_precomp_boxes,
                       all_topk_pred_targets, all_pred_targets):

            self._predictions.update({img_sent_id: [phrase_ids, phrase_type, pred_boxes, pred_sim, targets, precomp_boxes, topk_pred_box, topk_pred_sim, topk_fusion_pred_boxes, topk_fusion_pred_sim,
                                                    topk_precomp_boxes, topk_pred_targets, pred_targets]})


    def evaluate(self):
        """
        Evaluates Referring Segmentation IoU:
        """

        if self._distributed:
            synchronize()

            self._predictions = all_gather(self._predictions)

            if not is_main_process():
                return

            all_prediction = {}
            for p in self._predictions:
                all_prediction.update(p)
        else:
            all_prediction = self._predictions

        image_unique_ids = list(all_prediction.keys())

        total_num = 0
        recall_num = 0
        recall_t2_num = 0
        recall_t2_fusion_num = 0
        num_type = {}
        recall_type = {}
        acc_type = {}
        recall_topk_num = {5:0, 10:0}

        point_recall_num = 0
        point_recall_t2_num = 0
        point_recall_fusion_t2_num = 0

        self.storage = get_event_storage()
        # self.iter = 100000

        for img_sent_id in image_unique_ids:

            # phrase_ids, phrase_type, pred_boxes, pred_sim, targets, precomp_boxes, topk_pred_box, topk_pred_sim, topk_fusion_pred_boxes, topk_fusion_pred_sim,
            # topk_precomp_boxes, topk_pred_targets, pred_targets

            result = all_prediction[img_sent_id]
            phrase_types = result[1]
            pred_boxes = result[2]
            pred_similarity = result[3]
            targets = result[4]
            precomp_boxes = result[5]
            topk_pred_boxes = result[6]
            topk_fusion_pred_boxes = result[8]
            pred_targets = result[12]

            pred_boxes.clip()
            ious = pairwise_iou(targets, pred_boxes)  # this function will change the target_boxes into cuda mode
            iou = ious.numpy().diagonal()
            total_num += iou.shape[0]
            recall_num += int((iou >= cfg.MODEL.VG.EVAL_THRESH).sum())  # 0.5

            pred_boxes_tensor = pred_boxes.tensor
            pred_center = (pred_boxes_tensor[:, :2] + pred_boxes_tensor[:, 2:]) / 2.0
            pred_center = pred_center.repeat(1, 2)  ## x_c, y_c, x_c, y_c
            targets_tensor = targets.tensor
            fall_tensor = targets_tensor - pred_center
            fall_tensor = (fall_tensor[:, :2] <= 0).float().sum(1) + (fall_tensor[:, 2:] >= 0).float().sum(1)
            point_recall_num += (fall_tensor == 4).float().numpy().sum()

            topk_pred_boxes.clip()
            ious_topk = pairwise_iou(targets, topk_pred_boxes)
            recall_t2_num += int((ious_topk.numpy().diagonal()>=cfg.MODEL.VG.EVAL_THRESH).sum())

            topk_boxes_tensor = topk_pred_boxes.tensor
            pred_center = (topk_boxes_tensor[:, :2] + topk_boxes_tensor[:, 2:]) / 2.0
            pred_center = pred_center.repeat(1, 2)  ## x_c, y_c, x_c, y_c
            fall_tensor = targets_tensor - pred_center
            fall_tensor = (fall_tensor[:, :2] <= 0).float().sum(1) + (fall_tensor[:, 2:] >= 0).float().sum(1)
            point_recall_t2_num += (fall_tensor == 4).float().numpy().sum()


            topk_fusion_pred_boxes.clip()
            ious_fusion_topk = pairwise_iou(targets, topk_fusion_pred_boxes)
            recall_t2_fusion_num += int((ious_fusion_topk.numpy().diagonal() >= cfg.MODEL.VG.EVAL_THRESH).sum())

            topk_fusion_boxes_tensor = topk_fusion_pred_boxes.tensor
            pred_center = (topk_fusion_boxes_tensor[:, :2] + topk_fusion_boxes_tensor[:, 2:]) / 2.0
            pred_center = pred_center.repeat(1, 2)  ## x_c, y_c, x_c, y_c
            fall_tensor = targets_tensor - pred_center
            fall_tensor = (fall_tensor[:, :2] <= 0).float().sum(1) + (fall_tensor[:, 2:] >= 0).float().sum(1)
            point_recall_fusion_t2_num += (fall_tensor == 4).float().numpy().sum()


            for pid, p_type in enumerate(phrase_types):
                p_type = p_type[0]
                num_type[p_type] = num_type.setdefault(p_type, 0) + 1
                recall_type[p_type] = recall_type.setdefault(p_type, 0) + (iou[pid] >= cfg.MODEL.VG.EVAL_THRESH)

            precomp_boxes.clip()

            if self.storage.iter <= cfg.SOLVER.REG_START_ITER:
            # if self.iter <= cfg.SOLVER.REG_START_ITER:
                ious_top = pairwise_iou(targets, precomp_boxes).cpu()
            else:
                ious_top = []
                for pid, reg_targets in enumerate(pred_targets):
                    target_pid = targets[[pid]]
                    box_after_reg = self.box2box_translation.apply_deltas(reg_targets, precomp_boxes.tensor)
                    box_after_reg = Boxes(box_after_reg, precomp_boxes.size)
                    box_after_reg.clip()
                    ious_pid = pairwise_iou(target_pid, box_after_reg)
                    ious_top.append(ious_pid)
                ious_top = torch.cat(ious_top, dim=0)

            for k in [5, 10]:
                top_k = torch.topk(pred_similarity, k=k, dim=1)[0][:, [-1]]
                pred_similarity_topk = (pred_similarity >= top_k).float()
                ious_top_k = (ious_top * pred_similarity_topk).numpy()
                recall_topk_num[k] += int(((ious_top_k >= cfg.MODEL.VG.EVAL_THRESH).sum(1) > 0).sum())

        acc = recall_num / total_num
        acc_top5 = recall_topk_num[5] / total_num
        acc_top10 = recall_topk_num[10] / total_num

        acc_s2 = recall_t2_num/total_num
        acc_s2_fusion = recall_t2_fusion_num/total_num

        point_acc = point_recall_num / total_num
        point_acc_s2 = point_recall_t2_num / total_num
        point_acc_s2_fusion = point_recall_fusion_t2_num / total_num

        for type, type_num in num_type.items():
            acc_type[type] = recall_type[type] / type_num

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "prediction_{}.pkl".format(str(acc_s2).replace('.', '_')[:6]))
            with PathManager.open(file_path, "wb") as f:
                pickle.dump(all_prediction, f)

        del all_prediction
        self._logger.info('evaluation on {} expression instances, detailed_iou: {}'.format(len(image_unique_ids), acc_type))
        self._logger.info('Evaluate Pointing Accuracy: PointAcc:{}, PointAccS2:{}, PointAccS2Fusion:{}'.format(point_acc, point_acc_s2, point_acc_s2_fusion))
        results = OrderedDict({"acc": acc, "acc_top5": acc_top5, "acc_top10": acc_top10, 'acc_s2': acc_s2, 'acc_s2_fusion': acc_s2_fusion})
        return results