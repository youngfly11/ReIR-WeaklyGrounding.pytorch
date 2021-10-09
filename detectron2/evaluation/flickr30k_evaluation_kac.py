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

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
import pickle
from .evaluator import DatasetEvaluator


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

        # batch_phrase_ids, batch_phrase_types, batch_pred_boxes, batch_pred_similarity, batch_boxes_targets, batch_precomp_boxes, image_unique_id
        # batch_phrase_ids, batch_phrase_types, batch_pred_boxes, batch_pred_similarity, batch_boxes_targets, batch_precomp_boxes, image_unique_id


        all_phrase_ids = outputs[0]
        all_phrase_type = outputs[1]
        all_pred_boxes = outputs[2]
        all_pred_similarity = outputs[3]
        all_targets = outputs[4]
        all_precomp_boxes = outputs[5]
        all_img_sent_ids = outputs[6]


        for phrase_ids, phrase_type, pred_boxes, pred_sim, targets, precomp_boxes, img_sent_id in zip(all_phrase_ids, all_phrase_type, all_pred_boxes, all_pred_similarity, all_targets, all_precomp_boxes,all_img_sent_ids):
            self._predictions.update({img_sent_id: [phrase_ids, phrase_type, pred_boxes, pred_sim, targets, precomp_boxes]})


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
        num_type = {}
        recall_type = {}
        acc_type = {}
        recall_topk_num = {5:0, 10:0}
        point_recall_num = 0

        for img_sent_id in image_unique_ids:

            result = all_prediction[img_sent_id]
            phrase_ids = result[0]
            phrase_types = result[1]
            pred_boxes = result[2]
            pred_similarity = result[3]
            targets = result[4]
            precomp_boxes = result[5]


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



            for pid, p_type in enumerate(phrase_types):
                p_type = p_type[0]
                num_type[p_type] = num_type.setdefault(p_type, 0) + 1
                recall_type[p_type] = recall_type.setdefault(p_type, 0) + (iou[pid] >= cfg.MODEL.VG.EVAL_THRESH)

            precomp_boxes.clip()
            ious_top = pairwise_iou(targets, precomp_boxes).cpu()

            for k in [5, 10]:
                top_k = torch.topk(pred_similarity, k=k, dim=1)[0][:, [-1]]
                pred_similarity_topk = (pred_similarity >= top_k).float()
                ious_top_k = (ious_top * pred_similarity_topk).numpy()
                recall_topk_num[k] += int(((ious_top_k >= cfg.MODEL.VG.EVAL_THRESH).sum(1) > 0).sum())

        acc = recall_num / total_num
        acc_top5 = recall_topk_num[5] / total_num
        acc_top10 = recall_topk_num[10] / total_num
        point_acc = point_recall_num / total_num

        for type, type_num in num_type.items():
            acc_type[type] = recall_type[type] / type_num

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "prediction_{}.pkl".format(str(acc).replace('.', '_')[:6]))
            with PathManager.open(file_path, "wb") as f:
                pickle.dump(all_prediction, f)

        del all_prediction
        self._logger.info('evaluation on {} expression instances, detailed_iou: {}'.format(len(image_unique_ids), acc_type))
        self._logger.info('Evaluate Pointing Accuracy: PointAcc:{}'.format(point_acc))
        results = OrderedDict({"acc": acc, "acc_top5": acc_top5, "acc_top10": acc_top10})
        return results