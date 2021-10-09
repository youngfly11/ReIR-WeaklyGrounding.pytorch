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

        # batch_phrase_ids, batch_phrase_types, batch_pred_boxes, batch_pred_similarity, batch_boxes_targets, batch_precomp_boxes

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

        all_topk2_pred_similarity = outputs[11]
        all_topk2_fusion_similarity=outputs[12]
        all_topk2_pred_boxes = outputs[13]
        all_topk2_fusion_pred_boxes = outputs[14]


        for phrase_ids, phrase_type, pred_boxes, pred_sim, targets, precomp_boxes, img_sent_id, topk_pred_sim, topk_fusion_pred_sim, topk_pred_box, topk_fusion_pred_boxes, topk2_pred_sim, topk2_fusion_pred_sim, topk2_pred_box, topk2_fusion_pred_boxes \
                in zip(all_phrase_ids, all_phrase_type, all_pred_boxes, all_pred_similarity, all_targets, all_precomp_boxes,
                       all_img_sent_ids, all_topk_pred_similarity, all_topk_fusion_similarity,all_topk_pred_boxes,all_topk_fusion_pred_boxes, all_topk2_pred_similarity, all_topk2_fusion_similarity,
                       all_topk2_pred_boxes, all_topk2_fusion_pred_boxes):

            pred_boxes = pred_boxes.to(self._cpu_device)
            pred_sim = pred_sim.to(self._cpu_device)
            targets = targets.to(self._cpu_device)
            precomp_boxes = precomp_boxes.to(self._cpu_device)
            topk_pred_box = topk_pred_box.to(self._cpu_device)
            topk_fusion_pred_boxes = topk_fusion_pred_boxes.to(self._cpu_device)
            topk_pred_sim = topk_pred_sim.to(self._cpu_device)
            topk_fusion_pred_sim = topk_fusion_pred_sim.to(self._cpu_device)

            topk2_pred_box = topk2_pred_box.to(self._cpu_device)
            topk2_fusion_pred_boxes = topk2_fusion_pred_boxes.to(self._cpu_device)
            topk2_pred_sim = topk2_pred_sim.to(self._cpu_device)
            topk2_fusion_pred_sim = topk2_fusion_pred_sim.to(self._cpu_device)

            self._predictions.update({img_sent_id: [phrase_ids, phrase_type, pred_boxes, pred_sim, targets, precomp_boxes, topk_pred_box, topk_pred_sim, topk_fusion_pred_boxes, topk_fusion_pred_sim,
                                                    topk2_pred_box, topk2_pred_sim, topk2_fusion_pred_boxes, topk2_fusion_pred_sim]})


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
        recall_t3_num = 0
        recall_t3_fusion_num = 0

        num_type = {}
        recall_type = {}
        acc_type = {}
        recall_topk_num = {5:0, 10:0}

        for img_sent_id in image_unique_ids:

            result = all_prediction[img_sent_id]
            phrase_ids = result[0]
            phrase_types = result[1]
            pred_boxes = result[2]
            pred_similarity = result[3]
            targets = result[4]
            precomp_boxes = result[5]

            topk_pred_boxes = result[6]
            topk_fusion_pred_boxes = result[8]

            topk2_pred_boxes = result[10]
            topk2_fusion_pred_boxes = result[12]

            pred_boxes.clip()
            ious = pairwise_iou(targets, pred_boxes)  # this function will change the target_boxes into cuda mode
            iou = ious.numpy().diagonal()
            total_num += iou.shape[0]
            recall_num += int((iou >= cfg.MODEL.VG.EVAL_THRESH).sum())  # 0.5

            topk_pred_boxes.clip()
            ious_topk = pairwise_iou(targets, topk_pred_boxes)
            recall_t2_num += int((ious_topk.numpy().diagonal()>cfg.MODEL.VG.EVAL_THRESH).sum())

            topk_fusion_pred_boxes.clip()
            ious_fusion_topk = pairwise_iou(targets, topk_fusion_pred_boxes)
            recall_t2_fusion_num += int((ious_fusion_topk.numpy().diagonal() > cfg.MODEL.VG.EVAL_THRESH).sum())

            topk2_pred_boxes.clip()
            ious_topk2 = pairwise_iou(targets, topk2_pred_boxes)
            recall_t3_num += int((ious_topk2.numpy().diagonal() > cfg.MODEL.VG.EVAL_THRESH).sum())

            topk2_fusion_pred_boxes.clip()
            ious_fusion_topk2 = pairwise_iou(targets, topk2_fusion_pred_boxes)
            recall_t3_fusion_num += int((ious_fusion_topk2.numpy().diagonal() > cfg.MODEL.VG.EVAL_THRESH).sum())


            for pid, p_type in enumerate(phrase_types):
                p_type = p_type[0]
                num_type[p_type] = num_type.setdefault(p_type, 0) + 1
                recall_type[p_type] = recall_type.setdefault(p_type, 0) + (iou[pid] >= cfg.MODEL.VG.EVAL_THRESH)

            precomp_boxes.clip()
            ious_top = pairwise_iou(targets, precomp_boxes).cpu()
            # pred_similarity = F.softmax(pred_sim, dim=1)

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

        acc_s3 = recall_t3_num / total_num
        acc_s3_fusion = recall_t3_fusion_num / total_num

        for type, type_num in num_type.items():
            acc_type[type] = recall_type[type] / type_num

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "prediction_{}.pkl".format(str(acc).replace('.', '_')[:6]))
            with PathManager.open(file_path, "wb") as f:
                pickle.dump(all_prediction, f)

        self._logger.info('evaluation on {} expression instances, detailed_iou: {}'.format(len(image_unique_ids), acc_type))
        results = OrderedDict({"acc": acc, "acc_top5": acc_top5, "acc_top10": acc_top10, 'acc_s2': acc_s2, 'acc_s2_fusion': acc_s2_fusion, 'acc_s3': acc_s3, 'acc_s3_fusion': acc_s3_fusion})
        return results