#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/10/14 21:14



import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
import pickle
from .evaluator import DatasetEvaluator


class RECOCOEvaluator(DatasetEvaluator):

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

        all_pred_prob = outputs[0]
        all_meanIoU = outputs[1]
        all_meanIoU_bg = outputs[2]
        all_inter = outputs[3]
        all_union = outputs[4]
        all_img_ref_ids = outputs[5]

        for pred, meanIoU, meanIoU_bg, inter, union, img_ref_exp_id in zip(all_pred_prob, all_meanIoU, all_meanIoU_bg, all_inter, all_union, all_img_ref_ids):

            # pred = pred.to(self._cpu_device).numpy().astype(np.float16) ## here we transform the data into numpy
            iou = float(meanIoU.to(self._cpu_device).numpy())
            iou_bg = float(meanIoU_bg.to(self._cpu_device).numpy())
            inter = float(inter.to(self._cpu_device).numpy())
            union = float(union.to(self._cpu_device).numpy())

            # gt_mask = gt_mask.to(self._cpu_device)
            # init_mask = init_mask.to(self._cpu_device)
            self._predictions.update({img_ref_exp_id: [iou, iou_bg, inter, union]})


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
        all_mIoU = []
        all_inter = []
        all_union = []
        all_mIoU_bg = []

        for img_sent_id in image_unique_ids:
            result = all_prediction[img_sent_id]
            all_mIoU.append(result[0])
            all_mIoU_bg.append(result[1])
            all_inter.append(result[2])
            all_union.append(result[3])

        MIoU = np.array(all_mIoU).mean()
        MIoU_bg = np.array(all_mIoU_bg).mean()
        OverIoU = np.array(all_inter).sum()/np.array(all_union).sum()

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "prediction.pkl")
            with PathManager.open(file_path, "wb") as f:
                pickle.dump(all_prediction, f)
        self._logger.info('evaluation on {} expression instances'.format(len(image_unique_ids)))
        results = OrderedDict({"MeanIoU": MIoU, "OverIoU": OverIoU, "MeanIoU_bg": MIoU_bg})
        return results