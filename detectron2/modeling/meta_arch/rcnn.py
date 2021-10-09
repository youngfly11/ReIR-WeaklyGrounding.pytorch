# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.config.config import global_cfg as cfg
from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

from detectron2.modeling.weaklygrounding.kac_net import build_kac_head as build_kac_vg_head
from detectron2.modeling.weaklygrounding.vg_detection_weakly_v1 import build_vg_head as build_weakly_vg_head_v1
from detectron2.modeling.weaklygrounding.vg_detection_weakly_v3 import build_vg_head as build_weakly_vg_head_v2
from detectron2.modeling.weaklygrounding.weakly_visual_grounding_regression import build_vg_head as build_weakly_vg_head_v3
from detectron2.modeling.weaklygrounding.weakly_visual_grounding_reg_rel import build_vg_head as build_weakly_vg_head_v6



__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfgs):
        super().__init__()

        self.device = torch.device('cuda')
        self.backbone = build_backbone(cfgs, None)
        self.backbone_base = self.backbone.RCNN_base
        self.backbone_top = self.backbone.RCNN_top
        roi_pooler = build_roi_heads(cfgs, None)

        self.pixel_mean = cfg.MODEL.PIXEL_MEAN

        if cfg.DATASETS.NAME == 'flickr30k':
            if cfg.MODEL.VG.NETWORK == 'Kac':
                self.weakly_vg_head = build_kac_vg_head(roi_pooler, self.backbone_top)
            elif cfg.MODEL.VG.NETWORK == 'Baseline':
                self.weakly_vg_head = build_weakly_vg_head_v1(roi_pooler, self.backbone_top)
            elif cfg.MODEL.VG.NETWORK == 'Baseline_s2':
                self.weakly_vg_head = build_weakly_vg_head_v2(roi_pooler, self.backbone_top)
            elif cfg.MODEL.VG.NETWORK == 'Reg':
                self.weakly_vg_head = build_weakly_vg_head_v3(roi_pooler, self.backbone_top)
            elif cfg.MODEL.VG.NETWORK == 'RegRel':
                self.weakly_vg_head = build_weakly_vg_head_v6(roi_pooler, self.backbone_top)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError


        # assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.as_tensor(self.pixel_mean).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean)
        self.to(self.device)

        if cfg.SOLVER.FIX_BACKBONE:
            self.backbone.eval()
            for each in self.backbone.parameters():
                each.requires_grad = False

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        if not self.training:
            return self.inference(batched_inputs)

        if cfg.DATASETS.NAME == 'flickr30k':
            images, phrase_ids, gt_boxes, precomp_bbox, precomp_score, precomp_det_label, image_scale, relations, sentence, image_unique_id, det_label_embedding = batched_inputs
            images = self.preprocess_image(images)
            features = self.backbone_base(images.tensor)
            all_loss, results = self.weakly_vg_head(features, phrase_ids, gt_boxes, precomp_bbox, precomp_score, precomp_det_label, image_scale, relations, sentence, image_unique_id, det_label_embedding)
        else:
            raise NotImplementedError

        losses = {}
        losses.update(all_loss)
        return losses, results

    def inference(self, batched_inputs,  detected_instances=None, do_postprocess=False):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        losses = {}

        if cfg.DATASETS.NAME == 'flickr30k':
            images, phrase_ids, gt_boxes, precomp_bbox, precomp_score, precomp_det_label, image_scale, relations, sentence, image_unique_id, det_label_embedding = batched_inputs
            images = self.preprocess_image(images)
            features = self.backbone_base(images.tensor)
            all_losses, results = self.weakly_vg_head(features, phrase_ids, gt_boxes, precomp_bbox, precomp_score, precomp_det_label, image_scale, relations, sentence, image_unique_id, det_label_embedding)
        else:
            raise NotImplementedError


        losses.update(all_losses)
        return losses, results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x.to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, 2)
        return images


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        pixel_mean = torch.as_tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1).float()
        pixel_std = torch.as_tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1).float()
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
