import torch
from torch.nn import functional as F
import numpy as np
from detectron2.structures.boxes import pairwise_iou
from detectron2.layers.numerical_stability_softmax import numerical_stability_softmax
from detectron2.config import global_cfg as cfg
from detectron2.structures.boxes import boxlist_nms, boxlist_nms_tensor
from detectron2.layers import batched_nms, nms
from detectron2.modeling.box_regression import Box2BoxTransform
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from detectron2.structures import Boxes
from detectron2.utils.events import get_event_storage
import json
from detectron2.layers.weighted_smooth_l1_loss import smooth_l1_loss as weighted_smooth_l1_loss


class WeaklyVGLossCompute():
    def __init__(self):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.device = torch.device('cuda')
        self.margin = 0.2

    def __call__(self, batch_phrase_mask, batch_decode_logits, batch_phrase_dec_ids, batch_ctx_sim):

        noun_reconst_loss = torch.zeros(1).to(self.device)
        disc_img_sent_loss = torch.zeros(1).to(self.device)

        for (phrase_mask, decode_logits, phrase_dec_ids) in zip(batch_phrase_mask, batch_decode_logits, batch_phrase_dec_ids):
            phrase_dec_ids = torch.as_tensor(phrase_dec_ids).long().to(self.device)
            vx, vy = (phrase_mask > 0).nonzero().transpose(0, 1)
            noun_reconst_loss += self.cross_entropy(decode_logits[vx, vy], phrase_dec_ids[vx, vy])

        if cfg.SOLVER.DISC_IMG_SENT_LOSS:
            disc_img_sent_loss += generate_img_sent_discriminative_loss(batch_ctx_sim, self.margin)

        return noun_reconst_loss, disc_img_sent_loss


class WeaklyVGLossComputeV1():
    def __init__(self):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.device = torch.device('cuda')
        self.margin = 0.2

    def __call__(self, batch_phrase_mask, batch_decode_logits, batch_topk_decoder_logits, batch_phrase_dec_ids, batch_ctx_sim, batch_ctx_sim_s1,
                 batch_pred_similarity, batch_topk_pred_similarity, batch_boxes_targets, batch_precomp_boxes):

        noun_reconst_loss = torch.zeros(1).to(self.device)
        noun_topk_reconst_loss = torch.zeros(1).to(self.device)
        disc_img_sent_loss = torch.zeros(1).to(self.device)
        disc_img_sent_loss_s1 = torch.zeros(1).to(self.device)


        for (phrase_mask, decode_logits, decode_topk_logits, phrase_dec_ids, pred_sim, pred_topk_sim,
             targets, precomp_boxes) in zip(batch_phrase_mask, batch_decode_logits, batch_topk_decoder_logits, batch_phrase_dec_ids,
                                            batch_pred_similarity, batch_topk_pred_similarity, batch_boxes_targets, batch_precomp_boxes):

            phrase_dec_ids = torch.as_tensor(phrase_dec_ids).long().to(self.device)
            vx, vy = (phrase_mask > 0).nonzero().transpose(0, 1)
            # noun_reconst_loss += self.cross_entropy(decode_logits[vx, vy], phrase_dec_ids[vx, vy])
            noun_topk_reconst_loss += self.cross_entropy(decode_topk_logits[vx, vy], phrase_dec_ids[vx, vy])

        # disc_img_sent_loss += generate_img_sent_discriminative_loss(batch_ctx_sim, self.margin)
        disc_img_sent_loss_s1 += generate_img_sent_discriminative_loss(batch_ctx_sim_s1, self.margin)

        return noun_reconst_loss, noun_topk_reconst_loss, disc_img_sent_loss, disc_img_sent_loss_s1


class WeaklyVGLossComputeV3():
    def __init__(self):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.device = torch.device('cuda')
        self.margin = 0.2
        self.s2_topk = cfg.MODEL.VG.S2_TOPK
        self.box2box_translation = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        self.reg_iou = cfg.MODEL.VG.REG_IOU
        self.reg_loss_factor = cfg.MODEL.VG.REG_LOSS_FACTOR

    def __call__(self, batch_phrase_mask, batch_decode_logits, batch_topk_decoder_logits, batch_phrase_dec_ids, batch_ctx_sim, batch_ctx_sim_s1,
                 batch_pred_similarity, batch_topk_pred_similarity, batch_boxes_targets, batch_precomp_boxes, batch_pred_targets, batch_topk_pred_targets, batch_topk_precomp_boxes):

        noun_reconst_loss = torch.zeros(1).to(self.device)
        noun_topk_reconst_loss = torch.zeros(1).to(self.device)
        disc_img_sent_loss = torch.zeros(1).to(self.device)
        disc_img_sent_loss_s1 = torch.zeros(1).to(self.device)

        reg_loss = torch.zeros(1).to(self.device)
        reg_loss_s1 = torch.zeros(1).to(self.device)


        for (phrase_mask, decode_logits, decode_topk_logits, phrase_dec_ids, pred_sim, pred_topk_sim,
             targets, precomp_boxes, pred_delta_s1, pred_delta_s2, topk_precomp_boxes) in zip(batch_phrase_mask, batch_decode_logits, batch_topk_decoder_logits, batch_phrase_dec_ids,
                                            batch_pred_similarity, batch_topk_pred_similarity, batch_boxes_targets, batch_precomp_boxes, batch_pred_targets, batch_topk_pred_targets, batch_topk_precomp_boxes):

            phrase_dec_ids = torch.as_tensor(phrase_dec_ids).long().to(self.device)
            vx, vy = (phrase_mask > 0).nonzero().transpose(0, 1)
            noun_reconst_loss += self.cross_entropy(decode_logits[vx, vy], phrase_dec_ids[vx, vy])
            noun_topk_reconst_loss += self.cross_entropy(decode_topk_logits[vx, vy], phrase_dec_ids[vx, vy])

            # ## regression loss
            reg_loss += self.reg_loss_factor * non_maximum_regression_loss_stage1(pred_delta_s1, precomp_boxes, pred_sim, cfg.MODEL.VG.REG_GAP_SCORE, self.reg_iou, self.box2box_translation)
            reg_loss_s1 += self.reg_loss_factor * non_maximum_regression_loss_stage2(pred_delta_s2, topk_precomp_boxes, pred_topk_sim, cfg.MODEL.VG.REG_GAP_SCORE, self.reg_iou, self.box2box_translation)

        disc_img_sent_loss += generate_img_sent_discriminative_loss(batch_ctx_sim, self.margin)
        disc_img_sent_loss_s1 += generate_img_sent_discriminative_loss(batch_ctx_sim_s1, self.margin)

        return noun_reconst_loss, noun_topk_reconst_loss, disc_img_sent_loss, disc_img_sent_loss_s1, reg_loss, reg_loss_s1


class WeaklyVGLossComputeV3Ada():
    def __init__(self):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.device = torch.device('cuda')
        self.margin = 0.2
        self.s2_topk = cfg.MODEL.VG.S2_TOPK
        self.box2box_translation = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        self.reg_iou = cfg.MODEL.VG.REG_IOU
        self.reg_loss_factor = cfg.MODEL.VG.REG_LOSS_FACTOR

    def __call__(self, batch_phrase_mask, batch_decode_logits, batch_topk_decoder_logits, batch_phrase_dec_ids, batch_ctx_sim, batch_ctx_sim_s1,
                 batch_pred_similarity, batch_topk_pred_similarity, batch_boxes_targets, batch_precomp_boxes, batch_pred_targets, batch_topk_pred_targets, batch_topk_precomp_boxes, atten_mask):

        noun_reconst_loss = torch.zeros(1).to(self.device)
        noun_topk_reconst_loss = torch.zeros(1).to(self.device)
        disc_img_sent_loss = torch.zeros(1).to(self.device)
        disc_img_sent_loss_s1 = torch.zeros(1).to(self.device)

        reg_loss = torch.zeros(1).to(self.device)
        reg_loss_s1 = torch.zeros(1).to(self.device)

        for (phrase_mask, decode_logits, decode_topk_logits, phrase_dec_ids, pred_sim, pred_topk_sim,
             targets, precomp_boxes, pred_delta_s1, pred_delta_s2, topk_precomp_boxes, atten_mask_bid) in zip(batch_phrase_mask, batch_decode_logits, batch_topk_decoder_logits, batch_phrase_dec_ids,
                                            batch_pred_similarity, batch_topk_pred_similarity, batch_boxes_targets, batch_precomp_boxes, batch_pred_targets, batch_topk_pred_targets, batch_topk_precomp_boxes, atten_mask):

            phrase_dec_ids = torch.as_tensor(phrase_dec_ids).long().to(self.device)
            vx, vy = (phrase_mask > 0).nonzero().transpose(0, 1)
            noun_reconst_loss += self.cross_entropy(decode_logits[vx, vy], phrase_dec_ids[vx, vy])
            noun_topk_reconst_loss += self.cross_entropy(decode_topk_logits[vx, vy], phrase_dec_ids[vx, vy])

            # ## regression loss
            reg_loss += self.reg_loss_factor * non_maximum_regression_loss_stage1(pred_delta_s1, precomp_boxes, pred_sim, cfg.MODEL.VG.REG_GAP_SCORE, self.reg_iou, self.box2box_translation)
            reg_loss_s1 += self.reg_loss_factor * non_maximum_regression_loss_stage2_ada(pred_delta_s2, topk_precomp_boxes, pred_topk_sim, cfg.MODEL.VG.REG_GAP_SCORE, self.reg_iou, self.box2box_translation, atten_mask_bid)

        disc_img_sent_loss += generate_img_sent_discriminative_loss(batch_ctx_sim, self.margin)
        disc_img_sent_loss_s1 += generate_img_sent_discriminative_loss(batch_ctx_sim_s1, self.margin)

        return noun_reconst_loss, noun_topk_reconst_loss, disc_img_sent_loss, disc_img_sent_loss_s1, reg_loss, reg_loss_s1



class WeaklyVGLossComputeV3Cates():
    def __init__(self):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.device = torch.device('cuda')
        self.margin = 0.2
        self.s2_topk = cfg.MODEL.VG.S2_TOPK
        self.box2box_translation = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        self.reg_iou = cfg.MODEL.VG.REG_IOU
        self.reg_loss_factor = cfg.MODEL.VG.REG_LOSS_FACTOR
        self.rel_cls_loss_factor = cfg.MODEL.VG.REL_CLS_LOSS_FACTOR
        self.binary_cross_entropy = torch.nn.BCEWithLogitsLoss(weight=get_rel_weights(10), reduction='mean')

    def __call__(self, batch_phrase_mask, batch_decode_logits, batch_topk_decoder_logits, batch_phrase_dec_ids, batch_ctx_sim, batch_ctx_sim_s1,
                 batch_pred_similarity, batch_topk_pred_similarity, batch_boxes_targets, batch_precomp_boxes, batch_pred_targets, batch_topk_pred_targets, batch_topk_precomp_boxes,
                 batch_rel_cls_s0, batch_rel_cls_s1, batch_rel_cls_gt, batch_rel_reconst_s0, batch_rel_reconst_s1, batch_rel_mask, batch_rel_dec_idx):

        noun_reconst_loss = torch.zeros(1).to(self.device)
        noun_topk_reconst_loss = torch.zeros(1).to(self.device)
        disc_img_sent_loss = torch.zeros(1).to(self.device)
        disc_img_sent_loss_s1 = torch.zeros(1).to(self.device)

        reg_loss = torch.zeros(1).to(self.device)
        reg_loss_s1 = torch.zeros(1).to(self.device)

        rel_cls_loss = torch.zeros(1).to(self.device)
        rel_cls_loss_s1 = torch.zeros(1).to(self.device)
        rel_const_loss = torch.zeros(1).to(self.device)
        rel_const_loss_s1 = torch.zeros(1).to(self.device)


        for (phrase_mask, decode_logits, decode_topk_logits, phrase_dec_ids, pred_sim, pred_topk_sim,
             targets, precomp_boxes, pred_delta_s1, pred_delta_s2, topk_precomp_boxes, rel_cls_s0, rel_cls_s1, rel_cls_gt, rel_const_s0,
             rel_const_s1, rel_mask, rel_dec_idx) in zip(batch_phrase_mask, batch_decode_logits, batch_topk_decoder_logits, batch_phrase_dec_ids,
                                            batch_pred_similarity, batch_topk_pred_similarity, batch_boxes_targets, batch_precomp_boxes, batch_pred_targets,
                                            batch_topk_pred_targets, batch_topk_precomp_boxes, batch_rel_cls_s0, batch_rel_cls_s1, batch_rel_cls_gt,
                                            batch_rel_reconst_s0, batch_rel_reconst_s1, batch_rel_mask, batch_rel_dec_idx):

            phrase_dec_ids = torch.as_tensor(phrase_dec_ids).long().to(self.device)
            vx, vy = (phrase_mask > 0).nonzero().transpose(0, 1)
            # noun_reconst_loss += self.cross_entropy(decode_logits[vx, vy], phrase_dec_ids[vx, vy])
            noun_topk_reconst_loss += self.cross_entropy(decode_topk_logits[vx, vy], phrase_dec_ids[vx, vy])

            if cfg.MODEL.RELATION.IS_ON:
                if rel_cls_gt is not None:
                    rel_cls_index = rel_cls_gt.sum(1).nonzero().squeeze(1)
                    if rel_cls_index.shape[0] > 0:
                        rel_cls_gt = rel_cls_gt[rel_cls_index]
                        rel_cls_s1 = rel_cls_s1[rel_cls_index]
                        rel_cls_loss_s1 += self.rel_cls_loss_factor*self.binary_cross_entropy(rel_cls_s1, rel_cls_gt)


            # ## regression loss
            reg_loss += self.reg_loss_factor * non_maximum_regression_loss_stage1(pred_delta_s1, precomp_boxes, pred_sim, cfg.MODEL.VG.REG_GAP_SCORE, self.reg_iou, self.box2box_translation)
            reg_loss_s1 += self.reg_loss_factor * non_maximum_regression_loss_stage2(pred_delta_s2, topk_precomp_boxes, pred_topk_sim, cfg.MODEL.VG.REG_GAP_SCORE, self.reg_iou, self.box2box_translation)

        disc_img_sent_loss += generate_img_sent_discriminative_loss(batch_ctx_sim, self.margin)
        disc_img_sent_loss_s1 += generate_img_sent_discriminative_loss(batch_ctx_sim_s1, self.margin)

        return noun_reconst_loss, noun_topk_reconst_loss, disc_img_sent_loss, disc_img_sent_loss_s1, reg_loss, reg_loss_s1, rel_cls_loss, rel_cls_loss_s1, rel_const_loss, rel_const_loss_s1


class WeaklyVGLossComputeV3SemCates():
    def __init__(self):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.device = torch.device('cuda')
        self.margin = 0.2
        self.s2_topk = cfg.MODEL.VG.S2_TOPK
        self.box2box_translation = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        self.reg_iou = cfg.MODEL.VG.REG_IOU
        self.reg_loss_factor = cfg.MODEL.VG.REG_LOSS_FACTOR
        self.rel_cls_loss_factor = cfg.MODEL.VG.REL_CLS_LOSS_FACTOR
        self.sem_nouns_cls_factor = cfg.MODEL.VG.SEM_NOUNS_LOSS_FACTOR
        self.binary_cross_entropy = torch.nn.BCEWithLogitsLoss(weight=get_rel_weights(10), reduction='mean')
        self.binary_cross_entropy_nouns = torch.nn.BCEWithLogitsLoss(weight=get_nouns_weights(10), reduction='mean')
        self.binary_cross_entropy_attrs = torch.nn.BCEWithLogitsLoss(weight=get_attrs_weights(10), reduction='mean')

    def __call__(self, batch_phrase_mask, batch_decode_logits, batch_topk_decoder_logits, batch_phrase_dec_ids, batch_ctx_sim, batch_ctx_sim_s1,
                 batch_pred_similarity, batch_topk_pred_similarity, batch_boxes_targets, batch_precomp_boxes, batch_pred_targets, batch_topk_pred_targets, batch_topk_precomp_boxes,
                 batch_rel_cls_s0, batch_rel_cls_s1, batch_rel_cls_gt, batch_sem_nouns_cls_s0, batch_sem_nouns_cls_s1, batch_semantic_nouns, batch_semantic_attr_cls_s0, batch_semantic_attr_cls_s1, batch_semantic_attrs):

        noun_reconst_loss = torch.zeros(1).to(self.device)
        noun_topk_reconst_loss = torch.zeros(1).to(self.device)
        disc_img_sent_loss = torch.zeros(1).to(self.device)
        disc_img_sent_loss_s1 = torch.zeros(1).to(self.device)

        reg_loss = torch.zeros(1).to(self.device)
        reg_loss_s1 = torch.zeros(1).to(self.device)

        rel_cls_loss = torch.zeros(1).to(self.device)
        rel_cls_loss_s1 = torch.zeros(1).to(self.device)
        sem_nouns_cls_loss = torch.zeros(1).to(self.device)
        sem_nouns_cls_loss_s1 = torch.zeros(1).to(self.device)

        sem_attrs_cls_loss = torch.zeros(1).to(self.device)
        sem_attrs_cls_loss_s1 = torch.zeros(1).to(self.device)

        self.storage = get_event_storage()

        for (phrase_mask, decode_logits, decode_topk_logits, phrase_dec_ids, pred_sim, pred_topk_sim,
             targets, precomp_boxes, pred_delta_s1, pred_delta_s2, topk_precomp_boxes, rel_cls_s0, rel_cls_s1, rel_cls_gt, sem_nouns_cls_s0,
             sem_nouns_cls_s1, sem_nouns_gt, sem_attrs_cls_s0, sem_attrs_cls_s1, sem_attrs_gt) in zip(batch_phrase_mask, batch_decode_logits, batch_topk_decoder_logits, batch_phrase_dec_ids,
                                            batch_pred_similarity, batch_topk_pred_similarity, batch_boxes_targets, batch_precomp_boxes, batch_pred_targets,
                                            batch_topk_pred_targets, batch_topk_precomp_boxes, batch_rel_cls_s0, batch_rel_cls_s1, batch_rel_cls_gt,
                                            batch_sem_nouns_cls_s0, batch_sem_nouns_cls_s1, batch_semantic_nouns, batch_semantic_attr_cls_s0, batch_semantic_attr_cls_s1, batch_semantic_attrs):

            phrase_dec_ids = torch.as_tensor(phrase_dec_ids).long().to(self.device)
            vx, vy = (phrase_mask > 0).nonzero().transpose(0, 1)
            noun_reconst_loss += self.cross_entropy(decode_logits[vx, vy], phrase_dec_ids[vx, vy])
            noun_topk_reconst_loss += self.cross_entropy(decode_topk_logits[vx, vy], phrase_dec_ids[vx, vy])


            if rel_cls_gt is not None:
                rel_cls_index = rel_cls_gt.sum(1).nonzero().squeeze(1)
                if rel_cls_index.shape[0] > 0:
                    rel_cls_gt = rel_cls_gt[rel_cls_index]
                    rel_cls_s0 = rel_cls_s0[rel_cls_index]
                    rel_cls_loss += self.rel_cls_loss_factor * self.binary_cross_entropy(rel_cls_s0, rel_cls_gt)
                    rel_cls_s1 = rel_cls_s1[rel_cls_index]
                    rel_cls_loss_s1 += self.rel_cls_loss_factor*self.binary_cross_entropy(rel_cls_s1, rel_cls_gt)

            if self.storage.iter >= cfg.SOLVER.REG_START_ITER:
                # sem_nouns_index = sem_nouns_gt.sum(1).nonzero().squeeze(1)
                # if sem_nouns_index.shape[0] > 0:
                #     sem_nouns_gt = sem_nouns_gt[sem_nouns_index]
                #     # sem_nouns_cls_s0 = sem_nouns_cls_s0[sem_nouns_index]
                #     # sem_nouns_cls_loss += self.sem_nouns_cls_factor * self.binary_cross_entropy_nouns(sem_nouns_cls_s0, sem_nouns_gt)
                #     sem_nouns_cls_s1 = sem_nouns_cls_s1[sem_nouns_index]
                #     sem_nouns_cls_loss_s1 += self.sem_nouns_cls_factor * self.binary_cross_entropy_nouns(sem_nouns_cls_s1, sem_nouns_gt)

                sem_attrs_index = sem_attrs_gt.sum(1).nonzero().squeeze(1)
                if sem_attrs_index.shape[0] > 0:
                    sem_attrs_gt = sem_attrs_gt[sem_attrs_index]
                    # sem_attrs_cls_s0 = sem_attrs_cls_s0[sem_attrs_index]
                    # sem_attrs_cls_loss += self.sem_nouns_cls_factor * self.binary_cross_entropy_attrs(sem_attrs_cls_s0, sem_attrs_gt)
                    sem_attrs_cls_s1 = sem_attrs_cls_s1[sem_attrs_index]
                    sem_attrs_cls_loss_s1 += self.sem_nouns_cls_factor * self.binary_cross_entropy_attrs(sem_attrs_cls_s1, sem_attrs_gt)

            # ## regression loss
            reg_loss += self.reg_loss_factor * non_maximum_regression_loss_stage1(pred_delta_s1, precomp_boxes, pred_sim, cfg.MODEL.VG.REG_GAP_SCORE, self.reg_iou, self.box2box_translation)
            reg_loss_s1 += self.reg_loss_factor * non_maximum_regression_loss_stage2(pred_delta_s2, topk_precomp_boxes, pred_topk_sim, cfg.MODEL.VG.REG_GAP_SCORE, self.reg_iou, self.box2box_translation)

        disc_img_sent_loss += generate_img_sent_discriminative_loss(batch_ctx_sim, self.margin)
        disc_img_sent_loss_s1 += generate_img_sent_discriminative_loss(batch_ctx_sim_s1, self.margin)

        return noun_reconst_loss, noun_topk_reconst_loss, disc_img_sent_loss, disc_img_sent_loss_s1, reg_loss, reg_loss_s1, \
               rel_cls_loss, rel_cls_loss_s1, sem_nouns_cls_loss, sem_nouns_cls_loss_s1, sem_attrs_cls_loss, sem_attrs_cls_loss_s1



def generate_img_sent_discriminative_loss(batch_ctx_sim, margin=0.2):

    masks = torch.as_tensor(1 - np.eye(batch_ctx_sim.shape[0])).float().to(torch.device('cuda'))
    batch_ctx_sim_dim1 = numerical_stability_softmax(batch_ctx_sim, dim=1)
    ctx_sim_diag_dim1 = torch.diag(batch_ctx_sim_dim1)
    ctx_sim_v1 = batch_ctx_sim_dim1 - ctx_sim_diag_dim1.unsqueeze(1) + margin
    ctx_sim_v1 = F.relu(ctx_sim_v1 * masks).max(1)[0].sum()

    batch_ctx_sim_dim0 =  numerical_stability_softmax(batch_ctx_sim, dim=0)
    ctx_sim_diag_dim0 = torch.diag(batch_ctx_sim_dim0)
    ctx_sim_v0 = batch_ctx_sim_dim0 - ctx_sim_diag_dim0.unsqueeze(0) + margin
    ctx_sim_v0 = F.relu(ctx_sim_v0 * masks).max(0)[0].sum()
    dis_img_sent = ctx_sim_v0 + ctx_sim_v1

    return dis_img_sent


def generate_attention_entropy_loss(pred_sim, pred_topk_sim, targets, precomp_boxes, s2_topk=5):

    pred_sim = torch.clamp(pred_sim, min=1e-5)
    pred_topk_sim = torch.clamp(pred_topk_sim, min=1e-5)

    ious = pairwise_iou(targets, precomp_boxes)
    atten_topk = torch.topk(pred_sim, k=s2_topk, dim=1)[1]
    ious_topk = torch.gather(ious, dim=1, index=atten_topk)
    gt_score = F.normalize(ious * ious.ge(0.5).float(), p=1, dim=1)
    gt_score_topk = F.normalize(ious_topk*ious_topk.ge(0.5).float(), p=1, dim=1)

    cls_loss = -(gt_score * pred_sim.log()).mean()
    cls_loss_topk = -(gt_score_topk*pred_topk_sim.log()).mean()

    return cls_loss, cls_loss_topk


def noisy_contrastive_estimation(batch_ctx_sim, temp=0.05):

    """
    InfoNCE, which originate from noisy-contrastive-estimation
    Aims to maximize the mutual information between visual and language
    """

    batch_ctx_sim = batch_ctx_sim / temp

    batch_size = batch_ctx_sim.shape[0]
    label = torch.arange(0, batch_size).to(torch.device('cuda')).long()

    loss_v1 = F.cross_entropy(batch_ctx_sim, label, reduction='sum')  ## image-sents pairs
    loss_v2 = F.cross_entropy(batch_ctx_sim.permute(1, 0), label, reduction='sum') ## images-sent pairs
    loss = (loss_v2 + loss_v1)/2

    return loss


def non_maximum_regression_loss_stage1(box_reg_delta, proposals, box_score, score_gap=0.2, reg_iou=0.6, box2box_translation=None):

    device = torch.device('cuda')
    loss = torch.zeros(1).to(device)
    box_score_topk, box_score_index = torch.topk(box_score, k=1, dim=1)
    box_score_topk, box_score_index = box_score_topk.reshape(-1), box_score_index.reshape(-1)
    proposals_max = proposals[box_score_index]  ## (M)*4
    proposals_ious = pairwise_iou(proposals_max, proposals)
    proposals_ious[proposals_ious==1] = 0
    proposals_ious = (proposals_ious >= reg_iou).float()
    if proposals_ious.sum():
        vx, vy = proposals_ious.nonzero().transpose(0, 1)
        proposals_reg_target = proposals_max[vx]
        box_score_targets = box_score_topk[vx]
        box_score_reg = box_score[vx, vy].reshape(-1)
        score_att = (box_score_targets - box_score_reg) >= score_gap
        proposals_need_reg = proposals[vy]
        proposal_target = box2box_translation.get_deltas(proposals_need_reg.tensor, proposals_reg_target.tensor)
        reg_delta = box_reg_delta[vx, vy]
        loss += weighted_smooth_l1_loss(reg_delta, proposal_target, beta=1, weight=score_att.float(), reduction='mean')

    return loss


def non_maximum_regression_loss_stage2_ada(box_reg_delta, proposals, box_score, score_gap=0.2, reg_iou=0.6, box2box_translation=None, atten_mask=None):

    device = torch.device('cuda')
    loss = torch.zeros(1).to(device)
    box_score = box_score * atten_mask
    box_reg_delta = box_reg_delta.reshape(-1, 4)
    num_phrase, num_boxes = box_score.shape
    box_score = box_score.reshape(-1)
    box_score_topk, box_score_index = torch.topk(box_score.reshape(num_phrase, num_boxes), k=1, dim=1)
    box_score_topk, box_score_index = box_score_topk.reshape(-1), box_score_index.reshape(-1)
    box_score_index = box_score_index + torch.arange(num_phrase).to(device)*num_boxes
    proposals_max = proposals[box_score_index]  ## M*4
    proposals_ious = pairwise_iou(proposals_max, proposals)
    proposals_ious[proposals_ious == 1] = 0
    keep_mat = torch.zeros_like(proposals_ious)
    for pid in range(num_phrase):
        keep_mat[pid, pid*num_boxes:(pid+1)*num_boxes] = 1

    keep_mat = keep_mat * atten_mask.reshape(-1)

    proposals_ious = proposals_ious * keep_mat
    proposals_ious_flag = (proposals_ious>=reg_iou).float()

    if proposals_ious_flag.sum():
        vx, vy = proposals_ious_flag.nonzero().transpose(0,1)
        proposals_reg_target = proposals_max[vx]
        proposals_need_reg = proposals[vy]
        box_score_target = box_score_topk[vx]
        box_score_reg = box_score[vy]
        score_att = (box_score_target - box_score_reg)>=score_gap
        proposal_target = box2box_translation.get_deltas(proposals_need_reg.tensor, proposals_reg_target.tensor)
        reg_delta = box_reg_delta[vy]
        loss += weighted_smooth_l1_loss(reg_delta, proposal_target, beta=1, weight=score_att.float(), reduction='mean')

    return loss


def non_maximum_regression_loss_stage2(box_reg_delta, proposals, box_score, score_gap=0.2, reg_iou=0.6, box2box_translation=None):

    device = torch.device('cuda')
    loss = torch.zeros(1).to(device)
    box_reg_delta = box_reg_delta.reshape(-1, 4)
    num_phrase, num_boxes = box_score.shape
    box_score = box_score.reshape(-1)
    box_score_topk, box_score_index = torch.topk(box_score.reshape(num_phrase, num_boxes), k=1, dim=1)
    box_score_topk, box_score_index = box_score_topk.reshape(-1), box_score_index.reshape(-1)
    box_score_index = box_score_index + torch.arange(num_phrase).to(device)*num_boxes
    proposals_max = proposals[box_score_index]  ## M*4
    proposals_ious = pairwise_iou(proposals_max, proposals)
    proposals_ious[proposals_ious == 1] = 0
    keep_mat = torch.zeros_like(proposals_ious)
    for pid in range(num_phrase):
        keep_mat[pid, pid*num_boxes:(pid+1)*num_boxes] = 1

    proposals_ious = proposals_ious * keep_mat
    proposals_ious_flag = (proposals_ious>=reg_iou).float()

    if proposals_ious_flag.sum():
        vx, vy = proposals_ious_flag.nonzero().transpose(0,1)
        proposals_reg_target = proposals_max[vx]
        proposals_need_reg = proposals[vy]
        box_score_target = box_score_topk[vx]
        box_score_reg = box_score[vy]
        score_att = (box_score_target - box_score_reg)>=score_gap
        proposal_target = box2box_translation.get_deltas(proposals_need_reg.tensor, proposals_reg_target.tensor)
        reg_delta = box_reg_delta[vy]
        loss += weighted_smooth_l1_loss(reg_delta, proposal_target, beta=1, weight=score_att.float(), reduction='mean')

    return loss


def get_rel_weights(scale=10):

    cnts = np.array([35292.,  6563.,   217.,  2225., 52291.,   544.,  2989.,  9084.,
        1733.,   346.,  8986.,  6205., 21247.,   792., 46366.,  2369.,
        2942.,  1662.,  9530.,  2969.,  3437.,   216.,   611.,  1534.,
        2147.,   483.,  3114.,  1200.,   474.,  2276.,  1781.,  1474.,
         289.,  1957.,   563.,  3958.,   318.,   431.,   867.,   249.,
        3239.,   772.,   189.,   311.,   705.,   259.,   893.,   445.,
         291.,   370.,   275.,   862.,   753.,   340.,   347.,   967.,
         939.,   323.,  1827.,   238.,   383.,   668.,   568.,  1971.,
         930.,   536.,  1107.,   297.,   326.,  1052.,   239.,   240.,
         182.,   704.,   553.,   204.,   334.,   446.,   158.,   273.,
         220.,   363.,   320.,   363.,   559.,   317.,   391.,   293.])

    weights = 1/cnts**0.5
    weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    weights = weights * (scale-1) + 1
    weights = torch.as_tensor(weights).float().to(torch.device('cuda'))
    return weights


def get_nouns_weights(scale=10):

    nouns2cnt = json.load(open(cfg.MODEL.VG.SEMANTIC_NOUNS_PATH, 'r'))['counts'].values()
    cnts = np.array(list(nouns2cnt))[:cfg.MODEL.VG.SEMANTIC_NOUNS_TOPK]
    weights = 1 / cnts ** 0.5
    weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    weights = weights * (scale - 1) + 1
    weights = torch.as_tensor(weights).float().to(torch.device('cuda'))
    return weights


def get_attrs_weights(scale=10):

    nouns2cnt = json.load(open(cfg.MODEL.VG.SEMANTIC_ATTR_PATH, 'r'))['counts'].values()
    cnts = np.array(list(nouns2cnt))[:cfg.MODEL.VG.SEMANTIC_ATTR_TOPK]
    weights = 1 / cnts ** 0.5
    weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    weights = weights * (scale - 1) + 1
    weights = torch.as_tensor(weights).float().to(torch.device('cuda'))
    return weights



