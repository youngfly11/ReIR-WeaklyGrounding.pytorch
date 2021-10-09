#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 10:52


import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from detectron2.config import global_cfg as cfg
from detectron2.modeling.weaklygrounding.loss_kac import WeaklyVGLossCompute
from detectron2.modeling.weaklygrounding.phr_emebdding_kac import PhraseEmbeddingSent
from detectron2.layers.numerical_stability_softmax import numerical_stability_masked_softmax
from detectron2.layers.move2cpu import move2cpu


class KnowledgeAidedConsistencyNet(torch.nn.Module):
    def __init__(self, roi_pooler, RCNN_top):
        super(KnowledgeAidedConsistencyNet, self).__init__()
        self.det_roi_pooler = roi_pooler
        self.rcnn_top = RCNN_top
        self.obj_embed_dim = 2048
        self.img_channel = 1024
        self.phr_dim = 1024
        self.vis_dim = 1024
        self.device = torch.device('cuda')

        self.phrase_embed = PhraseEmbeddingSent(cfg, phrase_embed_dim=self.phr_dim, bidirectional=cfg.MODEL.VG.LSTM_BIDIRECTION)

        self.sim_fc = nn.Linear(self.phr_dim*2, 1) ## first is confidence score and last 4 is location
        self.box_delta = nn.Sequential(nn.Linear(self.phr_dim*3, 256), nn.ReLU(inplace=True), nn.Linear(256, 4))

        self.vis_bn_pro = nn.BatchNorm1d(num_features=1024)
        self.phr_batchnorm = nn.BatchNorm1d(num_features=1024)

        self.att_vis = nn.Sequential(nn.Linear(self.obj_embed_dim, self.phr_dim), nn.ReLU(inplace=True))
        self.dec_phrase_gru = nn.GRU(input_size=self.vis_dim, hidden_size=self.vis_dim, num_layers=1, batch_first=True, dropout=0, bias=True)
        self.vis2phr = nn.Linear(self.vis_dim, self.phrase_embed.phr_vocab_size)
        self.VGLoss = WeaklyVGLossCompute()


    def forward(self, features, all_phrase_ids, targets, precomp_boxes, precomp_score,
                precomp_det_label, image_scale, all_sent_sgs, all_sentences, image_unique_id, det_label_embedding):

        """
        :param obj_proposals: proposal from each images
        :param features: features maps from the backbone
        :param target: gt relation labels
        :param object_vocab, object_vocab_len [[xxx,xxx],[xxx],[xxx]], [2,1,1]
        :param sent_sg: sentence scene graph
        :return: prediction, loss

        note that first dimension is images
        """
        img_num_per_gpu = len(features)

        batch_decode_logits = []
        batch_pred_similarity = []
        batch_precomp_boxes = []
        batch_pred_boxes = []
        batch_boxes_targets = []
        batch_pred_delta = []
        batch_gt_delta = []


        """ Language Embedding"""
        batch_phrase_ids, batch_phrase_types, batch_phrase_embed, batch_phrase_len, batch_phrase_dec_ids, batch_phrase_mask, batch_decoder_word_embed, batch_phrase_glove_embed, batch_sent_embed = \
            self.phrase_embed(all_sentences, all_phrase_ids, all_sent_sgs)


        h, w = features.shape[-2:]

        for bid in range(img_num_per_gpu):

            """ Visual Embedding """

            precomp_boxes_bid = precomp_boxes[bid].to(self.device)  ## 100*4
            ## extract the groundtruth bounding boxes for evaluation
            order = []
            for phr_ids in batch_phrase_ids[bid]:
                order.append(all_phrase_ids[bid].index(phr_ids))
            target_filter = targets[bid][np.array(order)]
            batch_boxes_targets.append(target_filter.to(self.device))
            batch_precomp_boxes.append(precomp_boxes_bid)

            img_feat_bid = features[[bid]]
            visual_features_bid = self.rcnn_top(self.det_roi_pooler(tuple([img_feat_bid]), [precomp_boxes_bid])).mean(dim=[2, 3]).contiguous()
            visual_global_feat = self.rcnn_top(img_feat_bid).mean(dim=[2,3]) ## global_feat

            visual_features_bid = torch.cat((visual_features_bid, visual_global_feat), dim=0)

            phrase_embed_bid = batch_phrase_embed[bid]
            if phrase_embed_bid.shape[0] == 1 and self.training:
                phrase_embed_bid = self.phr_batchnorm(phrase_embed_bid.repeat(2, 1))[[0]]
            else:
                phrase_embed_bid = self.phr_batchnorm(phrase_embed_bid)


            ## calculate the gt_delta
            precomp_boxes_tensor, box_size = precomp_boxes_bid.tensor.clone().to(self.device), precomp_boxes_bid.size
            precomp_boxes_tensor = precomp_boxes_tensor / torch.as_tensor(np.array(box_size[::-1]), dtype=torch.float32).to(self.device).repeat(2)
            batch_gt_delta.append(precomp_boxes_tensor)

            # reconstruct the noun phrase
            visual_features_bid = self.att_vis(visual_features_bid)
            visual_features_bid = self.vis_bn_pro(visual_features_bid)

            visual_global_feat = visual_features_bid[[-1]]
            visual_features_bid = visual_features_bid[:-1]

            num_phr = phrase_embed_bid.shape[0]
            num_box = visual_features_bid.shape[0]
            ind_phr, ind_box = self.make_pair(num_phr, num_box)


            mm_feat_c = torch.cat((phrase_embed_bid[ind_phr], visual_features_bid[ind_box]), dim=1)
            sim = self.sim_fc(mm_feat_c)  ## 300*5
            sim_score = sim.reshape(num_phr, num_box)  ## 5*50

            mm_feat = torch.cat((visual_features_bid, visual_global_feat.repeat(num_box, 1)), dim=1)
            box_delta = self.box_delta(torch.cat((phrase_embed_bid[ind_phr], mm_feat[ind_box]), dim=1))
            box_delta = box_delta.reshape(num_phr, num_box, 4)  ## 5*50*4


            det_embed_bid = det_label_embedding[bid].to(self.device)
            det_sim = self.cal_det_sim(det_embed_bid, batch_phrase_glove_embed[bid])
            sim_score = sim_score * det_sim
            sim_mask = (det_sim > 0).float()
            atten_bid = numerical_stability_masked_softmax(sim_score, sim_mask, dim=1)

            visual_reconst_bid = torch.mm(atten_bid, visual_features_bid)
            decode_phr_logits = self.phrase_decoder(visual_reconst_bid, batch_decoder_word_embed[bid])
            batch_decode_logits.append(decode_phr_logits)

            batch_pred_similarity.append(atten_bid)
            batch_pred_delta.append(box_delta)
            max_box_ind = atten_bid.detach().cpu().numpy().argmax(1)
            batch_pred_boxes.append(precomp_boxes_bid[max_box_ind])

        noun_reconst_loss, vis_consistency_loss = self.VGLoss(batch_phrase_mask, batch_decode_logits, batch_phrase_dec_ids, batch_pred_delta, batch_gt_delta, batch_pred_similarity)
        all_loss = dict(noun_reconst_loss=noun_reconst_loss, vis_consistency_loss=vis_consistency_loss)

        if self.training:
            return all_loss, None
        else:
            return all_loss, (batch_phrase_ids, batch_phrase_types, move2cpu(batch_pred_boxes),
                              move2cpu(batch_pred_similarity), move2cpu(batch_boxes_targets), move2cpu(batch_precomp_boxes), image_unique_id)


    def phrase_decoder(self, visual_reconst, phrase_enc_embed):
        """
        Phrase_enc_embed, bt*max_length*embedding_dim
        visual_reconst: bt*embedding_dim
        """
        bs, maxl, embed_dim = phrase_enc_embed.shape
        visual_reconst = visual_reconst.unsqueeze(1)
        # visual_fake = torch.zeros_like(visual_reconst).repeat(1, maxl-1, 1).to(self.device)
        # visual_reconst = torch.cat((visual_reconst, visual_fake), dim=1)
        enc_embed = torch.cat((visual_reconst, phrase_enc_embed), dim=1)
        dec_emb, _ = self.dec_phrase_gru(enc_embed)
        decode_phr_logits = self.vis2phr(dec_emb)

        return decode_phr_logits


    def cal_det_sim(self, det_embed_bid, decoder_word_embed):

        sim = []
        for pid, phr_embed in enumerate(decoder_word_embed):
            sim_pid = F.cosine_similarity(phr_embed.unsqueeze(1), det_embed_bid.unsqueeze(0), dim=2).mean(0)
            sim.append(sim_pid)
        sim = torch.stack(sim, dim=0)
        sim = F.relu(sim)  ## ignore the negative reponse
        return sim


    def make_pair(self, phr_num, box_num):
        ind_phr, ind_box = np.meshgrid(range(phr_num), range(box_num), indexing='ij')
        ind_phr = ind_phr.reshape(-1)
        ind_box = ind_box.reshape(-1)
        return ind_phr, ind_box


def build_kac_head(cfg, det_roi_heads):
    return KnowledgeAidedConsistencyNet(cfg, det_roi_heads)