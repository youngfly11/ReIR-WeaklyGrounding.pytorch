#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/5/9 09:29



import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from detectron2.config import global_cfg as cfg
from detectron2.modeling.weaklygrounding.loss import WeaklyVGLossComputeV3
from detectron2.modeling.weaklygrounding.phrase_embedding_weakly_v1 import PhraseEmbeddingSent
from detectron2.layers.spatial_coordinate import meshgrid_generation
from detectron2.layers.numerical_stability_softmax import numerical_stability_masked_softmax, numerical_stability_softmax
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures.boxes import Boxes
from detectron2.utils.events import get_event_storage
from detectron2.layers.move2cpu import move2cpu


class DetProposalVGHead(torch.nn.Module):
    def __init__(self, roi_pooler, RCNN_top):
        super(DetProposalVGHead, self).__init__()

        self.det_roi_pooler = roi_pooler
        self.rcnn_top = RCNN_top
        self.obj_embed_dim = 2048
        self.img_channel = 1024
        self.phrase_embed_dim = 1024
        self.rnn_hidden_dim = 1024
        self.device = torch.device('cuda')
        self.s2_topk = cfg.MODEL.VG.S2_TOPK

        self.phrase_embed = PhraseEmbeddingSent(cfg, phrase_embed_dim=self.phrase_embed_dim, bidirectional=cfg.MODEL.VG.LSTM_BIDIRECTION)
        self.visual_embed_dim = self.phrase_embed_dim
        self.box2box_translation = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

        if cfg.MODEL.VG.SPATIAL_FEAT:
            self.spatial_trans = nn.Sequential(nn.Linear(2*cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION ** 2, 256), nn.ReLU(inplace=True))
            self.obj_embed_dim = self.obj_embed_dim + 256

        self.visual_embedding = nn.Sequential(nn.Linear(self.obj_embed_dim, self.visual_embed_dim), nn.ReLU(inplace=True))
        self.similarity_fc = nn.Sequential(nn.Linear(self.phrase_embed_dim*2, 1))
        self.box_reg_fc = nn.Sequential(nn.Linear(self.phrase_embed_dim*2, 256), nn.ReLU(inplace=True), nn.Linear(256, 4))

        self.vis_batchnorm = nn.BatchNorm1d(num_features=1024)
        self.phr_batchnorm = nn.BatchNorm1d(num_features=1024)

        self.sent_mlp = nn.Linear(self.phrase_embed_dim, 512)
        self.visual_mlp = nn.Linear(self.phrase_embed_dim, 512)
        self.dec_phrase_gru = nn.GRU(input_size=self.visual_embed_dim, hidden_size=self.rnn_hidden_dim, num_layers=1, batch_first=True, dropout=0, bias=True)

        self.vis2phr = nn.Linear(self.rnn_hidden_dim, self.phrase_embed.phr_vocab_size)
        self.VGLoss = WeaklyVGLossComputeV3()
        self.iter = 100000


    def init_weights(self):
        # Initialize LSTM Weights and Biases

        nn.init.kaiming_normal_(self.visual_embedding[0].weight.data)
        self.visual_embedding[0].bias.data.zero_()

        nn.init.kaiming_normal_(self.similarity_fc[0].weight.data)
        self.similarity_fc[0].bias.data.zero_()
        nn.init.kaiming_normal_(self.vis2phr.weight.data)
        self.vis2phr.bias.data.zero_()

        for layer in self.dec_phrase_gru._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.dec_phrase_gru, param_name)
                    nn.init.xavier_normal_(weight.data)
                else:
                    bias = getattr(self.dec_phrase_gru, param_name)
                    bias.data.zero_()
                    # nn.init.uniform_(bias.data, a=-0.01, b=0.01)
        # nn.init.uniform_(self.enc_embedding.weight.data, a=-0.01, b=0.01)

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
        batch_topk_decoder_logits = []
        batch_pred_similarity = []
        batch_precomp_boxes = []
        batch_topk_precomp_boxes=[]
        batch_pred_boxes = []
        batch_topk_pred_boxes = []
        batch_topk_fusion_pred_boxes = []
        batch_topk_pred_similarity = []
        batch_topk_fusion_similarity = []
        batch_boxes_targets = []
        batch_ctx_embed = []
        batch_ctx_s1_embed = []

        batch_pred_targets = []
        batch_topk_pred_targets = []


        """ Language Embedding"""
        batch_phrase_ids, batch_phrase_types, batch_phrase_embed, batch_phrase_len, \
        batch_phrase_dec_ids, batch_phrase_mask, batch_decoder_word_embed, batch_phrase_glove_embed, batch_rel_phrase_embed, batch_relation_conn, batch_sent_embed,\
        batch_decoder_rel_word_embed, batch_rel_mask, batch_rel_dec_idx = self.phrase_embed(all_sentences, all_phrase_ids, all_sent_sgs)

        h, w = features.shape[-2:]

        # self.storage = get_event_storage()


        for bid in range(img_num_per_gpu):

            """ Visual Embedding """
            precomp_boxes_bid = precomp_boxes[bid].to(self.device)  ## 100*4

            order = []
            for phr_ids in batch_phrase_ids[bid]:
                order.append(all_phrase_ids[bid].index(phr_ids))
            target_filter = targets[bid][np.array(order)]
            batch_boxes_targets.append(target_filter.to(self.device))
            batch_precomp_boxes.append(precomp_boxes_bid)

            img_feat_bid = features[[bid]]
            visual_features_bid = self.rcnn_top(self.det_roi_pooler([img_feat_bid], [precomp_boxes_bid])).mean(dim=[2, 3]).contiguous()
            if cfg.MODEL.VG.SPATIAL_FEAT:
                spa_feat = meshgrid_generation(h, w)
                spa_feat = self.det_roi_pooler([spa_feat], [precomp_boxes_bid]).view(visual_features_bid.shape[0], -1)
                spa_feat = self.spatial_trans(spa_feat)
                visual_features_bid = torch.cat((visual_features_bid, spa_feat), dim=1)

            visual_features_bid = self.visual_embedding(visual_features_bid)
            visual_features_bid = self.vis_batchnorm(visual_features_bid)

            """ Noun Phrase embedding """
            phrase_embed_bid = batch_phrase_embed[bid]
            if phrase_embed_bid.shape[0] == 1 and self.training:
                phrase_embed_bid = self.phr_batchnorm(phrase_embed_bid.repeat(2,1))[[0]]
            else:
                phrase_embed_bid = self.phr_batchnorm(phrase_embed_bid)


            """ Similarity and attention prediction """
            num_box = precomp_boxes_bid.tensor.size(0)
            num_phrase = phrase_embed_bid.size(0)
            phr_inds, obj_inds = self.make_pair(num_phrase, num_box)
            pred_similarity_bid, pred_targets_bid = self.similarity(visual_features_bid, phrase_embed_bid, obj_inds, phr_inds)
            pred_similarity_bid = pred_similarity_bid.reshape(num_phrase, num_box)
            pred_targets_bid = pred_targets_bid.reshape(num_phrase, num_box, 4)
            batch_pred_targets.append(pred_targets_bid)


            if cfg.MODEL.VG.USING_DET_KNOWLEDGE :
                det_label_embedding_bid = det_label_embedding[bid].to(self.device)
                sim = self.cal_det_label_sim_max(det_label_embedding_bid, batch_phrase_glove_embed[bid])
                pred_similarity_bid = pred_similarity_bid * sim
                sim_mask = (sim > 0).float()
                atten_bid = numerical_stability_masked_softmax(pred_similarity_bid, sim_mask, dim=1)
            else:
                atten_bid = F.softmax(pred_similarity_bid, dim=1)

            ## reconstruction visual features
            visual_reconst_bid = torch.mm(atten_bid, visual_features_bid)
            decode_phr_logits = self.phrase_decoder(visual_reconst_bid, batch_decoder_word_embed[bid])
            batch_decode_logits.append(decode_phr_logits)

            atten_score_topk, atten_ranking_topk = torch.topk(atten_bid, dim=1, k=self.s2_topk) ## (N, 10)
            ind_phr_topk = np.arange(num_phrase).repeat(self.s2_topk)


            ## -----------------------------------------------------##
            ## crop 2st features
            ## -----------------------------------------------------##

            if self.storage.iter <= cfg.SOLVER.REG_START_ITER:
                visual_features_topk_bid = visual_features_bid[atten_ranking_topk.reshape(-1)]
                precomp_boxes_topk_bid = precomp_boxes_bid[atten_ranking_topk.reshape(-1)]
                batch_topk_precomp_boxes.append(precomp_boxes_topk_bid)
            else:
                topk_box_ids = atten_ranking_topk.reshape(-1) + torch.as_tensor(ind_phr_topk, dtype=torch.long).to(self.device)*num_box
                precomp_boxes_tensor, box_size = precomp_boxes_bid.tensor, precomp_boxes_bid.size
                precomp_boxes_topk_tensor = precomp_boxes_tensor[atten_ranking_topk.reshape(-1)]  ## (N*10, 4)
                pred_targets_s0 = pred_targets_bid.view(-1, 4)[topk_box_ids]
                precomp_boxes_topk_bid = self.box2box_translation.apply_deltas(pred_targets_s0, precomp_boxes_topk_tensor)
                precomp_boxes_topk_bid = Boxes(precomp_boxes_topk_bid, box_size)
                precomp_boxes_topk_bid.clip()
                batch_topk_precomp_boxes.append(precomp_boxes_topk_bid)
                visual_features_topk_bid = self.rcnn_top(self.det_roi_pooler([img_feat_bid], [precomp_boxes_topk_bid])).mean(dim=[2, 3]).contiguous()

                if cfg.MODEL.VG.SPATIAL_FEAT:
                    spa_feat = meshgrid_generation(h, w)
                    spa_feat = self.det_roi_pooler([spa_feat], [precomp_boxes_topk_bid]).view(visual_features_topk_bid.shape[0], -1)
                    spa_feat = self.spatial_trans(spa_feat)
                    visual_features_topk_bid = torch.cat((visual_features_topk_bid, spa_feat), dim=1)

                visual_features_topk_bid = self.visual_embedding(visual_features_topk_bid)## (N*10, 1024)
                visual_features_topk_bid = self.vis_batchnorm(visual_features_topk_bid)


            pred_similarity_topk_bid, pred_targets_topk_bid = self.similarity_topk(visual_features_topk_bid, phrase_embed_bid, ind_phr_topk)
            pred_similarity_topk_bid = pred_similarity_topk_bid.reshape(num_phrase, self.s2_topk)
            pred_targets_topk_bid = pred_targets_topk_bid.reshape(num_phrase, self.s2_topk, 4)
            batch_topk_pred_targets.append(pred_targets_topk_bid)


            if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
                sim_topk = torch.gather(sim, dim=1, index=atten_ranking_topk.long())
                sim_mask = (sim_topk>0).float()
                pred_similarity_topk_bid = pred_similarity_topk_bid * sim_topk
                atten_topk_bid = numerical_stability_masked_softmax(pred_similarity_topk_bid, sim_mask, dim=1)
            else:
                atten_topk_bid = F.softmax(pred_similarity_topk_bid, dim=1)

            atten_fusion = atten_topk_bid * atten_score_topk  ## N*10
            visual_features_topk_bid = visual_features_topk_bid.view(num_phrase, self.s2_topk, -1)
            visual_reconst_topk_bid = (atten_fusion.unsqueeze(2)*visual_features_topk_bid).sum(1) ## N*1024
            decoder_phr_topk_logits = self.phrase_decoder(visual_reconst_topk_bid, batch_decoder_word_embed[bid])
            batch_topk_decoder_logits.append(decoder_phr_topk_logits)


            ## construct the discriminative loss
            batch_ctx_s1_embed.append(self.visual_mlp(visual_reconst_bid.mean(0, keepdim=True)))
            batch_ctx_embed.append(self.visual_mlp(visual_reconst_topk_bid.mean(0, keepdim=True)))


            batch_pred_similarity.append(atten_bid)
            batch_topk_pred_similarity.append(atten_topk_bid)
            batch_topk_fusion_similarity.append(atten_fusion)

            ### transform boxes for stage-1
            num_phrase_indices = torch.arange(num_phrase).long().to(self.device)
            max_box_ind = atten_bid.detach().cpu().numpy().argmax(1)
            precomp_boxes_delta_max = pred_targets_bid[num_phrase_indices, max_box_ind] ## numPhrase*4

            max_topk_id = torch.topk(atten_topk_bid, dim=1, k=1)[1].long().squeeze(1)
            precomp_boxes_delta_max_topk = pred_targets_topk_bid[num_phrase_indices, max_topk_id]  ## num_phrase*4
            precomp_boxes_topk_bid_tensor = precomp_boxes_topk_bid.tensor.reshape(-1, self.s2_topk, 4)

            max_fusion_topk_id = torch.topk(atten_fusion, dim=1, k=1)[1].long().squeeze()
            precomp_boxes_delta_max_topk_fusion = pred_targets_topk_bid[num_phrase_indices, max_fusion_topk_id]  ## num_phrase*4

            phr_index = torch.arange(num_phrase).to(self.device) * self.s2_topk

            if self.storage.iter <= cfg.SOLVER.REG_START_ITER:
                max_select_boxes = precomp_boxes_bid[max_box_ind]
                max_precomp_boxes = precomp_boxes_topk_bid[max_topk_id + phr_index]
                max_fusion_precomp_boxes = precomp_boxes_topk_bid[max_fusion_topk_id + phr_index]
            else:
                max_select_boxes = Boxes(self.box2box_translation.apply_deltas(precomp_boxes_delta_max, precomp_boxes_bid[max_box_ind].tensor), precomp_boxes_bid.size)
                max_precomp_boxes = Boxes(self.box2box_translation.apply_deltas(precomp_boxes_delta_max_topk, precomp_boxes_topk_bid_tensor[num_phrase_indices, max_topk_id]), precomp_boxes_bid.size)
                max_fusion_precomp_boxes = Boxes(self.box2box_translation.apply_deltas(precomp_boxes_delta_max_topk_fusion, precomp_boxes_topk_bid_tensor[num_phrase_indices, max_fusion_topk_id]), precomp_boxes_bid.size)

            batch_pred_boxes.append(max_select_boxes)
            batch_topk_pred_boxes.append(max_precomp_boxes)
            batch_topk_fusion_pred_boxes.append(max_fusion_precomp_boxes)


        batch_ctx_sim, batch_ctx_sim_s1 = self.generate_image_sent_discriminative(batch_sent_embed, batch_ctx_embed, batch_ctx_s1_embed)

        noun_reconst_loss, noun_topk_reconst_loss, disc_img_sent_loss_s1, disc_img_sent_loss_s2,  reg_loss, \
        reg_loss_s1 = self.VGLoss(batch_phrase_mask, batch_decode_logits, batch_topk_decoder_logits, batch_phrase_dec_ids,
                                  batch_ctx_sim, batch_ctx_sim_s1, batch_pred_similarity, batch_topk_pred_similarity, batch_boxes_targets, batch_precomp_boxes,
                                  batch_pred_targets, batch_topk_pred_targets,
                                  batch_topk_precomp_boxes)

        all_loss = dict(noun_reconst_loss=noun_reconst_loss, noun_topk_reconst_loss=noun_topk_reconst_loss, disc_img_sent_loss_s1=disc_img_sent_loss_s1,
                        disc_img_sent_loss_s2=disc_img_sent_loss_s2, reg_loss_s1=reg_loss, reg_loss_s2=reg_loss_s1)


        if self.training:
            return all_loss, None
        else:
            return all_loss, (batch_phrase_ids, batch_phrase_types, move2cpu(batch_pred_boxes), move2cpu(batch_pred_similarity),
                              move2cpu(batch_boxes_targets), move2cpu(batch_precomp_boxes), image_unique_id, move2cpu(batch_topk_pred_similarity),
                              move2cpu(batch_topk_fusion_similarity), move2cpu(batch_topk_pred_boxes), move2cpu(batch_topk_fusion_pred_boxes),
                              move2cpu(batch_topk_precomp_boxes), move2cpu(batch_topk_pred_targets), move2cpu(batch_pred_targets))


    def phrase_decoder(self, visual_reconst, phrase_enc_embed):
        """
        Phrase_enc_embed, bt*max_length*embedding_dim
        visual_reconst: bt*embedding_dim
        """
        enc_embed = torch.cat((visual_reconst.unsqueeze(1), phrase_enc_embed), dim=1) ## bs*maxl*L
        dec_emb, _ = self.dec_phrase_gru(enc_embed)
        decode_phr_logits = self.vis2phr(dec_emb)
        return decode_phr_logits

    def rel_phrase_decoder(self, visual_reconst, rel_enc_embed):
        """
        Phrase_enc_embed, bt*max_length*embedding_dim
        visual_reconst: bt*embedding_dim
        """

        enc_embed = torch.cat((visual_reconst, rel_enc_embed), dim=1) ## bs*maxl*L
        # dec_emb, _ = self.dec_rel_gru(enc_embed)
        # decode_rel_logits = self.vis2rel(dec_emb)
        dec_emb, _ = self.dec_phrase_gru(enc_embed)
        decode_rel_logits = self.vis2phr(dec_emb)

        return decode_rel_logits

    def generate_image_sent_discriminative(self, batch_sent_embed, batch_ctx_embed, batch_ctx_s1_embed):

        batch_sent_embed = torch.cat(batch_sent_embed, dim=0)  ## b*1024
        batch_sent_embed = self.sent_mlp(batch_sent_embed)  ## N*512
        batch_ctx_embed = torch.cat(batch_ctx_embed, dim=0)  ## N*512
        batch_ctx_s1_embed = torch.cat(batch_ctx_s1_embed, dim=0)  ## N*512
        batch_ctx_sim = torch.mm(batch_ctx_embed, batch_sent_embed.permute(1, 0)) / 512 ** 0.5
        batch_ctx_sim_s1 = torch.mm(batch_ctx_s1_embed, batch_sent_embed.permute(1, 0)) / 512 ** 0.5

        return batch_ctx_sim, batch_ctx_sim_s1


    def cal_det_label_sim_max(self, det_label_embed, decoder_word_embed):

        """
        phrase_mask, to indicate the word  N*L
        det_label_embed:  topk*300
        decoder_word_embed: N*300
        """

        sim = []
        for pid, phr_embed in enumerate(decoder_word_embed):
            sim_pid = F.cosine_similarity(phr_embed.unsqueeze(1), det_label_embed.unsqueeze(0), dim=2).mean(0)
            sim.append(sim_pid)
        sim = torch.stack(sim, dim=0)
        sim = F.relu(sim) ## ignore the negative reponse
        return sim

    def similarity(self, vis_embed, phrase_embed, obj_ind, phr_ind):
        fusion_embed = torch.cat((phrase_embed[phr_ind], vis_embed[obj_ind]), 1)
        pred_similarity = self.similarity_fc(fusion_embed)
        pred_delta = self.box_reg_fc(fusion_embed)
        return pred_similarity, pred_delta

    def similarity_topk(self, vis_embed, phrase_embed, phr_ind_topk):
        fusion_embed = torch.cat((phrase_embed[phr_ind_topk], vis_embed), 1)
        pred_similarity = self.similarity_fc(fusion_embed)
        pred_delta = self.box_reg_fc(fusion_embed)
        return pred_similarity, pred_delta

    def filtering_phrase(self, phrases, all_phrase):
        phrase_valid = []
        for phr in phrases:
            if phr['phrase_id'] in all_phrase:
                phrase_valid.append(phr)
        return phrase_valid

    def make_pair(self, phr_num, box_num):
        ind_phr, ind_box = np.meshgrid(range(phr_num), range(box_num), indexing='ij')
        ind_phr = ind_phr.reshape(-1)
        ind_box = ind_box.reshape(-1)
        return ind_phr, ind_box

    def make_pair_rel(self, rel_num, rel_box_num):
        ind_phr = np.arange(rel_num).repeat(rel_box_num)
        ind_box = np.arange(rel_num*rel_box_num)
        return ind_phr, ind_box


    def make_pair_topN(self, topN_boxes_ids, num_phrase):
        """
        in topN setting, to pair the phrases and objects. Every phrase have it own topN objects. But they save in previous setting.
        So we need to minus the ids into 0~100
        :param topN_boxes_ids: array([[1,2,5],..., [200,210,240],[35,37,xx]]) M*N.
        :param num_phrase: the number of phrases to locate in current sentence. int
        """
        topN = topN_boxes_ids.shape[1]
        ind_phr = np.arange(num_phrase).repeat(topN)
        ind_box = topN_boxes_ids.cpu().numpy().reshape(-1)
        return ind_phr, ind_box


def build_vg_head(cfg, det_roi_heads):
    return DetProposalVGHead(cfg, det_roi_heads)

