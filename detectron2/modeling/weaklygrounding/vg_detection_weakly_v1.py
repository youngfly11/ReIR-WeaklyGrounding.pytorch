import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from detectron2.config import global_cfg as cfg
from detectron2.modeling.weaklygrounding.loss import WeaklyVGLossCompute
from detectron2.modeling.weaklygrounding.phrase_embedding_weakly_v1 import PhraseEmbeddingSent
from detectron2.modeling.weaklygrounding.phrase_embedding_weakly_v1 import PhraseEmbeddingPhr
from detectron2.layers.spatial_coordinate import meshgrid_generation
from detectron2.layers.numerical_stability_softmax import numerical_stability_masked_softmax
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
        self.phrase_embed = PhraseEmbeddingPhr(cfg, phrase_embed_dim=self.phrase_embed_dim, bidirectional=cfg.MODEL.VG.LSTM_BIDIRECTION)

        self.visual_embed_dim = self.phrase_embed_dim

        if cfg.MODEL.VG.SPATIAL_FEAT:
            self.spatial_trans = nn.Sequential(
                nn.Linear(2*cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION ** 2, 256),
                nn.ReLU(inplace=True),
            )
            self.obj_embed_dim = self.obj_embed_dim + 256

        self.visual_embedding = nn.Sequential(nn.Linear(self.obj_embed_dim, self.visual_embed_dim), nn.ReLU(inplace=True))
        self.similarity_fc = nn.Sequential(nn.Linear(self.phrase_embed_dim*2, 1))
        self.vis_batchnorm = nn.BatchNorm1d(num_features=1024)
        self.phr_batchnorm = nn.BatchNorm1d(num_features=1024)
        if cfg.SOLVER.DISC_IMG_SENT_LOSS:
            self.sent_mlp = nn.Linear(self.phrase_embed_dim, 512)
            self.visual_mlp = nn.Linear(self.phrase_embed_dim, 512)

        self.dec_phrase_gru = nn.GRU(input_size=self.visual_embed_dim, hidden_size=self.rnn_hidden_dim, num_layers=1, batch_first=True, dropout=0, bias=True)

        self.vis2phr = nn.Linear(self.rnn_hidden_dim, self.phrase_embed.phr_vocab_size)
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
        batch_ctx_embed = []


        """ Language Embedding"""

        batch_phrase_ids, batch_phrase_types, batch_phrase_embed, batch_phrase_len, \
        batch_phrase_dec_ids, batch_phrase_mask, batch_decoder_word_embed, batch_phrase_glove_embed = self.phrase_embed(all_sentences, all_phrase_ids, all_sent_sgs)

        batch_sent_embed = None
        h, w = features.shape[-2:]


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
            visual_features_bid = self.rcnn_top(self.det_roi_pooler(tuple([img_feat_bid]), [precomp_boxes_bid])).mean(dim=[2, 3]).contiguous()

            if cfg.MODEL.VG.SPATIAL_FEAT:
                spa_feat = meshgrid_generation(h, w)
                spa_feat = self.det_roi_pooler(tuple([spa_feat]), [precomp_boxes_bid]).view(visual_features_bid.shape[0], -1)
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
            pred_similarity_bid = self.similarity(visual_features_bid, phrase_embed_bid, obj_inds, phr_inds)
            pred_similarity_bid = pred_similarity_bid.reshape(num_phrase, num_box)


            if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
                det_label_embedding_bid = det_label_embedding[bid].to(self.device)
                sim = self.cal_det_label_sim_max(det_label_embedding_bid, batch_phrase_glove_embed[bid], precomp_score[bid])
                pred_similarity_bid = pred_similarity_bid * sim
                sim_mask = (sim > 0).float()
                atten_bid = numerical_stability_masked_softmax(pred_similarity_bid, sim_mask, dim=1)
            else:
                atten_bid = F.softmax(pred_similarity_bid, dim=1)


            ## reconstruction visual features
            visual_reconst_bid = torch.mm(atten_bid, visual_features_bid)
            decode_phr_logits = self.phrase_decoder(visual_reconst_bid, batch_decoder_word_embed[bid])
            batch_decode_logits.append(decode_phr_logits)

            if cfg.SOLVER.DISC_IMG_SENT_LOSS:
                batch_ctx_embed.append(self.visual_mlp(visual_reconst_bid.mean(0, keepdim=True)))
            batch_pred_similarity.append(atten_bid)
            max_box_ind = atten_bid.detach().cpu().numpy().argmax(1)
            batch_pred_boxes.append(precomp_boxes_bid[max_box_ind])


        if cfg.SOLVER.DISC_IMG_SENT_LOSS:
            batch_ctx_embed = torch.cat(batch_ctx_embed, dim=0)  ## N*512
            batch_sent_embed = torch.cat(batch_sent_embed, dim=0)
            batch_sent_embed = self.sent_mlp(batch_sent_embed)  ## N*512
            batch_ctx_sim = torch.mm(batch_ctx_embed, batch_sent_embed.permute(1, 0)) / 512 ** 0.5
        else:
            batch_ctx_sim = None

        noun_reconst_loss,disc_img_sent_loss = self.VGLoss(batch_phrase_mask, batch_decode_logits, batch_phrase_dec_ids, batch_ctx_sim)
        all_loss = dict(noun_reconst_loss=noun_reconst_loss, disc_img_sent_loss_s1=disc_img_sent_loss)

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
        enc_embed = torch.cat((visual_reconst.unsqueeze(1), phrase_enc_embed), dim=1) ## bs*maxl*L
        dec_emb, _ = self.dec_phrase_gru(enc_embed)
        decode_phr_logits = self.vis2phr(dec_emb)

        return decode_phr_logits


    def cal_det_label_sim_max(self, det_label_embed, decoder_word_embed, precomp_score):

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
        return pred_similarity

    def make_pair(self, phr_num, box_num):
        ind_phr, ind_box = np.meshgrid(range(phr_num), range(box_num), indexing='ij')
        ind_phr = ind_phr.reshape(-1)
        ind_box = ind_box.reshape(-1)
        return ind_phr, ind_box


def build_vg_head(cfg, det_roi_heads):
    return DetProposalVGHead(cfg, det_roi_heads)

