#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/04/25 16:13

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import global_cfg as cfg
import json
import numpy as np
import pickle
from allennlp.modules.elmo import Elmo, batch_to_ids
from detectron2.layers.numerical_stability_softmax import numerical_stability_softmax


class PhraseEmbeddingSent(torch.nn.Module):

    def __init__(self, cfg, phrase_embed_dim=1024, bidirectional=True):
        super(PhraseEmbeddingSent, self).__init__()

        self.device = torch.device('cuda')
        self.bidirectional = bidirectional

        vocab_file = open(cfg.MODEL.VG.VOCAB_FILE)
        self.vocab = json.load(vocab_file)
        vocab_file.close()
        add_vocab = ['relate', 'butted']
        self.vocab.extend(add_vocab)
        self.vocab_to_id = {v: i + 1 for i, v in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab) + 1

        phr_vocab_file = open(cfg.MODEL.VG.VOCAB_PHR_FILE)
        self.phr_vocab = json.load(phr_vocab_file)
        self.phr_vocab_to_id = {v:i+1 for i, v in enumerate(self.phr_vocab)}
        self.phr_vocab_size = len(self.phr_vocab) + 1

        self.embed_dim = phrase_embed_dim
        if self.bidirectional:
            self.hidden_dim = phrase_embed_dim // 2
        else:
            self.hidden_dim = self.embed_dim

        self.sent_rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim, num_layers=1,
                               batch_first=True, dropout=0, bidirectional=self.bidirectional, bias=True)

        self.enc_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim, padding_idx=0, sparse=False)

        with open(cfg.MODEL.VG.GLOVE_DICT_FILE, 'rb') as load_f:
            self.glove_embedding = pickle.load(load_f)  ## dict, which contain word embedding.

        if cfg.SOLVER.INIT_PARA:
            self.init_para()

    def init_para(self, ):

        # Initialize LSTM Weights and Biases
        for layer in self.sent_rnn._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.sent_rnn, param_name)
                    nn.init.xavier_normal_(weight.data)
                else:
                    bias = getattr(self.sent_rnn, param_name)
                    # bias.data.zero_()
                    nn.init.uniform_(bias.data, a=-0.01, b=0.01)
        nn.init.uniform_(self.enc_embedding.weight.data, a=-0.01, b=0.01)


    @staticmethod
    def filtering_phrase(phrases, all_phrase):
        phrase_valid = []
        for phr in phrases:
            if phr['phrase_id'] in all_phrase:
                phrase_valid.append(phr)
        return phrase_valid

    def forward(self, all_sentences, all_phrase_ids, all_sent_sgs):

        batch_phrase_ids = []
        batch_phrase_types = []
        batch_phrase_embed = []
        batch_phrase_len = []
        batch_phrase_dec_ids = []
        batch_phrase_mask = []
        batch_decoder_word_embed = []
        batch_glove_phrase_embed = []
        batch_sent_embed = []
        batch_max_len = []

        for idx, sent in enumerate(all_sentences):

            seq = sent['sentence'].lower()
            phrases = sent['phrases']
            phrase_ids = []
            phrase_types = []
            lengths = []
            pid2phr = {}

            valid_phrases = self.filtering_phrase(phrases, all_phrase_ids[idx])
            tokenized_seq = seq.split(' ')
            seq_enc_ids = [[self.vocab_to_id[w] for w in tokenized_seq]]

            seq_embed_b = self.enc_embedding(torch.as_tensor(seq_enc_ids).long().to(self.device)) # 1*L*1024
            seq_embed, hn = self.sent_rnn(seq_embed_b)

            # tokenized the phrase
            max_len = np.array([len(phr['phrase'].split(' ')) for phr in valid_phrases]).max()
            phrase_dec_ids = np.zeros((len(valid_phrases), max_len+1)) ## to predict end token
            phrase_mask = np.zeros((len(valid_phrases), max_len+1)) ## to predict the "end" token
            batch_max_len.append(max_len)

            phrase_decoder_word_embeds = torch.zeros(len(valid_phrases), max_len, seq_embed.shape[-1]).to(self.device)  ##
            phrase_embeds = []

            phrase_glove_embedding = []
            for pid, phr in enumerate(valid_phrases):
                phrase_ids.append(phr['phrase_id'])
                phrase_types.append(phr['phrase_type'])
                tokenized_phr = phr['phrase'].lower().split(' ')
                pid2phr[phr['phrase_id']] = tokenized_phr
                phr_len = len(tokenized_phr)
                start_ind = phr['first_word_index']

                word_glove_embedding = []
                for wid, word in enumerate(tokenized_phr):

                    phrase_dec_ids[pid][wid] = self.phr_vocab_to_id[word]
                    phr_glo_vec = self.glove_embedding.get(word)
                    if phr_glo_vec is not None:
                        word_glove_embedding.append(phr_glo_vec)

                if len(word_glove_embedding) == 0:
                    word_glove_embedding = 0 * torch.as_tensor(self.glove_embedding.get('a')).float().unsqueeze(0).to(self.device)  ## 1*300
                else:
                    word_glove_embedding = torch.as_tensor(np.array(word_glove_embedding)).float().to(self.device) ##L*300
                phrase_glove_embedding.append(word_glove_embedding)

                phrase_mask[pid][:phr_len+1] = 1
                phrase_decoder_word_embeds[pid, :phr_len, :] += seq_embed_b[0][start_ind:start_ind+phr_len]

                if cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Sum':
                    phrase_embeds.append(seq_embed[:, start_ind:start_ind+phr_len].sum(1))  # average the embedding
                elif cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Mean':
                    phrase_embeds.append(seq_embed[:, start_ind:start_ind+phr_len].mean(1))


            phrase_embeds = torch.cat(phrase_embeds, dim=0)
            phrase_mask = torch.as_tensor(phrase_mask).float().to(self.device)
            batch_sent_embed.append(seq_embed.mean(1))
            batch_phrase_ids.append(phrase_ids)
            batch_phrase_types.append(phrase_types)
            batch_phrase_embed.append(phrase_embeds)
            batch_phrase_len.append(lengths)
            batch_phrase_dec_ids.append(phrase_dec_ids)
            batch_phrase_mask.append(phrase_mask)
            batch_decoder_word_embed.append(phrase_decoder_word_embeds)
            batch_glove_phrase_embed.append(phrase_glove_embedding)

        return batch_phrase_ids, batch_phrase_types, batch_phrase_embed, batch_phrase_len, batch_phrase_dec_ids, batch_phrase_mask, batch_decoder_word_embed, batch_glove_phrase_embed, batch_sent_embed



def select_embedding(x, lengths, select_type=None):
    batch_size = x.size(0)
    mask = x.data.new().resize_as_(x.data).fill_(0)
    for i in range(batch_size):
        if select_type == 'Mean':
            mask[i][:lengths[i]].fill_(1/lengths[i])
        elif select_type == 'Sum':
            mask[i][:lengths[i]].fill_(1)
        else:
            raise NotImplementedError

    x = x.mul(mask)
    x = x.sum(1).view(batch_size, -1)
    return x




def construct_contrastive_reconst(batch_glove_phrase_embedding, batch_phrase_decoder_word_embed, batch_phrase_dec_ids, batch_phrase_mask, batch_max_len, batch_cst_qid, num_cst):

    """

    :param batch_glove_phrase_embedding:
    :param batch_phrase_decoder_word_embed:
    :param batch_phrase_dec_ids:
    :param batch_phrase_mask:
    :param batch_max_len: [5,4,2,2,1,2,3,5] to save max_len in every sentence
    :param batch_phr_qid: 2*L, the first dimension is sent_id and the second dimension is
    :return:
    """



    ## select the last top5 phrase as contrastive reconst
    batch_glove_phrase_embedding = torch.cat(batch_glove_phrase_embedding, dim=0) ## L*300
    batch_glo_sim = F.cosine_similarity(batch_glove_phrase_embedding.unsqueeze(1), batch_glove_phrase_embedding.unsqueeze(0), dim=2) ## L*L
    batch_rtop_loc  = torch.topk(-batch_glo_sim, k=num_cst, dim=1)[1].detach().cpu().numpy()


    cum = 0

    batch_contrastive_word_embedding = []
    batch_contrastive_phrase_mask = []
    batch_contrastive_dec_ids = []

    for sent_id, all_phrase_word_embed in enumerate(batch_phrase_decoder_word_embed):  ## B*M*1024, phrase_dec, B*M*C

        num_phrase, num_word, num_dim = all_phrase_word_embed.shape

        max_len_sent = batch_max_len[batch_cst_qid[:, batch_rtop_loc[cum:cum+num_phrase, :].reshape(-1)][0]].max()
        contrastive_phrase_word_embed = torch.zeros(num_phrase*num_cst, max_len_sent, num_dim).to(torch.device('cuda'))
        contrastive_phrase_dec_ids = np.zeros((num_phrase*num_cst, max_len_sent))
        contrastive_phrase_mask = torch.zeros(num_phrase*num_cst, max_len_sent).to(torch.device('cuda'))


        for phr_id in range(num_phrase):

            phr_rtop_idx = batch_cst_qid[:, batch_rtop_loc[cum+phr_id]].transpose(1, 0)

            for rid, phr_rtop in enumerate(phr_rtop_idx):
                phr_rtop = phr_rtop.tolist()
                select_word_embed = batch_phrase_decoder_word_embed[phr_rtop[0]][phr_rtop[1]]
                contrastive_phrase_word_embed[phr_id*num_cst+rid, :select_word_embed.shape[0], :] += select_word_embed
                contrastive_phrase_mask[phr_id*num_cst+rid, :select_word_embed.shape[0]] += batch_phrase_mask[phr_rtop[0]][phr_rtop[1]][1:] ## ignore the endtoken in the contrastive reconstruction
                contrastive_phrase_dec_ids[phr_id*num_cst+rid, :select_word_embed.shape[0]] += batch_phrase_dec_ids[phr_rtop[0]][phr_rtop[1]][:-1] ## drop the last endtoken

        cum += num_phrase

        batch_contrastive_word_embedding.append(contrastive_phrase_word_embed)
        batch_contrastive_dec_ids.append(contrastive_phrase_dec_ids)
        batch_contrastive_phrase_mask.append(contrastive_phrase_mask)

    return batch_contrastive_word_embedding, batch_contrastive_phrase_mask, batch_contrastive_dec_ids





















