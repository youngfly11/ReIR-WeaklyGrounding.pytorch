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
from detectron2.layers.numerical_stability_softmax import numerical_stability_softmax


class PhraseEmbeddingPhr(torch.nn.Module):

    def __init__(self, cfg, phrase_embed_dim=1024, bidirectional=True):
        super(PhraseEmbeddingPhr, self).__init__()


        self.device = torch.device('cuda')
        self.bidirectional = bidirectional
        phr_vocab_file = open(cfg.MODEL.VG.VOCAB_PHR_FILE)
        self.phr_vocab = json.load(phr_vocab_file)
        phr_vocab_file.close()
        self.phr_vocab_to_id = {v:i+1 for i, v in enumerate(self.phr_vocab)}
        self.phr_vocab_size = len(self.phr_vocab) + 1

        self.embed_dim = phrase_embed_dim
        if self.bidirectional:
            self.hidden_dim = phrase_embed_dim // 2
        else:
            self.hidden_dim = phrase_embed_dim

        self.enc_embedding = nn.Embedding(num_embeddings=self.phr_vocab_size, embedding_dim=self.embed_dim, padding_idx=0, sparse=False)
        self.sent_rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, dropout=0, bidirectional=self.bidirectional, bias=True)

        if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
            with open(cfg.MODEL.VG.GLOVE_DICT_FILE, 'rb') as load_f:
                self.glove_embedding = pickle.load(load_f)  ## dict, which contain word embedding.


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

        for idx, sent in enumerate(all_sentences):

            seq = sent['sentence'].lower()
            phrases = sent['phrases']
            phrase_ids = []
            phrase_types = []
            input_phr = []
            lengths = []

            valid_phrases = self.filtering_phrase(phrases, all_phrase_ids[idx])
            tokenized_seq = seq.split(' ')

            # tokenized the phrase
            max_len = np.array([len(phr['phrase'].split(' ')) for phr in valid_phrases]).max()
            phrase_enc_ids = np.zeros((len(valid_phrases), max_len))
            phrase_dec_ids = np.zeros((len(valid_phrases), max_len+1)) ## to predict end token
            phrase_mask = np.zeros((len(valid_phrases), max_len+1)) ## to predict the "end" token

            phrase_glove_embedding = []

            for pid, phr in enumerate(valid_phrases):
                phrase_ids.append(phr['phrase_id'])
                phrase_types.append(phr['phrase_type'])
                tokenized_phr = phr['phrase'].lower().split(' ')

                word_glove_embedding = []
                for tid, w in enumerate(tokenized_phr):
                    phrase_enc_ids[pid, tid] = self.phr_vocab_to_id[w]
                    phrase_dec_ids[pid, tid] = self.phr_vocab_to_id[w]
                    phrase_mask[pid, tid] = 1

                    if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
                        phr_glo_vec = self.glove_embedding.get(w)
                        if phr_glo_vec is not None:
                            word_glove_embedding.append(phr_glo_vec)

                if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
                    if len(word_glove_embedding) == 0:
                        word_glove_embedding = 0 * torch.as_tensor(np.array(self.glove_embedding.get('a'))).float().unsqueeze(0).to(self.device)  ## 1*300
                    else:
                        word_glove_embedding = torch.as_tensor(np.array(word_glove_embedding), dtype=torch.float32).to(self.device)
                    phrase_glove_embedding.append(word_glove_embedding)

                phrase_mask[pid, tid+1] = 1
                phr_len = len(tokenized_phr)
                lengths.append(phr_len)
                input_phr.append(tokenized_phr)

            phrase_word_embeds_b = self.enc_embedding(torch.as_tensor(phrase_enc_ids).long().to(self.device))
            phrase_mask = torch.as_tensor(phrase_mask).float().to(self.device)


            if self.bidirectional:
                phrase_embeds = []
                for pid, phr in enumerate(input_phr):
                    phrase_embed_phr, last_embed = self.sent_rnn(phrase_word_embeds_b[[pid]][:, :len(phr)])
                    if cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Sum':
                        phrase_embeds.append(phrase_embed_phr.sum(1))  # average the embedding
                    elif cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Mean':
                        phrase_embeds.append(phrase_embed_phr.mean(1))
                phrase_embeds = torch.cat(phrase_embeds, dim=0) ## n*1024

            else:
                phrase_word_embeds, last_embed = self.sent_rnn(phrase_word_embeds_b)
                phrase_word_embeds = phrase_word_embeds * phrase_mask[:, 1:, None]

                if cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Sum':
                    phrase_embeds = phrase_word_embeds.sum(1)  # average the embedding
                elif cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Mean':
                    phrase_embeds = phrase_word_embeds.sum(1)/phrase_mask[:, 1:].sum(1, keepdim=True)
                else:
                    raise NotImplementedError

            phrase_decoder_word_embeds = phrase_word_embeds_b
            batch_phrase_ids.append(phrase_ids)
            batch_phrase_types.append(phrase_types)
            batch_phrase_embed.append(phrase_embeds)
            batch_phrase_len.append(lengths)
            batch_phrase_dec_ids.append(phrase_dec_ids)
            batch_phrase_mask.append(phrase_mask)
            batch_decoder_word_embed.append(phrase_decoder_word_embeds)
            batch_glove_phrase_embed.append(phrase_glove_embedding)

        return batch_phrase_ids, batch_phrase_types, batch_phrase_embed, batch_phrase_len, \
               batch_phrase_dec_ids, batch_phrase_mask, batch_decoder_word_embed, batch_glove_phrase_embed


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

        rel_vocab_file = open(cfg.MODEL.VG.VOCAB_REL_FILE)
        self.rel_vocab = json.load(rel_vocab_file)
        self.rel_vocab_to_id = {v: i + 1 for i, v in enumerate(self.rel_vocab)}
        self.rel_vocab_size = len(self.rel_vocab) + 1


        self.embed_dim = phrase_embed_dim

        if self.bidirectional:
            self.hidden_dim = phrase_embed_dim // 2
        else:
            self.hidden_dim = self.embed_dim

        self.sent_rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim, num_layers=1,
                                   batch_first=True, dropout=0, bidirectional=self.bidirectional, bias=True)


        self.enc_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim,
                                          padding_idx=0, sparse=False)


        if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
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
        batch_relation_conn = []
        batch_rel_phrase_embed = []
        batch_sent_embed = []
        batch_rel_dec_idx = []
        batch_input_rel_embed = []
        batch_rel_mask = []

        batch_cst_qid = []
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

            sent_qid = np.arange(0, len(valid_phrases))[None, :].repeat(2, 0)
            sent_qid[0, :]  = idx
            batch_cst_qid.append(sent_qid)
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
                    if cfg.MODEL.RELATION.REL_RECONST:
                        phrase_dec_ids[pid][wid] = self.vocab_to_id[word]
                    else:
                        phrase_dec_ids[pid][wid] = self.phr_vocab_to_id[word]

                    if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
                        phr_glo_vec = self.glove_embedding.get(word)
                        if phr_glo_vec is not None:
                            word_glove_embedding.append(phr_glo_vec)

                if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
                    if len(word_glove_embedding) == 0:
                        word_glove_embedding = 0 * torch.as_tensor(np.array(self.glove_embedding.get('a'))).float().unsqueeze(0).to(self.device)  ## 1*300
                    else:
                        # word_glove_embedding = torch.as_tensor(np.array(word_glove_embedding)).float().mean(0, keepdim=True)
                        word_glove_embedding = torch.as_tensor(np.array(word_glove_embedding)).float().to(self.device) ##L*300
                    phrase_glove_embedding.append(word_glove_embedding)

                phrase_mask[pid][:phr_len+1] = 1
                phrase_decoder_word_embeds[pid, :phr_len, :] += seq_embed_b[0][start_ind:start_ind+phr_len]

                if cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Sum':
                    phrase_embeds.append(seq_embed[:, start_ind:start_ind+phr_len].sum(1))  # average the embedding
                elif cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Mean':
                    phrase_embeds.append(seq_embed[:, start_ind:start_ind+phr_len].mean(1))
                elif cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'SimV1':
                    peb = seq_embed[:, start_ind:start_ind+phr_len].squeeze(0) ## 5*1024
                    peb_last = peb[[-1]]  ## 1*1024
                    sim = F.softmax(F.cosine_similarity(peb_last, peb, dim=1).unsqueeze(0) * 10, dim=1)
                    phrase_embeds.append(torch.mm(sim, peb))
                elif cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'SimV2':
                    peb = seq_embed[:, start_ind:start_ind + phr_len].squeeze(0)  ## 1*5*1024
                    if peb.shape[0] > 2:
                        peb_last = peb[-2:].mean(0, keepdim=True)
                    else:
                        peb_last = peb[[-1]]  ## 1*1024
                    sim = F.softmax(F.cosine_similarity(peb_last, peb, dim=1).unsqueeze(0) * 10, dim=1)
                    phrase_embeds.append(torch.mm(sim, peb))

            phrase_embeds = torch.cat(phrase_embeds, dim=0)
            phrase_mask = torch.as_tensor(phrase_mask).float().to(self.device)

            # batch_sent_embed.append(torch.cat((seq_embed[:,0], seq_embed[:, -1]), dim=1))
            batch_sent_embed.append(seq_embed.mean(1))
            batch_phrase_ids.append(phrase_ids)
            batch_phrase_types.append(phrase_types)
            batch_phrase_embed.append(phrase_embeds)
            batch_phrase_len.append(lengths)
            batch_phrase_dec_ids.append(phrase_dec_ids)
            batch_phrase_mask.append(phrase_mask)
            batch_decoder_word_embed.append(phrase_decoder_word_embeds)
            batch_glove_phrase_embed.append(phrase_glove_embedding)


            if cfg.MODEL.RELATION.IS_ON:

                "rel phrase embedding"

                sent_sg = all_sent_sgs[idx]
                relation_conn = []
                if len(sent_sg)>0:
                    rel_lens = np.array([len(pid2phr[rel[0]]) + len(rel[2].lower().split(' ')) + len(pid2phr[rel[1]]) for rel in sent_sg])
                    max_rel_len = rel_lens.max()
                input_rel_phr_idx = []
                input_rel_phr_mask = []

                for rel_id, rel in enumerate(sent_sg):
                    sbj_id, obj_id, rel_phrase = rel

                    if sbj_id not in phrase_ids or obj_id not in phrase_ids:
                        continue
                    relation_conn.append([phrase_ids.index(sbj_id), phrase_ids.index(obj_id), rel_id])

                    uni_rel_phr_idx = torch.zeros(max_rel_len+1)
                    tokenized_rel = pid2phr[sbj_id] + rel_phrase.lower().split(' ') + pid2phr[obj_id]
                    uni_rel_mask = torch.zeros(max_rel_len+1)
                    uni_rel_mask[:len(tokenized_rel)+1] = 1
                    for wid, w in enumerate(tokenized_rel):
                        uni_rel_phr_idx[wid] = self.vocab_to_id[w]

                    input_rel_phr_idx.append(uni_rel_phr_idx)
                    input_rel_phr_mask.append(uni_rel_mask)

                if len(relation_conn) > 0:
                    input_rel_phr_idx = torch.stack(input_rel_phr_idx).long().to(self.device)
                    rel_phrase_embed = self.enc_embedding(input_rel_phr_idx[:, :-1])
                    rel_phrase_embeds, _ = self.sent_rnn(rel_phrase_embed)
                    rel_phrase_embeds = select_embedding(rel_phrase_embeds, rel_lens.tolist(), cfg.MODEL.VG.PHRASE_SELECT_TYPE)
                    batch_rel_phrase_embed.append(rel_phrase_embeds)
                    batch_input_rel_embed.append(rel_phrase_embed)  ## M*1024
                    batch_rel_mask.append(torch.stack(input_rel_phr_mask, dim=0).to(self.device)) ## M*(L+1)
                    batch_rel_dec_idx.append(input_rel_phr_idx) ## M*(L+1)

                else:
                    batch_rel_phrase_embed.append(None)
                    batch_rel_mask.append(None)
                    batch_input_rel_embed.append(None)
                    batch_rel_dec_idx.append(None)

                batch_relation_conn.append(relation_conn)

            else:
                batch_rel_mask.append(None)
                batch_rel_dec_idx.append(None)
                batch_input_rel_embed.append(None)
                batch_rel_phrase_embed.append(None)

        return batch_phrase_ids, batch_phrase_types, batch_phrase_embed, batch_phrase_len, \
               batch_phrase_dec_ids, batch_phrase_mask, batch_decoder_word_embed, batch_glove_phrase_embed, batch_rel_phrase_embed, batch_relation_conn, batch_sent_embed, batch_input_rel_embed, \
               batch_rel_mask, batch_rel_dec_idx


class PhraseEmbeddingSentCate(torch.nn.Module):

    def __init__(self, cfg, phrase_embed_dim=1024, bidirectional=True):
        super(PhraseEmbeddingSentCate, self).__init__()

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

        rel_vocab_file = open(cfg.MODEL.VG.VOCAB_REL_FILE)
        self.rel_vocab = json.load(rel_vocab_file)
        self.rel_vocab_to_id = {v: i + 1 for i, v in enumerate(self.rel_vocab)}
        self.rel_vocab_size = len(self.rel_vocab) + 1


        self.embed_dim = phrase_embed_dim

        if self.bidirectional:
            self.hidden_dim = phrase_embed_dim // 2
        else:
            self.hidden_dim = self.embed_dim

        self.sent_rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim, num_layers=1,
                                   batch_first=True, dropout=0, bidirectional=self.bidirectional, bias=True)


        self.enc_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim,
                                          padding_idx=0, sparse=False)

        rel_cates_dict = json.load(open(cfg.MODEL.RELATION.REL_CATE_PATH, 'r'))['rel cates']
        self.rel_cater_size = len(rel_cates_dict) - 1

        if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
            with open(cfg.MODEL.VG.GLOVE_DICT_FILE, 'rb') as load_f:
                self.glove_embedding = pickle.load(load_f)  ## dict, which contain word embedding.


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
        batch_relation_conn = []
        batch_rel_phrase_embed = []
        batch_sent_embed = []
        batch_rel_dec_idx = []
        batch_input_rel_embed = []
        batch_rel_mask = []

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
                    # if cfg.MODEL.RELATION.REL_RECONST:
                    # phrase_dec_ids[pid][wid] = self.vocab_to_id[word]
                    # else:
                    phrase_dec_ids[pid][wid] = self.phr_vocab_to_id[word]

                    if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
                        phr_glo_vec = self.glove_embedding.get(word)
                        if phr_glo_vec is not None:
                            word_glove_embedding.append(phr_glo_vec)

                if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
                    if len(word_glove_embedding) == 0:
                        word_glove_embedding = 0 * torch.as_tensor(np.array(self.glove_embedding.get('a'))).float().unsqueeze(0).to(self.device)  ## 1*300
                    else:
                        word_glove_embedding = torch.as_tensor(np.array(word_glove_embedding)).float().to(self.device) ##L*300
                    phrase_glove_embedding.append(word_glove_embedding)

                phrase_mask[pid][:phr_len+1] = 1
                phrase_decoder_word_embeds[pid, :phr_len, :] += seq_embed_b[0][start_ind:start_ind+phr_len]

                if cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Sum':
                    phrase_embeds.append(seq_embed[:, start_ind:start_ind+phr_len].sum(1))  # sum the embedding
                elif cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Mean':
                    phrase_embeds.append(seq_embed[:, start_ind:start_ind+phr_len].mean(1)) ## average the embedding
                else:
                    raise NotImplementedError

            phrase_embeds = torch.cat(phrase_embeds, dim=0)
            phrase_mask = torch.as_tensor(phrase_mask).float().to(self.device)

            # batch_sent_embed.append(torch.cat((seq_embed[:,0], seq_embed[:, -1]), dim=1))
            batch_sent_embed.append(seq_embed.mean(1))
            batch_phrase_ids.append(phrase_ids)
            batch_phrase_types.append(phrase_types)
            batch_phrase_embed.append(phrase_embeds)
            batch_phrase_len.append(lengths)
            batch_phrase_dec_ids.append(phrase_dec_ids)
            batch_phrase_mask.append(phrase_mask)
            batch_decoder_word_embed.append(phrase_decoder_word_embeds)
            batch_glove_phrase_embed.append(phrase_glove_embedding)


            if cfg.MODEL.RELATION.IS_ON:

                "rel phrase embedding"

                sent_sg = all_sent_sgs[idx]
                relation_conn = []
                if len(sent_sg)>0:
                    rel_lens = np.array([len(pid2phr[rel[0]]) + len(rel[3].lower().split(' ')) + len(pid2phr[rel[1]]) for rel in sent_sg])
                    max_rel_len = rel_lens.max()
                input_rel_phr_idx = []
                input_rel_phr_mask = []

                for rel_id, rel in enumerate(sent_sg):
                    sbj_id, obj_id, rel_cate, rel_phrase = rel
                    rel_cate = [i - 1 for i in rel_cate if i > 0]
                    if sbj_id not in phrase_ids or obj_id not in phrase_ids:
                        continue
                    relation_conn.append([phrase_ids.index(sbj_id), phrase_ids.index(obj_id), rel_cate, rel_id])

                    uni_rel_phr_idx = torch.zeros(max_rel_len+1)
                    tokenized_rel = rel_phrase.lower().split(' ')
                    uni_rel_mask = torch.zeros(max_rel_len+1)
                    uni_rel_mask[:len(tokenized_rel)+1] = 1
                    for wid, w in enumerate(tokenized_rel):
                        uni_rel_phr_idx[wid] = self.rel_vocab_to_id[w]

                    input_rel_phr_idx.append(uni_rel_phr_idx)
                    input_rel_phr_mask.append(uni_rel_mask)

                if len(relation_conn) > 0:
                    input_rel_phr_idx = torch.stack(input_rel_phr_idx).long().to(self.device)
                    rel_phrase_embed = self.enc_embedding(input_rel_phr_idx[:, :-1])
                    # rel_phrase_embeds, _ = self.sent_rnn(rel_phrase_embed)
                    # rel_phrase_embeds = select_embedding(rel_phrase_embeds, rel_lens.tolist(), cfg.MODEL.VG.PHRASE_SELECT_TYPE)
                    # batch_rel_phrase_embed.append(rel_phrase_embeds)
                    batch_input_rel_embed.append(rel_phrase_embed)  ## M*1024
                    batch_rel_mask.append(torch.stack(input_rel_phr_mask, dim=0).to(self.device)) ## M*(L+1)
                    batch_rel_dec_idx.append(input_rel_phr_idx) ## M*(L+1)

                else:
                    # batch_rel_phrase_embed.append(None)
                    batch_rel_mask.append(None)
                    batch_input_rel_embed.append(None)
                    batch_rel_dec_idx.append(None)

                batch_relation_conn.append(relation_conn)

            else:
                batch_rel_mask.append(None)
                batch_rel_dec_idx.append(None)
                batch_input_rel_embed.append(None)
                # batch_rel_phrase_embed.append(None)

        return batch_phrase_ids, batch_phrase_types, batch_phrase_embed, batch_phrase_len, \
               batch_phrase_dec_ids, batch_phrase_mask, batch_decoder_word_embed, batch_glove_phrase_embed, batch_relation_conn, batch_sent_embed, batch_input_rel_embed, \
               batch_rel_mask, batch_rel_dec_idx


class PhraseEmbeddingSentSemCate(torch.nn.Module):

    def __init__(self, cfg, phrase_embed_dim=1024, bidirectional=True):
        super(PhraseEmbeddingSentSemCate, self).__init__()

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

        rel_vocab_file = open(cfg.MODEL.VG.VOCAB_REL_FILE)
        self.rel_vocab = json.load(rel_vocab_file)
        self.rel_vocab_to_id = {v: i + 1 for i, v in enumerate(self.rel_vocab)}
        self.rel_vocab_size = len(self.rel_vocab) + 1


        self.embed_dim = phrase_embed_dim

        if self.bidirectional:
            self.hidden_dim = phrase_embed_dim // 2
        else:
            self.hidden_dim = self.embed_dim

        self.sent_rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim, num_layers=1,
                                   batch_first=True, dropout=0, bidirectional=self.bidirectional, bias=True)

        self.sem_nouns_cates = json.load(open(cfg.MODEL.VG.SEMANTIC_NOUNS_PATH, 'r'))['cates']
        self.sem_attr_cates = json.load(open(cfg.MODEL.VG.SEMANTIC_ATTR_PATH, 'r'))['cates']

        self.enc_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim,
                                          padding_idx=0, sparse=False)

        rel_cates_dict = json.load(open(cfg.MODEL.RELATION.REL_CATE_PATH, 'r'))['rel cates']
        self.rel_cater_size = len(rel_cates_dict) - 1

        if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
            with open(cfg.MODEL.VG.GLOVE_DICT_FILE, 'rb') as load_f:
                self.glove_embedding = pickle.load(load_f)  ## dict, which contain word embedding.


    @staticmethod
    def filtering_phrase(phrases, all_phrase):
        phrase_valid = []
        for phr in phrases:
            if phr['phrase_id'] in all_phrase:
                phrase_valid.append(phr)
        return phrase_valid

    def get_sem_nouns(self, cates):
        sem_nouns_label = torch.zeros(cfg.MODEL.VG.SEMANTIC_NOUNS_TOPK)
        for cats in cates:
            cates_ids = self.sem_nouns_cates[cats]
            if cates_ids < cfg.MODEL.VG.SEMANTIC_NOUNS_TOPK:
                sem_nouns_label[cates_ids] = 1
        return sem_nouns_label

    def get_attr_nouns(self, cates):

        sem_attr_label = torch.zeros(cfg.MODEL.VG.SEMANTIC_ATTR_TOPK)
        for cats in cates:
            cates_ids = self.sem_attr_cates[cats]
            if cates_ids < cfg.MODEL.VG.SEMANTIC_ATTR_TOPK:
                sem_attr_label[cates_ids] = 1
        return sem_attr_label


    def forward(self, all_sentences, all_phrase_ids, all_sent_sgs):

        batch_phrase_ids = []
        batch_phrase_types = []
        batch_phrase_embed = []
        batch_phrase_len = []
        batch_phrase_dec_ids = []
        batch_phrase_mask = []
        batch_decoder_word_embed = []
        batch_glove_phrase_embed = []
        batch_relation_conn = []
        batch_sent_embed = []
        batch_rel_dec_idx = []
        batch_input_rel_embed = []
        batch_rel_mask = []
        batch_semantic_nouns = []
        batch_semantic_attrs = []
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
            sent_sem_nouns = []
            sent_sem_attrs = []
            for pid, phr in enumerate(valid_phrases):
                phrase_ids.append(phr['phrase_id'])
                phrase_types.append(phr['phrase_type'])
                tokenized_phr = phr['phrase'].lower().split(' ')
                pid2phr[phr['phrase_id']] = tokenized_phr
                phr_len = len(tokenized_phr)
                start_ind = phr['first_word_index']

                sent_sem_nouns.append(self.get_sem_nouns(phr['cates']))
                sent_sem_attrs.append(self.get_attr_nouns(phr['clean_att']))

                word_glove_embedding = []
                for wid, word in enumerate(tokenized_phr):
                    phrase_dec_ids[pid][wid] = self.phr_vocab_to_id[word]

                    if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
                        phr_glo_vec = self.glove_embedding.get(word)
                        if phr_glo_vec is not None:
                            word_glove_embedding.append(phr_glo_vec)

                if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
                    if len(word_glove_embedding) == 0:
                        word_glove_embedding = 0 * torch.as_tensor(np.array(self.glove_embedding.get('a'))).float().unsqueeze(0).to(self.device)  ## 1*300
                    else:
                        word_glove_embedding = torch.as_tensor(np.array(word_glove_embedding)).float().to(self.device) ##L*300
                    phrase_glove_embedding.append(word_glove_embedding)

                phrase_mask[pid][:phr_len+1] = 1
                phrase_decoder_word_embeds[pid, :phr_len, :] += seq_embed_b[0][start_ind:start_ind+phr_len]

                if cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Sum':
                    phrase_embeds.append(seq_embed[:, start_ind:start_ind+phr_len].sum(1))  # sum the embedding
                elif cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Mean':
                    phrase_embeds.append(seq_embed[:, start_ind:start_ind+phr_len].mean(1)) ## average the embedding
                else:
                    raise NotImplementedError

            sent_sem_nouns = torch.stack(sent_sem_nouns, dim=0).to(self.device)
            batch_semantic_nouns.append(sent_sem_nouns)

            sent_sem_attrs = torch.stack(sent_sem_attrs, dim=0).to(self.device)
            batch_semantic_attrs.append(sent_sem_attrs)

            phrase_embeds = torch.cat(phrase_embeds, dim=0)
            phrase_mask = torch.as_tensor(phrase_mask).float().to(self.device)

            # batch_sent_embed.append(torch.cat((seq_embed[:,0], seq_embed[:, -1]), dim=1))
            batch_sent_embed.append(seq_embed.mean(1))
            batch_phrase_ids.append(phrase_ids)
            batch_phrase_types.append(phrase_types)
            batch_phrase_embed.append(phrase_embeds)
            batch_phrase_len.append(lengths)
            batch_phrase_dec_ids.append(phrase_dec_ids)
            batch_phrase_mask.append(phrase_mask)
            batch_decoder_word_embed.append(phrase_decoder_word_embeds)
            batch_glove_phrase_embed.append(phrase_glove_embedding)


            if cfg.MODEL.RELATION.IS_ON:

                "rel phrase embedding"

                sent_sg = all_sent_sgs[idx]
                relation_conn = []
                if len(sent_sg)>0:
                    rel_lens = np.array([len(pid2phr[rel[0]]) + len(rel[3].lower().split(' ')) + len(pid2phr[rel[1]]) for rel in sent_sg])
                    max_rel_len = rel_lens.max()
                input_rel_phr_idx = []
                input_rel_phr_mask = []

                for rel_id, rel in enumerate(sent_sg):
                    sbj_id, obj_id, rel_cate, rel_phrase = rel
                    rel_cate = [i - 1 for i in rel_cate if i > 0]
                    if sbj_id not in phrase_ids or obj_id not in phrase_ids:
                        continue
                    relation_conn.append([phrase_ids.index(sbj_id), phrase_ids.index(obj_id), rel_cate, rel_id])

                    uni_rel_phr_idx = torch.zeros(max_rel_len+1)
                    tokenized_rel = rel_phrase.lower().split(' ')
                    uni_rel_mask = torch.zeros(max_rel_len+1)
                    uni_rel_mask[:len(tokenized_rel)+1] = 1
                    for wid, w in enumerate(tokenized_rel):
                        uni_rel_phr_idx[wid] = self.rel_vocab_to_id[w]

                    input_rel_phr_idx.append(uni_rel_phr_idx)
                    input_rel_phr_mask.append(uni_rel_mask)

                if len(relation_conn) > 0:
                    input_rel_phr_idx = torch.stack(input_rel_phr_idx).long().to(self.device)
                    rel_phrase_embed = self.enc_embedding(input_rel_phr_idx[:, :-1])
                    batch_input_rel_embed.append(rel_phrase_embed)  ## M*1024
                    batch_rel_mask.append(torch.stack(input_rel_phr_mask, dim=0).to(self.device)) ## M*(L+1)
                    batch_rel_dec_idx.append(input_rel_phr_idx) ## M*(L+1)

                else:
                    batch_rel_mask.append(None)
                    batch_input_rel_embed.append(None)
                    batch_rel_dec_idx.append(None)

                batch_relation_conn.append(relation_conn)

            else:
                batch_rel_mask.append(None)
                batch_rel_dec_idx.append(None)
                batch_input_rel_embed.append(None)


        return batch_phrase_ids, batch_phrase_types, batch_phrase_embed, batch_phrase_len, \
               batch_phrase_dec_ids, batch_phrase_mask, batch_decoder_word_embed, batch_glove_phrase_embed, batch_relation_conn, batch_sent_embed, batch_input_rel_embed, \
               batch_rel_mask, batch_rel_dec_idx, batch_semantic_nouns, batch_semantic_attrs


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





















