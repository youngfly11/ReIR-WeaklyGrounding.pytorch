import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
from detectron2.config import global_cfg as cfg
import json
import numpy as np
import pickle


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


        self.enc_embedding = nn.Embedding(num_embeddings=self.phr_vocab_size,
                                              embedding_dim=self.embed_dim,
                                              padding_idx=0, sparse=False)

        self.sent_rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim, num_layers=1,
                               batch_first=True, dropout=0, bidirectional=self.bidirectional, bias=True)

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
                    bias.data.zero_()
                    # nn.init.uniform_(bias.data, a=-0.01, b=0.01)

        if not cfg.MODEL.VG.USING_ELMO:
            nn.init.xavier_normal_(self.enc_embedding.weight.data)


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

            # phrase_mask_last = np.zeros((len(valid_phrases), max_len))

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
                        word_glove_embedding = 0 * torch.as_tensor(self.glove_embedding.get('a')).float().unsqueeze(0) ## 1*300
                    else:
                        word_glove_embedding = torch.as_tensor(np.array(word_glove_embedding)).float().mean(0, keepdim=True)
                    phrase_glove_embedding.append(word_glove_embedding.to(self.device))

                phrase_mask[pid, tid+1] = 1
                # phrase_dec_ids[:, :-1] = phrase_enc_ids[:, 1:]
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
                phrase_embeds = None
                if cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Sum':
                    phrase_embeds = phrase_word_embeds.sum(1)  # average the embedding
                elif cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Mean':
                    phrase_embeds = phrase_word_embeds.sum(1)/phrase_mask[:, 1:].sum(1).unsqueeze(1)

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


        self.embed_dim = phrase_embed_dim

        if self.bidirectional:
            self.hidden_dim = phrase_embed_dim // 2
        else:
            self.hidden_dim = self.embed_dim

        if cfg.MODEL.VG.USING_ELMO:
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self.elmo = Elmo(options_file, weight_file, 2, dropout=0, requires_grad=False)
            self.elmo.eval()
        else:
            self.enc_embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                              embedding_dim=self.embed_dim,
                                              padding_idx=0, sparse=False)

        self.sent_rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim, num_layers=1,
                               batch_first=True, dropout=0, bidirectional=self.bidirectional, bias=True)

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

        for idx, sent in enumerate(all_sentences):

            seq = sent['sentence'].lower()
            phrases = sent['phrases']
            phrase_ids = []
            phrase_types = []
            input_phr = []
            lengths = []

            valid_phrases = self.filtering_phrase(phrases, all_phrase_ids[idx])
            tokenized_seq = seq.split(' ')
            seq_enc_ids = [[self.vocab_to_id[w] for w in tokenized_seq]]

            """ Extract the word embedding and feed into sent_rnn"""
            if cfg.MODEL.VG.USING_ELMO:
                input_seq_idx = batch_to_ids([tokenized_seq]).to(self.device)
                seq_embed_b = self.elmo(input_seq_idx)['elmo_representations'][1] ## 1*L*1024
                seq_embed, hn = self.sent_rnn(seq_embed_b)
            else:

                seq_embed_b = self.enc_embedding(torch.as_tensor(seq_enc_ids).long().to(self.device)) # 1*L*1024
                seq_embed, hn = self.sent_rnn(seq_embed_b)

            # tokenized the phrase
            max_len = np.array([len(phr['phrase'].split(' ')) for phr in valid_phrases]).max()
            phrase_dec_ids = np.zeros((len(valid_phrases), max_len+1)) ## to predict end token
            phrase_mask = np.zeros((len(valid_phrases), max_len+1)) ## to predict the "end" token


            phrase_decoder_word_embeds = torch.zeros(len(valid_phrases), max_len, seq_embed.shape[-1]).to(self.device)  ##
            phrase_embeds = []

            phrase_glove_embedding = []
            for pid, phr in enumerate(valid_phrases):
                phrase_ids.append(phr['phrase_id'])
                phrase_types.append(phr['phrase_type'])
                tokenized_phr = phr['phrase'].lower().split(' ')
                phr_len = len(tokenized_phr)
                start_ind = phr['first_word_index']

                word_glove_embedding = []
                for wid, word in enumerate(tokenized_phr):
                    phrase_dec_ids[pid][wid] = self.phr_vocab_to_id[word]

                    if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
                        phr_glo_vec = self.glove_embedding.get(word)
                        if phr_glo_vec is not None:
                            word_glove_embedding.append(phr_glo_vec)

                if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
                    if len(word_glove_embedding) == 0:
                        word_glove_embedding = 0 * torch.as_tensor(self.glove_embedding.get('a')).float().unsqueeze(0)  ## 1*300
                    else:
                        word_glove_embedding = torch.as_tensor(np.array(word_glove_embedding)).float().mean(0, keepdim=True)
                    phrase_glove_embedding.append(word_glove_embedding)

                phrase_mask[pid][:phr_len+1] = 1
                phrase_decoder_word_embeds[pid, :phr_len, :] =  phrase_decoder_word_embeds[pid, :phr_len, :] + seq_embed_b[0][start_ind:start_ind+phr_len]

                if cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Sum':
                    phrase_embeds.append(seq_embed[[0], start_ind:start_ind+phr_len].sum(1))  # average the embedding
                elif cfg.MODEL.VG.PHRASE_SELECT_TYPE == 'Mean':
                    phrase_embeds.append(seq_embed[[0], start_ind:start_ind+phr_len].mean(1))

            phrase_embeds = torch.cat(phrase_embeds, dim=0)
            phrase_mask = torch.as_tensor(phrase_mask).float().to(self.device)
            if cfg.MODEL.VG.USING_DET_KNOWLEDGE:
                phrase_glove_embedding = torch.cat(phrase_glove_embedding, dim=0).to(self.device)  ## numP, 300


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




