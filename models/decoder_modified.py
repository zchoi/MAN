# -*- coding: utf-8 -*-
from http.client import NON_AUTHORITATIVE_INFORMATION
import random
from socket import NI_MAXHOST
import torch
import torch.nn as nn
from .module import *
from .attention import *
from .allennlp_beamsearch import *
from torch.nn.utils.weight_norm import weight_norm
from .attention_TRM import MultiHeadAttention
from .utils import sinusoid_encoding_table, PositionWiseFeedForward
from .containers import Module, ModuleList
import ipdb

class Decoder(nn.Module):
    def __init__(self, opt, filed):
        super(Decoder, self).__init__()
        self.region_projected_size = opt.region_projected_size
        self.hidden_size = opt.hidden_size
        self.att_size = opt.att_size
        self.word_size = opt.word_size
        self.max_words = opt.max_words
        self.field = filed
        self.vocab_size = len(filed.vocab)
        self.beam_size = opt.beam_size
        self.use_multi_gpu = opt.use_multi_gpu
        self.use_loc = opt.use_loc
        self.use_rel = opt.use_rel
        self.use_func = opt.use_func
        self.batch_size = 32
        self.topk = opt.topk
        self.dataset = opt.dataset

        # modules
        if opt.attention == 'soft':
            self.att = SoftAttention(self.hidden_size, self.hidden_size, self.att_size)
        elif opt.attention == 'gumbel':
            self.att = GumbelTopkAttention(self.hidden_size, self.hidden_size, self.hidden_size, self.topk)
        elif opt.attention == 'myatt':
            self.att = MYAttention(self.hidden_size, self.hidden_size, self.att_size)

        # word embedding matrix
        self.word_embed = nn.Embedding(self.vocab_size, self.word_size)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(99, self.word_size, 0), freeze=True)
        self.word_drop = nn.Dropout(p=opt.dropout)
        self.fc_word = nn.Linear(self.word_size, self.hidden_size,)

        # transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead = 8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=1)

        # # attention lstm
        # visual_feat_size = opt.hidden_size #* 4 + opt.region_projected_size
        # att_insize = opt.hidden_size + opt.word_size + visual_feat_size
        # self.att_lstm = nn.LSTMCell(att_insize, opt.hidden_size)
        # self.att_lstm_drop = nn.Dropout(p=opt.dropout)

        # # language lstm
        # self.lang_lstm = nn.LSTMCell(opt.hidden_size * 2, opt.hidden_size)
        # self.lstm_drop = nn.Dropout(p=opt.dropout)

        # final output layer
        self.out_fc = nn.Linear(opt.hidden_size * 3, self.hidden_size)
        nn.init.xavier_normal_(self.out_fc.weight)
        self.word_restore = nn.Linear(opt.hidden_size, self.vocab_size)
        nn.init.xavier_normal_(self.word_restore.weight)

        # beam search: The BeamSearch class is imported from Allennlp
        # DOCUMENTS: https://docs.allennlp.org/master/api/nn/beam_search/
        self.beam_search = BeamSearch(self.field.vocab.stoi['<eos>'], self.max_words, self.beam_size, per_node_beam_size=self.beam_size)
        self.init_weights()

    def init_weights(self):
        self.word_embed.weight.data.copy_(torch.from_numpy(self.field.vocab.vectors))
        # self.word_embed.weight.data.copy_(self.field.vocab.vectors)

    def forward(self, frame_feats, cluster_feats, captions, teacher_forcing_ratio=1.0):
        self.batch_size = frame_feats.size(0)
        infer = True if captions is None else False

        # visual input of attention lstm
        global_feat = torch.mean(frame_feats, dim=1)

        # training stage
        outputs = []
        module_weights = []
        if not infer:
            # add a '<start>' sign
            start_id = self.field.vocab.stoi['<bos>'] # 1
            start_id = global_feat.data.new(global_feat.size(0)).long().fill_(start_id).reshape(-1, 1) # bs,
            cap_id = torch.cat([start_id, captions], dim=1) # bs 1+seq_len 
            
            cap_mask = (cap_id == self.field.vocab.stoi['<pad>'])

            mask_queries = (cap_id != self.field.vocab.stoi['<pad>']).float()  # (b_s, seq_len)
            seq = torch.arange(1, cap_id.size(1) + 1).view(1, -1).expand(cap_id.size(0), -1).to(cap_id.device)  # (b_s, seq_len)
            seq = seq.masked_fill(mask_queries == 0, 0)

            cap_emb = self.word_embed(cap_id) + self.pos_emb(seq) # bs seq 300
            cap_emb = self.fc_word(cap_emb)
            cap_emb = self.word_drop(cap_emb)  # b seq 300
            
            outputs = self.transformer_decoder(
                tgt = cap_emb.permute(1, 0, 2), 
                memory = frame_feats.permute(1, 0, 2),
                tgt_key_padding_mask = cap_mask
                )
            outputs = self.word_restore(outputs).permute(1, 0, 2)

        else:
            # apply beam search if beam size > 1 during testing
            # add a '<start>' sign
            start_id = self.field.vocab.stoi['<bos>']
            start_id = global_feat.data.new(global_feat.size(0)).long().fill_(start_id)
            
            # start_state = {'att_lstm_h': att_lstm_h, 'att_lstm_c': att_lstm_c, 'lang_lstm_h': lang_lstm_h,
            #                'lang_lstm_c': lang_lstm_c, 'global_feat': global_feat, 'frame_feats': frame_feats, 'cluster_feats' : cluster_feats,
            #                'previous_cells': previous_cells, }
            start_state = {
                'frame_feats': frame_feats,
                'selected_words': start_id.unsqueeze(1)
            }
            predictions, log_prob = self.beam_search.search(start_id, start_state, self.beam_step)
            max_prob, max_index = torch.topk(log_prob, 1)  # b*1
            max_index = max_index.squeeze(1)  # b
            for i in range(self.batch_size):
                outputs.append(predictions[i, max_index[i], :])
            outputs = torch.stack(outputs)
            module_weights = None

        return outputs, None

    def beam_step(self, last_predictions, current_state):
        '''
        A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept two arguments. The first being a tensor
            of shape ``(group_size,)``, representing the index of the predicted
            tokens from the last time step, and the second being the current state.
            The ``group_size`` will be ``batch_size * beam_size``, except in the initial
            step, for which it will just be ``batch_size``.
            The function is expected to return a tuple, where the first element
            is a tensor of shape ``(group_size, target_vocab_size)`` containing
            the log probabilities of the tokens for the next step, and the second
            element is the updated state. The tensor in the state should have shape
            ``(group_size, *)``, where ``*`` means any other number of dimensions.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, beam_size, max_steps)`` and ``log_probabilities``
            has shape ``(batch_size, beam_size)``.
        '''
        
        group_size = last_predictions.size(0)  # batch_size or batch_size*beam_size
        batch_size = self.batch_size
        log_probs = []
        new_state = {}
        num = int(group_size / batch_size)  # 1 or beam_size
        for k, state in current_state.items():
            if isinstance(state, list):
                state = torch.stack(state, dim=1)
            _, *last_dims = state.size()
            current_state[k] = state.reshape(batch_size, num, *last_dims)
            new_state[k] = []
        for i in range(num):
            # read current state
            # att_lstm_h = current_state['att_lstm_h'][:, i, :]
            # att_lstm_c = current_state['att_lstm_c'][:, i, :]
            # lang_lstm_h = current_state['lang_lstm_h'][:, i, :]
            # lang_lstm_c = current_state['lang_lstm_c'][:, i, :]
            # global_feat = current_state['global_feat'][:, i, :]
            frame_feats = current_state['frame_feats'][:, i, :] # bs num_frames dim
            selected_words = current_state['selected_words'][:, i, :] # bs previous_seq

            # cluster_feats = current_state['cluster_feats'][:, i, :]
            # previous_cells = current_state['previous_cells'][:, i, :]

            # decoding stage
            if num == 1:
                word_id = last_predictions.reshape(batch_size, -1)[:, i].unsqueeze(1)
            else:
                word_id = torch.cat([selected_words, last_predictions.reshape(batch_size, -1)[:, i].unsqueeze(1)], dim=1)
            # attention lstm
            # att_lstm_h, att_lstm_c = self.att_lstm(torch.cat([lang_lstm_h, global_feat, word], dim=1),
            #                                        (att_lstm_h, att_lstm_c))
            # att_lstm_h = self.att_lstm_drop(att_lstm_h)

            # language lstm decoder
            word_logits = self.decode(frame_feats, word_id)[:, -1, :].squeeze()
            # previous_cells = torch.cat([previous_cells, lang_lstm_c.unsqueeze(1)], dim=1)
            # store log probabilities
            log_prob = F.log_softmax(word_logits, dim=1)  # b*v
            log_probs.append(log_prob)

            # update new state
            # new_state['att_lstm_h'].append(att_lstm_h)
            # new_state['att_lstm_c'].append(att_lstm_c)
            # new_state['lang_lstm_h'].append(lang_lstm_h)
            # new_state['lang_lstm_c'].append(lang_lstm_c)
            # new_state['global_feat'].append(global_feat)
            new_state['frame_feats'].append(frame_feats)
            new_state['selected_words'].append(word_id)
            # new_state['cluster_feats'].append(cluster_feats)
            # new_state['previous_cells'].append(previous_cells)

        # transform log probabilities
        # from list to tensor(batch_size*beam_size, vocab_size)
        log_probs = torch.stack(log_probs, dim=0).permute(1, 0, 2).reshape(group_size, -1)  # group_size*vocab_size

        # transform new state
        # from list to tensor(batch_size*beam_size, *)
        for k, state in new_state.items():
            new_state[k] = torch.stack(state, dim=0)  # (beam_size, batch_size, *)
            _, _, *last_dims = new_state[k].size()
            dim_size = len(new_state[k].size())
            dim_size = range(2, dim_size)
            new_state[k] = new_state[k].permute(1, 0, *dim_size)  # (batch_size, beam_size, *)
            new_state[k] = new_state[k].reshape(group_size, *last_dims)  # (batch_size*beam_size, *)
        return (log_probs, new_state)

    def decode(self, frame_feats, last_word_id):

        # LOCATE, RELATE, FUNC modules
        
        if not self.use_rel and not self.use_loc:
            raise ValueError('use locate or relation, all use both')
        # ipdb.set_trace()

        mask_queries = (last_word_id != self.field.vocab.stoi['<pad>']).float()  # (b_s, seq_len)
        seq = torch.arange(1, last_word_id.size(1) + 1).view(1, -1).expand(last_word_id.size(0), -1).to(last_word_id.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries == 0, 0)

        cap_emb = self.word_embed(last_word_id) + self.pos_emb(seq) # bs 1 300
        cap_emb = self.fc_word(cap_emb)
        
        output = self.transformer_decoder(
            tgt = cap_emb.permute(1, 0, 2), 
            memory = frame_feats.permute(1, 0, 2)
            )
        word_logits = self.word_restore(output).permute(1, 0, 2)
        return word_logits

    def decode_tokens(self, tokens):
        '''
        convert word index to caption
        :param tokens: input word index
        :return: capiton
        '''
        words = []
        for token in tokens:
            if token == self.field.vocab.stoi['<eos>']:
                break
            if self.dataset == 'msvd':
                word = self.field.vocab.itos[str(token.item())]
            elif self.dataset == 'msr-vtt':
                word = self.field.vocab.itos[int(token.item())]
            elif self.dataset == 'vatex':
                word = self.field.vocab.itos[str(token.item())]
            words.append(word)
            # if len(words)== 0:
            #     words.append(word)
            # elif word != words[-1]:
            #         words.append(word)
        captions = ' '.join(words)
        return captions
