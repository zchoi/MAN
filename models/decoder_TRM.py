# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .module import *
from .attention import *
from .allennlp_beamsearch import *
from .beam_search import *
from .attention_TRM import MultiHeadAttention
from .utils import sinusoid_encoding_table, PositionWiseFeedForward
from .containers import Module, ModuleList

class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        # MHA+AddNorm
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad
        # MHA+AddNorm:Image
        enc_att = self.enc_att(self_att, enc_output, enc_output, mask_enc_att)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        enc_att = enc_att * mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff

class Decoder_TRM(Module):
    def __init__(self, opt, filed):
        super(Decoder_TRM, self).__init__()
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
        self.w = nn.Linear(self.word_size, self.hidden_size)
        self.word_drop = nn.Dropout(p=opt.dropout)

        # attention lstm
        visual_feat_size = opt.hidden_size #* 4 + opt.region_projected_size
        att_insize = opt.hidden_size + opt.word_size + visual_feat_size

        ############################TRM############################
        self.d_model = opt.hidden_size
        self.N_dec = 1
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(self.max_words + 1, opt.hidden_size, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(opt.hidden_size) for _ in range(self.N_dec)])
        
        
        self.fc = nn.Linear(opt.hidden_size, self.vocab_size, bias=False)
        self.max_len = self.max_words
        self.padding_idx = self.field.vocab.stoi['<pad>']

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)

        # final output layer
        self.out_fc = nn.Linear(opt.hidden_size * 3, self.hidden_size)
        nn.init.xavier_normal_(self.out_fc.weight)
        self.word_restore = nn.Linear(opt.hidden_size, self.vocab_size)
        nn.init.xavier_normal_(self.word_restore.weight)

        self.beam_search = BeamSearch(self, self.max_words, self.field.vocab.stoi['<eos>'], self.beam_size)
        
        self.init_weights()

    def init_weights(self):
        self.word_embed.weight.data.copy_(torch.from_numpy(self.field.vocab.vectors))
        # self.word_embed.weight.data.copy_(self.field.vocab.vectors)

    def forward(self, frame_feats, cluster_feats, input, teacher_forcing_ratio=1.0):
        # input (b_s, seq_len)
        self.batch_size = frame_feats.size(0)
        infer = True if input is None else False
        if not infer or self.beam_size == 1:
            b_s, seq_len = input.shape[:2]
            mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
            mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                            diagonal=1)
            mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
            mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
            if self._is_stateful:
                self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
                mask_self_attention = self.running_mask_self_attention

            seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
            seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
            if self._is_stateful:
                self.running_seq.add_(1)
                seq = self.running_seq
            
            out = self.w(self.word_embed(input)) + self.pos_emb(seq)
            for i, l in enumerate(self.layers):
                out = l(out, frame_feats, mask_queries, mask_self_attention, mask_enc_att=None)
            out = self.fc(out)
            out = F.log_softmax(out.float(), dim=-1)
        else:
            out, log_probs = self.beam_search.apply(frame_feats, out_size=1, return_probs=False)

        return out, None

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

    def decode_TRM(self, input, frame_feats, mask_encoder=None):
        # input (b_s, seq_len)
        self.batch_size = frame_feats.size(0)
        
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                        diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.w(self.word_embed(input)) + self.pos_emb(seq)
        for i, l in enumerate(self.layers):
            out = l(out, frame_feats, mask_queries, mask_self_attention, mask_enc_att=None)
        out = self.fc(out)
        
        return F.log_softmax(out.float(), dim=-1)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.enc_output, self.mask_enc = visual, (torch.sum(visual, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.field.vocab.stoi['<bos>']).long() # self.bos_idx: '<bos>'
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.field.vocab.stoi['<bos>']).long()
            else:
                it = prev_output
        return self.decode_TRM(it, self.enc_output, self.mask_enc)
