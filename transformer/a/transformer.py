import torch
import torch.nn as nn

from transformer.a.decoder import Decoder
from transformer.a.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self,
                 src_pad_ix,
                 trg_pad_ix,
                 enc_vocab_size,
                 dec_vocab_size,
                 d_model,
                 max_len,
                 num_heads,
                 ffn_hidden,
                 num_layers,
                 dropout):
        super(Transformer, self).__init__()
        self.encoder = Encoder(enc_vocab_size, max_len, d_model, ffn_hidden, num_heads, num_layers, dropout)
        self.decoder = Decoder(dec_vocab_size, max_len, d_model, ffn_hidden, num_heads, num_layers, dropout)

        self.src_pad_ix = src_pad_ix
        self.trg_pad_ix = trg_pad_ix

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(3)
        k = k.repeat(1, 1, 1, len_q)
        mask = q & k
        return mask

    def make_casual_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.trill(torch.ones(len_q, len_k)).type(torch.BoolTensor)
        return mask

    def forward(self, src, trg):
        s_mask = self.make_pad_mask(src, src, self.src_pad_ix, self.src_pad_ix)
        t_mask = self.make_pad_mask(trg, trg, self.trg_pad_ix, self.trg_pad_ix) * self.make_casual_mask(trg, trg) # 逻辑与
        enc = self.encoder(src, s_mask)
        out = self.decoder(trg, enc, t_mask, s_mask)
        return out



