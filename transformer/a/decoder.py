import torch
import torch.nn as nn

from transformer.a.attention import MultiHeadAttention
from transformer.a.embedding import TransformerEmbedding
from transformer.a.encoder import PositionWiseFeedForward
from transformer.a.layernorm import LayerNorm


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, dec, enc, t_mask, s_mask):
        raw = dec
        x = self.attn(dec, dec, dec, t_mask)
        x = self.dropout1(x)
        x = self.norm1(x + raw)
        raw = x
        x = self.cross_attn(x, enc, enc, s_mask)
        x = self.dropout2(x)
        x = self.norm2(x + raw)
        raw = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + raw)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, ffn_hidden, num_heads, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, ffn_hidden, num_heads,dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc, t_mask, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, enc, t_mask, s_mask)
        return self.fc(x)




