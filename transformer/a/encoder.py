import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.a.attention import MultiHeadAttention
from transformer.a.embedding import TransformerEmbedding
from transformer.a.layernorm import LayerNorm


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        raw = x
        x = self.attn(x, x, x, mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + raw)
        raw = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + raw)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, ffn_hidden, num_heads, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask=s_mask)
        return x


