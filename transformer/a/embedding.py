import torch
from torch import nn


class TokenEmbedding(nn.Embedding):
    """将输入的词汇表转换成指定维度的嵌入向量"""

    def __init__(self, vocab_size, d_model):
        # 索引为1的token是填充符号
        super().__init__(vocab_size, d_model, padding_idx=1)

class PositionalEmbedding(nn.Module):
    """
    计算位置信息
    PE_{(pos, 2i)}    =   sin(pos / 10000^{2i/d_{model}})
    PE_{(pos, 2i+1)}  =   cos(pos / 10000^{2i/d_{model}})
    """

    def __init__(self, d_model, max_len):
        super().__init__()
        self.PE = torch.zeros(max_len, d_model)
        self.PE.requires_grad = False #
        pos = (torch.arange(0, max_len).reshape(-1, 1)
               / torch.pow(1e4, torch.arange(0, d_model, 2) / d_model))
        # 对每个行向量添加如下位置编码
        self.PE[:, 0::2] = torch.sin(pos)
        self.PE[:, 1::2] = torch.cos(pos)

    def forward(self, X):
        batch_size, seq_len = X.size()
        return self.PE[:seq_len, :]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout):
        super().__init__()
        self.token_embed = TokenEmbedding(vocab_size, d_model)
        self.pos_embed = PositionalEmbedding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        token_embed = self.token_embed(X)
        pos_embed = self.pos_embed(X)
        return self.dropout(token_embed + pos_embed)

