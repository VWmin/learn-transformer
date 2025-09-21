import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    层正则化
    \hat{x_i} = \frac{x_i-\mu}{\sqrt{\sigma^2-\varepsilon}}
    y_i = \gamma \hat{x_i} + \beta
    what，层正则化是接在每个小模块后的模块，它的作用是加快模型的收敛，防止梯度爆炸或梯度消失
    how，计算向量的均值方差归一化
    why，经过均值方差归一化后的向量偏向稳定：
        1. 其所有项的均值为零。即向量的分布居中，不会整体偏移
        2. 方差为1。即向量的尺度被标准化
    """
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.gamma * x + self.beta
        return x

class SinusoidalPositionalEncoding(nn.Module):
    """
    正余弦位置编码
    PE_{(pos, 2i)}   =  sin(pos / 10000^{2i/d_{model}})
    PE_{(pos, 2i+1)} =  cos(pos / 10000^{2i/d_{model}})
    其中pos是词在上下文中的位置，i是词的嵌入向量的维度的位置整除2
    what, 给嵌入向量添加其在上下文中的位置信息
    how, 奇数位置用正弦函数，偶数位置用余弦函数
    why, 能很好的适应训练过程中没见过的文本长度
    """
    def __init__(self, max_len, d_model, dropout=.1):
        super(SinusoidalPositionalEncoding, self).__init__()
        w = torch.arange(max_len).reshape(-1, 1) / torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        self.PE = torch.zeros(max_len, d_model)
        self.PE[:, 0::2] = torch.sin(w)
        self.PE[:, 1::2] = torch.cos(w)
        self.requires_grad_ = False
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, seq_len = x.size()
        pe = self.PE[:seq_len, :]
        pe = pe.unsqueeze_(0)
        return self.dropout(pe)


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=.1):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        # [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        # [batch_size, seq_len, d_model]
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)
        # [batch_size, seq_len, num_heads, head_dim]
        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # [batch_size, num_heads, seq_len, head_dim]
        xq = xq.permute(0, 2, 1, 3)
        xk = xk.permute(0, 2, 1, 3)
        xv = xv.permute(0, 2, 1, 3)
        # [batch_size, num_heads, seq_len, seq_len]
        scores = xq @ xk.transpose(-2, -1) / torch.sqrt(torch.tensor(self.head_dim))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        # [batch_size, num_heads, seq_len, head_dim]
        output = scores @ xv
        # [batch_size, seq_len, num_heads, head_dim]
        output = output.permute(0, 2, 1, 3).contiguous()
        # [bath_size, seq_len, d_model]
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.o_proj(output)
        return self.dropout(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DecoderOnlyLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=.1):
        super(DecoderOnlyLayer, self).__init__()
        self.attn = Attention(d_model, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        _x = x
        x = self.attn(x, mask)
        x = self.dropout1(x + _x)
        x = self.norm1(x)
        _x = x
        x = self.ff(x)
        x = self.dropout2(x + _x)
        x = self.norm2(x)
        return x

class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, num_heads, d_ff, num_layers, dropout=.1):
        super(DecoderOnly, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = SinusoidalPositionalEncoding(max_len, d_model, dropout=dropout)
        self.layers = nn.ModuleList([DecoderOnlyLayer(d_model, num_heads, d_ff, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.embed(x) + self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, pad_ix, max_len, d_model, num_heads, d_ff, num_layers, dropout=.1):
        super(GPT, self).__init__()
        self.pad_ix = pad_ix
        self.decoder = DecoderOnly(vocab_size, max_len, d_model, num_heads, d_ff, num_layers, dropout=dropout)

    @staticmethod
    def make_causal_mask(x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool))
        # [1, 1, seq_len, seq_len]
        mask = mask.unsqueeze(0).unsqueeze(1)
        return mask

    def forward(self, x):
        causal_mask = self.make_causal_mask(x)
        dec = self.decoder(x, causal_mask)
        return dec















