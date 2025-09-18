import math

import torch
import torch.nn as nn


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
        self.beta = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        x = (x - mean) / math.sqrt(var + self.eps)
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

    def __init__(self, max_len, d_model, dropout=0.1):
        """
        max_len: 上下文长度
        d_model: 嵌入向量维度，或者特征维度
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        pos = torch.arange(max_len).reshape(-1, 1)  # shape [max_len, 1]
        w = torch.pow(10000.0, torch.arange(0, d_model, 2, dtype=torch.float64) / d_model)  # shape [1, d_model//2]
        # 矩阵形式
        self.PE = torch.zeros(max_len, d_model)
        self.PE.requires_grad = False  # 位置编码是固定的，不需要梯度
        self.PE[:, 0::2] = torch.sin(pos / w)  # 这里刚好也是d_model的一半
        self.PE[:, 1::2] = torch.cos(pos / w)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, seq_len = x.size()
        x = x + self.PE[:seq_len, :]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # 多头注意力需要把输入维度拆成 num_heads 个小空间，并行分别计算（每个头学习不同的关系模式）
        # 学的时候的 gpt-3 的 128 维就是这个 head_dim
        self.head_dim = d_model // num_heads
        # assert d_model % kq_dim == 0
        # 这里先用最简单的设置，即键-查询空间维度与嵌入空间相等
        # W_Q, W_K, W_V, W_{OUT}
        # 实际实现时，投影矩阵是所有头拼起来的大小
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # 注意力分数的 dropout
        self.attn_dropout = nn.Dropout(dropout)
        # out 投影后的 dropout
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 原本的形状是 [batch_size, seq_len, d_model]
        batch_size, seq_len = x.size()
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 调整成多头注意力需要的形状 [batch_size, seq_len, num_heads, head_dim]
        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # 调整下维度顺序，方便做batch矩阵运算 [batch_size, num_heads, seq_len, head_dim]
        xq = xq.permute(0, 2, 1, 3)
        xk = xk.permute(0, 2, 1, 3)
        xv = xv.permute(0, 2, 1, 3)

        pattern = xq @ xk.transpose(2, 3) / torch.sqrt(self.head_dim)


