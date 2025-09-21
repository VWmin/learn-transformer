import torch
import torch.nn as nn
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
        """

        :param d_model: 嵌入向量维度，或者特征维度
        :param eps: 极小值，防止除零
        """
        super(LayerNorm, self).__init__()
        # 为什么一个初始为全1，一个初始为全0？TODO
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        # 默认无偏估计是打开，为什么要关闭？TODO
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

    def __init__(self, max_len, d_model, dropout=0.1):
        """

        :param max_len: 上下文长度
        :param d_model: 嵌入向量维度，或者特征维度
        :param dropout: 丢弃概率
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
        pe = self.PE[:seq_len, :].unsqueeze(0)  # [1, seq_len, d_model]，实际相加时，1会广播到对方的 batch_size 维度
        return self.dropout(pe)


class Attention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        """

        :param d_model: 嵌入向量维度，或者特征维度
        :param num_heads: 注意力头数量
        :param dropout: 丢弃概率
        """
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

    def forward(self, q, k, v, mask=None):
        # 原本的形状是 [batch_size, seq_len, d_model]
        # 交叉注意力时，q来自decoder，kv来自encoder，此时两者seq_len可能不等长
        batch_size, tgt_len, d_model = q.size()
        _, src_len, _ = k.size()
        xq, xk, xv = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        # 调整成多头注意力需要的形状 [batch_size, seq_len, num_heads, head_dim]
        xq = xq.view(batch_size, tgt_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, src_len, self.num_heads, self.head_dim)
        xv = xv.view(batch_size, src_len, self.num_heads, self.head_dim)
        # 调整下维度顺序，方便做batch矩阵运算 [batch_size, num_heads, seq_len, head_dim]
        xq = xq.permute(0, 2, 1, 3)
        xk = xk.permute(0, 2, 1, 3)
        xv = xv.permute(0, 2, 1, 3)
        # @ 对最后两个维度做矩阵乘法，因此转置替换最后两个维度
        # scores 维度是 [batch_size, num_heads, tgt_len, src_len]
        scores = xq @ xk.transpose(-2, -1) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        # 防止依赖特定的模式
        scores = self.attn_dropout(scores)
        # output 维度是 [batch_size, num_heads, tgt_len, head_dim]
        output = scores @ xv
        # 转换后 output 维度是 [batch_size, tgt_len, num_heads, head_dim]
        output = output.permute(0, 2, 1, 3).contiguous()
        # 最终 output 需要回到初始形状
        output = output.view(batch_size, tgt_len, d_model)
        # 对多头结果做整合 [batch_zie, tgt_len, d_model] * [d_model, d_model]
        output = self.o_proj(output)
        # 防止依赖单一头或局部特征
        output = self.out_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """

        :param d_model: 嵌入向量维度，或者特征维度
        :param d_ff: 隐藏层维度
        :param dropout: 丢弃概率
        """
        super(FeedForward, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 上投影矩阵，映射到一个隐藏层维度
        # 已经包含了bias
        self.fc1 = nn.Linear(d_model, d_ff)
        # 下投影矩阵，回到特征维度
        self.fc2 = nn.Linear(d_ff, d_model)

        # Xavier 初始化权重
        # 其原理是调整初始权重范围，与输入和输出的方差保持一致，加快模型收敛
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        # dropout 放在激活函数之后
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        :param d_model: 特征维度
        :param num_heads: 注意力头数量
        :param d_ff: 前向传播层维度
        :param dropout: 丢弃概率
        """
        super(EncoderLayer, self).__init__()
        self.attn = Attention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # 残差连接之后，layer norm 之前
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, s_mask):
        _x = x
        x = self.attn(x, x, x, s_mask)
        # 先残差 + dropout，再 norm。下同
        x = self.dropout1(x + _x)
        x = self.norm1(x)
        _x = x
        x = self.ff(x)
        x = self.dropout2(x + _x)
        x = self.norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, num_heads, d_ff, num_layers, dropout=0.1):
        """

        :param vocab_size: 词库大小，构建词嵌入矩阵
        :param d_model: 特征维度
        :param max_len: 最大上下文长度
        :param num_heads: 注意力头数量
        :param d_ff: 隐藏层维度
        :param num_layers: 子层数量
        :param dropout: 丢弃概率
        """
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = SinusoidalPositionalEncoding(max_len, d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, s_mask):
        x = self.embed(x) + self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        :param d_model: 特征维度
        :param num_heads: 注意力头数量
        :param d_ff: 前向传播层维度
        :param dropout: 丢弃概率
        """
        super(DecoderLayer, self).__init__()
        self.attn = Attention(d_model, num_heads, dropout)
        self.cross_attn = Attention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc, s_mask, t_mask):
        _x = x
        x = self.attn(x, x, x, t_mask)
        x = self.dropout1(x + _x)
        x = self.norm1(x)

        _x = x
        x = self.cross_attn(x, enc, enc, s_mask)
        x = self.dropout2(x + _x)
        x = self.norm2(x)

        _x = x
        x = self.ff(x)
        x = self.dropout3(x + _x)
        x = self.norm3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, num_heads, d_ff, num_layers, dropout=0.1):
        """
        :param vocab_size: 词库大小，构建词嵌入矩阵
        :param d_model: 特征维度
        :param max_len: 最大上下文长度
        :param num_heads: 注意力头数量
        :param d_ff: 隐藏层维度
        :param num_layers: 子层数量
        :param dropout: 丢弃概率
        """
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = SinusoidalPositionalEncoding(max_len, d_model, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, enc, s_mask, t_mask):
        x = self.embed(x) + self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, enc, s_mask, t_mask)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, src_pad_ix, tgt_pad_ix, src_vocab_size, tgt_vocab_size, d_model, max_len, num_heads, d_ff,
                 num_layers, dropout=0.1):
        """

        :param src_pad_ix: 源序列中 padding 位置
        :param tgt_pad_ix: 目标序列中 padding 位置
        :param src_vocab_size: 源序列词库大小
        :param tgt_vocab_size: 目标序列词库大小
        :param d_model: 特征维度
        :param max_len: 最大上下文长度
        :param num_heads: 注意力头大小
        :param d_ff: 隐藏层维度
        :param num_layers: 子层数量
        :param dropout: 丢弃概率
        """
        super(EncoderDecoder, self).__init__()
        self.src_pad_ix = src_pad_ix
        self.tgt_pad_ix = tgt_pad_ix
        self.encoder = Encoder(src_vocab_size, d_model, max_len, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, max_len, num_heads, d_ff, num_layers, dropout)

    @staticmethod
    def make_padding_mask(k, pad_idx_k):
        mask = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        return mask

    @staticmethod
    def make_casual_mask(q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.triu(torch.ones(len_q, len_k, dtype=torch.bool))
        # [1, 1, len_q, len_k]
        mask = mask.unsqueeze(0).unsqueeze(1)
        return mask

    def forward(self, src, tgt):
        s_mask = self.make_padding_mask(src, self.src_pad_ix)
        t_mask = self.make_padding_mask(tgt, self.tgt_pad_ix) & self.make_casual_mask(tgt, tgt)
        enc = self.encoder(src, s_mask)
        dec = self.decoder(tgt, enc, s_mask, t_mask)
        return dec

