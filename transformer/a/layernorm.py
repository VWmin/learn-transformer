from torch import nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # 计算均值
        mean = x.mean(-1, keepdim=True)
        # 计算方差
        var = x.var(-1, unbiased=False, keepdim=True)
        # 归一化
        out = (x - mean) / (var + self.eps).sqrt()
        # 通过可训练参数让向量在归一化后仍有表达能力
        out = self.gamma * out + self.beta
        return out


