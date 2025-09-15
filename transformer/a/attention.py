from torch import nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch, time, dim = q.shape
        n_d = self.d_model // self.num_heads
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch, time, self.num_heads, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.num_heads, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.num_heads, n_d).permute(0, 2, 1, 3)
        score = q @ k.transpose(2, 3) / math.sqrt(n_d)
        if mask:
            score = score.masked_fill(mask == 0, -1e9)
        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dim)
        return self.w_combine(score)


