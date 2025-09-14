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

    def forward(self, query, key, value, mask=None):
        batch, time, dim = query.shape
        n_d = self.d_model // self.num_heads
        query, key, value = self.w_q(query), self.w_k(key), self.w_v(value)
        query = query.view(batch, time, self.num_heads, n_d).permute(0, 2, 1, 3)
        key = key.view(batch, time, self.num_heads, n_d).permute(0, 2, 1, 3)
        value = value.view(batch, time, self.num_heads, n_d).permute(0, 2, 1, 3)
        score = query @ key.transpose(2, 3) / math.sqrt(n_d)
        if mask:
            score = score.masked_fill(mask == 0, -1e9)
        score = self.softmax(score) @ value
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dim)
        return self.w_combine(score)


