# 从输入-》模型-》输出的顺序实现

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
from typing import Optional, Tuple, Any
from typing import List, Optional, Tuple
import math
import warnings

# 假设文本有两个句子，每个句子4个词
X = torch.zeros((2, 4),dtype=torch.long)
# 词库大小为10（包括一个接受任意词的unk和一个pad），每个嵌入向量的特征维度是8
embed = nn.Embedding(10,8)


def positional_encoding(X, num_features, dropout_p=0.1, max_len=512) -> torch.Tensor:
    """
    添加位置编码
    :param X:
    :param num_features:
    :param dropout_p: dropout probability
    :param max_len: max token size
    :return:
    """
    dropout = nn.Dropout(dropout_p)
    p = torch.zeros(1, max_len, num_features)
    """
    生成角度
    PE_{(pos, 2i)}   =  sin(pos / 10000^{2i/d_{model}})
    PE_{(pos, 2i+1)} =  cos(pos / 10000^{2i/d_{model}})
    所以arange生成位置序号，
    """
    X_ = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(1e4, torch.arange(0, num_features, 2, dtype=torch.float32) / num_features)
    print(X_.shape)

# positional_encoding(X, 8)
a = torch.arange(512, dtype=torch.float32).reshape(-1, 1)
print(a.shape)