# ==== Demo ====
import torch

from transformer.gpt.model import GPT

if __name__ == '__main__':
    batch_size = 2
    seq_len = 6
    d_model = 16
    num_heads = 4
    d_ff = 64
    num_layers = 2
    max_len = 10

    # 假设词表大小
    vocab_size = 20
    pad_ix = 0

    # 随机生成输入（int 类型，表示 token index）
    src = torch.randint(1, vocab_size, (batch_size, seq_len))

    # pad 部分填 0
    src[0, -1] = pad_ix

    # 初始化模型
    model = GPT(
        vocab_size=vocab_size,
        pad_ix=pad_ix,
        d_model=d_model,
        max_len=max_len,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=0.1
    )

    print("源序列输入:", src)

    # 前向计算
    out = model(src)

    print("模型输出:", out)  # [batch_size, tgt_len, d_model]
