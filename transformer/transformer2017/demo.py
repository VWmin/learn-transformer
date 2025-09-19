# ==== Demo ====
import torch

from transformer.transformer2017.model import EncoderDecoder

if __name__ == '__main__':
    batch_size = 2
    src_len = 6
    tgt_len = 5
    d_model = 16
    num_heads = 4
    d_ff = 64
    num_layers = 2
    max_len = 10

    # 假设词表大小
    vocab_size_src = 20
    vocab_size_tgt = 30
    src_pad_ix = 0
    tgt_pad_ix = 0

    # 随机生成输入（int 类型，表示 token index）
    src = torch.randint(1, vocab_size_src, (batch_size, src_len))
    tgt = torch.randint(1, vocab_size_tgt, (batch_size, tgt_len))

    # pad 部分填 0
    src[0, -1] = src_pad_ix
    tgt[1, -2:] = tgt_pad_ix

    # 初始化模型
    model = EncoderDecoder(
        src_pad_ix=src_pad_ix,
        tgt_pad_ix=tgt_pad_ix,
        src_vocab_size=vocab_size_src,
        tgt_vocab_size=vocab_size_tgt,
        d_model=d_model,
        max_len=max_len,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=0.1
    )

    print("源序列输入:", src)  # [batch_size, src_len]
    print("目标序列输入:", tgt)  # [batch_size, tgt_len]

    # 前向计算
    out = model(src, tgt)

    print("模型输出:", out)  # [batch_size, tgt_len, d_model]
