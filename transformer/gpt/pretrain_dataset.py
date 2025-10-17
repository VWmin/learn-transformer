import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# 按照 pytorch 的接口构建 dataset 类
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512, max_samples=None):
        super(PretrainDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = self.load_data(data_path, max_samples)

    def load_data(self, path, max_samples=None):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if max_samples is not None and i > max_samples:
                    break
                data = json.loads(line.strip())
                samples.append(data)

        return samples

    def __len__(self):
        """返回样本数量"""
        return len(self.samples)

    def __getitem__(self, idx):
        """返回第 idx 个样本"""
        sample = self.samples[idx]

        # 将样本中的文本进行分词
        encoding = self.tokenizer(
            str(sample['text']),  # 确保数据类型
            max_length=self.max_len,  # 限制最大长度
            padding='max_length',  # 按照max_length模式填padding：强制补充pad到最大长度
            truncation=True,  # 超出部分截断
            return_tensors='pt'  # 返回pytorch tensor 形式
        )

        # 获取 input_ids 张量，去除 batch 维度
        # [max_len]
        input_ids = encoding.input_ids.squeeze()

        # 计算 loss_mask，pad 位置不参与 loss
        # [max_len]
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # 使用前一个 token 预测下一词
        X = input_ids[:-1].clone()  # [0, ..., n-2]
        Y = input_ids[1:].clone()  # [1, ..., n-1]
        # loss mask 与目标 Y 对齐
        loss_mask = loss_mask[1:].long()

        return X, Y, loss_mask


if __name__ == '__main__':
    # 测试代码
    max_len = 512
    datapath = "./dataset/pretrain_hq.jsonl"  # 使用相对路径
    tokenizer = AutoTokenizer.from_pretrained('./model')  # 使用相对路径
    train_ds = PretrainDataset(datapath, tokenizer, max_len=max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print(f"训练数据加载器批次数: {len(train_loader)}")
