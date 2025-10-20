# 模型微调，训练对话能力
import argparse
import json
import math
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from transformer.gpt.model import GPT


class SFTDataSet(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=1024) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len  # 根据最大输入长度进行阶段或填充
        self.samples = self.load_data(jsonl_path)
        # [1, 1078, 538, 501]， [1]是<|im_start|>这个特殊token的id，[1078, 538, 501]是assistant的分词id
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        # [2]
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, jsonl_path):
        """
        加载对话数据
        {'conversations': [{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '你好'}, ...]}
        """
        samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """
        将对话轮构建成符合 ChatML 格式的字符串
        然后用 tokenizer 的 apply_chat_template 方法统一构造 prompt
        """
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({'role': role, 'content': turn['content']})

        # 返回字符串格式的 prompt，而非直接 tokenize
        # ChatML 模板如下：
        #   {% for message in messages %}
        #   <|im_start|>{{ message['role'] }}
        #   {{ message['content'] }}<|im_end|>
        #   {% endfor %}
        # tokenize=False, 不进行分词，值返回字符串
        # add_generation_prompt=False，不在最后添加<|im_start|>assistant这样的生成提示，训练阶段有现成的。
        # 应用后得到的prompt形如：'<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好<|im_end|>\n'
        return self.tokenizer.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=False)

    def _generate_loss_mask(self, input_ids):
        """
        构建损失掩码，只有 assistant 的回答部分才参与 loss 计算
        找出每一段 assistant 的响应，在其 <|im_start|>assistant 和 <|im_end|> 之间，设置 loss_mask 为1
        """
        loss_mask = [0] * len(input_ids)
        for i in range(len(input_ids)):
            # 找 assistant 开始标记
            if input_ids[i: i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    # 找 assistant 结束标志
                    if input_ids[end: end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                loss_mask[start + 1: min(end + len(self.eos_id) + 1, self.max_len)] = 1
        return loss_mask

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 构建 ChatML 格式的 prompt 字符串
        prompt = self._create_chat_prompt(sample['conversations'])

        # 分词并截断，确保长度 <= max_len
        input_ids = self.tokenizer(prompt).input_ids[:self.max_len]

        # 不到长度则填充 pad token
        input_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))

        # 仅对 assistant 的回答部分参与 loss 计算
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练样本
        # 输入为前 n-1 个 token，预测第 2 到第 n 个 token
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def build_dataloader(data_path, tokenizer_path, max_len=512):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_ds = SFTDataSet(data_path, tokenizer, max_len=max_len)
    train_dl = DataLoader(train_ds,
                          batch_size=2,
                          pin_memory=True,
                          drop_last=False,
                          shuffle=False,
                          num_workers=0)
    return train_dl


def train_epoch(epoch):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_dl):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1),
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f'{args.save_dir}/sft_{args.d_model}.pth'
            state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()

def init_model(model_path, lm_config):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPT(lm_config)
    ckp = f'{args.save_dir}/pretrain_{lm_config.d_model}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(args.device)
    return model, tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl")
    args = parser.parse_args()



