import json
import os
import random
from tokenizers import ( # tokenizers 库是一个分词器库，用于训练和使用像BPE、WordPiece、Unigram等子词模型
    Tokenizer, # 核心分词器对象，控制整个分词、编码、解码过程
    decoders, # 用于将分词后的 token 转回原始文本
    models, # 包含这个子词分词模型，例如BPE、WordPiece、Unigram
    pre_tokenizers, # 定义文本的预处理方式，例如按空格分、字节分等
    trainers # 用于训练分词模型的工具，包括设置词表大小、特殊符号等，产出 model
)


# 定义一个使用BPE模型的分词器。（使用BPE算法将文本拆成子词单元，以增强对低频词和未登录词的处理能力）
tokenizer = Tokenizer(models.BPE())
# 预处理器将文本转换为字节级别的单位。中文不需要在每个单词前加空格
# ByteLevel是一个字符级别的预处理方式，将文本拆解为字节级的子单元
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 定义特殊token
special_tokens = [
    "<|endoftext|>", # 表示文本结束，用来结束生成
    "<|im_start|>",  # 表示对话开始，标记输入文本的起始
    "<|im_end|>"   # 表示对话结束，用来标记输入文本的结束
]

# 设置训练器，并添加特殊token
trainer = trainers.BpeTrainer(
    vocab_size=6400, # 训练过程最多会生成 6400 个子词，包括特殊
    special_tokens=special_tokens, # 防止特殊 token 被训练处理
    show_progress=True, #
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet() # 指定了BPE模型的初始字母表
)

def read_texts_from_jsonl(filepath, max_sample=100):
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_sample:
                break
            data = json.loads(line)
            yield data['text']

datapath = "D:\workspace\minimind\dataset\pretrain_hq.jsonl"
texts = read_texts_from_jsonl(datapath, 0x7fffffff)

# 训练 tokenizer
tokenizer.train_from_iterator(texts, trainer=trainer)

# 设置解码器，从 token id 序列转成为本时，能够正确还原被分词器按字节切分的内容
tokenizer.decoder = decoders.ByteLevel()

# 保存训练好的 tokenizer
tokenizer_dir = r"./model"
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
tokenizer.model.save(tokenizer_dir)


# 手动创建配置文件
config = {
    "add_bos_token": False,
    "add_eos_token": False,
    "add_prefix_space": False,
    "added_tokens_decoder": {
        "0": {
            "content": "<|endoftext|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        },
        "1": {
            "content": "<|im_start|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        },
        "2": {
            "content": "<|im_end|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        }
    },
    "additional_special_tokens": [],
    "bos_token": "<|im_start|>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "<|im_end|>",
    "legacy": True,
    "model_max_length": 32768,
    "pad_token": "<|endoftext|>",
    "sp_model_kwargs": {},
    "spaces_between_special_tokens": False,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "unk_token": "<|endoftext|>",
    "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
}

# 保存配置文件
with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
    json.dump(config, config_file, ensure_ascii=False, indent=4)

# 测试训练好的 tokenizer
def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print('decoder和原始文本是否一致：', response == new_prompt)

    print('\n输入文本：\n', new_prompt, '\n')
    print('解码文本：\n', response, '\n')


eval_tokenizer()
