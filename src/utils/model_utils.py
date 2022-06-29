import os
import pickle
import random
import time
from datetime import timedelta

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.models.config import BaseConfig

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
MAX_VOCAB_SIZE = 10000
UNK = '<UNK>'


def get_all_models() -> list:
    filedir = os.path.dirname(__file__)
    models_dir = os.path.join(filedir, "../models/")
    model_files = [file_name.replace(".py", "") for file_name in os.listdir(models_dir) if file_name[0].isupper()]
    return model_files

def get_bert_types():
    e = {
        "bert_type": {
            "bert": ["bert-base-chinese", "hfl/chinese-bert-wwm-ext", ],
            "robert": ["hfl/chinese-roberta-wwm-ext", ],
            "albert": ["albert-base-v2", "albert-base-v1"],
            "xlnet": ["hfl/chinese-xlnet-base", "hfl/chinese-xlnet-mid"],
            "electr": ["hfl/chinese-electra-180g-base-discriminator"],
            "distilbert": ["distilbert-base-uncased", "ASCCCCCCCC/distilbert-base-chinese-amazon_zh_20000", ],
            "ernie": ["nghuyong/ernie-gram-zh", "nghuyong/ernie-1.0"]
        }
    }
    return e


def set_seed(config: BaseConfig):
    """
    设置随机种子
    :param config:
    :return:
    """
    # random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    # TODO 检查是否将代码放在此处
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


def init_network(model: nn.Module, method="xavier", exclude="embedding", seed=123):
    """
    初始化网络参数
    :param model:
    :param method:
    :param exclude: 需要排除的层
    :param seed:
    :return:
    """
    for name, w in model.named_parameters():
        if exclude not in name:
            if "weight" in name:
                if method == "xavier":
                    nn.init.xavier_normal_(w)
                elif method == "kaiming":
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif "bias" in name:
                nn.init.constant_(w, 0)
            else:
                pass

def get_vocab(config: BaseConfig, use_word=False):
    if use_word:
        tokenizer = lambda x: x.split(" ")  # 空格分割
    else:
        tokenizer = lambda x: [y for y in x]  # char分类
    if os.path.exists(config.vocab_path):
        vocab = pickle.load(open(config.vocab_path, "rb"))
    else:
        vocab = build_vocab(config.train_file, tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pickle.dump(vocab, open(config.vocab_path, "wb"))
    print(f"Vocab size:{len(vocab)}")
    return vocab


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def load_and_cache_examples():
    pass
    # glue_convert_examples_to_features


def get_time_dif(start_time: float):
    time_dif = time.time() - start_time
    return timedelta(seconds=int(round(time_dif)))



def modelsize(model, input, type_size=4):
    """
    # 模型显存占用监测函数
    # model：输入的模型
    # input：实际中需要输入的Tensor变量
    # type_size 默认为 4 默认类型为 float32
    :param model:
    :param input:
    :param type_size:
    :return:
    """
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))


if __name__ == "__main__":
    pass
    # print(all_model_type())
