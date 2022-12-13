import argparse
import os
import types

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.config import BaseConfig
from src.models.plugs import add_cnn, conv_and_pool, add_rcnn


class Config(BaseConfig):
    def __init__(self, args: argparse.ArgumentParser):
        super(Config, self).__init__(args)

        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels)
        self.embedding_pretrain = torch.tensor(
            torch.load(os.path.join(args.data_dir, self.embedding))["embedding"].astype("float32")
        ) if self.embedding != "random" else None  # 是否加载预训练的词向量
        self.embed = self.embedding_pretrain.size(1) if self.embedding_pretrain is not None else 300 # 词向量维度
        self.hidden_size = 256  # lstm隐藏层
        self.num_layers = 1  # lstm层数


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        if config.embedding_pretrain is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrain, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        # 添加rcnn
        self.rcnn = types.MethodType(add_rcnn, self)
        self.rcnn(config)

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
