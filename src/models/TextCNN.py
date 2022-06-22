import argparse
import os
import types

import torch
import torch.nn as nn
import torch.functional as F

from src.models.config import BaseConfig
from src.models.plugs import add_cnn, conv_and_pool


class Config(BaseConfig):
    def __init__(self, args: argparse.ArgumentParser):
        super(Config, self).__init__(args)

        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels)
        self.embedding_pretrain = torch.tensor(
            torch.load(os.path.join(args.data_dir, self.embedding))["embedding"].astype("float32")
        ) if self.embedding != "random" else None  # 是否加载预训练的词向量
        self.embed = self.embedding_pretrain.size(1) if self.embedding_pretrain is not None else 300 # 词向量维度


class Model(nn.Module):

    def __init__(self, config: Config):
        super(Model, self).__init__()
        if config.embedding_pretrain is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrain, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, embedding_dim=config.embed, padding_idx=config.n_vocab - 1)
        # 添加cnn
        self.cnn = types.MethodType(add_cnn, self)
        self.cnn(config)
        self.conv_and_pool = types.MethodType(conv_and_pool, self)

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

