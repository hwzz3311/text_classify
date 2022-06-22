import argparse
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import BaseConfig
from src.models.plugs import add_attention, add_lstm


class Config(BaseConfig):
    def __init__(self, args: argparse.ArgumentParser):
        super(Config, self).__init__(args)

        self.embedding_pretrain = torch.tensor(
            torch.load(os.path.join(args.data_dir,self.embedding))["embedding"].astype("float32")
        ) if self.embedding != "random" else None # 是否加载预训练的词向量
        self.embed = self.embedding_pretrain.size(1) if self.embedding_pretrain is not None else 300 # 词向量维度

        self.hidden_size = 128
        self.num_layers = 2
        self.hidden_size2 = 64

class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        if config.embedding_pretrain is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrain, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        x, _ = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out

