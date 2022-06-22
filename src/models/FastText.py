import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.config import BaseConfig


class Config(BaseConfig):
    def __init__(self, args):
        super(Config, self).__init__(args)

        self.hidden_size = 256
        self.embedding_pretrained = torch.tensor(
            np.load(os.path.join(args.data_dir, args.embedding))["embeddings"].astype('float32')) \
            if args.embedding != 'random' else None  # 预训练词向量
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度
        self.n_gram_vocab = 250499  # ngram 词表大小


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(num_embeddings=config.n_vocab,
                                          embedding_dim=config.embed, padding_idx=config.n_vocab-1)
        self.embedding_n_gram2 = nn.Embedding(num_embeddings=config.n_gram_vocab, embedding_dim=config.embed)
        self.embedding_n_gram3 = nn.Embedding(num_embeddings=config.n_gram_vocab, embedding_dim=config.embed)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed * 3, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_n_gram2(x[2])
        out_trigram = self.embedding_n_gram3(x[3])

        out = torch.cat((out_word, out_bigram, out_trigram), -1)
        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out