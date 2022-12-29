import argparse
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput

from src.models import Bert, MYBert
from src.models.plugs import add_lstm


class Config(MYBert.Config):

    def __init__(self, args: argparse.ArgumentParser):
        super(Config, self).__init__(args)
        self.rnn_hidden = self.hidden_size
        self.num_layers = 2


class Model(MYBert.Model):

    def __init__(self, config: Config):
        super(Model, self).__init__(config)
        # 添加cnn
        # self._lstm = types.MethodType(add_lstm, self)
        # self._lstm(config)
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, x):
        out = self.get_my_bert_res(x)
        out = out[0]

        out, _ = self.lstm(out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out
