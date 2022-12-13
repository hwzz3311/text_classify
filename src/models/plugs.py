import torch
import torch.nn.functional as F
from torch import nn

from src.models.config import BaseConfig


def conv_and_pool(cls, x, conv):
    x = F.relu(conv(x)).squeeze(3)
    x = F.max_pool1d(x, x.size(2)).squeeze(2)
    return x


def add_cnn(cls, config: BaseConfig):
    if "bert" in config.model_name.lower():
        cls.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
    else:
        cls.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
    cls.dropout = nn.Dropout(config.dropout)

    cls.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
    return cls


def add_lstm(cls, config: BaseConfig):
    if hasattr(config, "rnn_hidden"):
        cls.lstm = nn.LSTM(config.rnn_hidden,
                           config.hidden_size,
                           config.num_layers,
                           bidirectional=True,
                           batch_first=True,
                           dropout=config.dropout)
    else:
        cls.lstm = nn.LSTM(config.embed,
                           config.hidden_size,
                           config.num_layers,
                           bidirectional=True,
                           batch_first=True,
                           dropout=config.dropout)
    cls.dropout = nn.Dropout(config.dropout)
    cls.fc = nn.Linear(config.hidden_size * 2,
                       out_features=config.num_classes)
    return cls


def add_attention(cls, config: BaseConfig):
    cls.tanh1 = nn.Tanh()
    cls.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
    cls.tanh2 = nn.Tanh()
    cls.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
    cls.fc = nn.Linear(config.hidden_size2, config.num_classes)
    return cls


def add_rcnn(cls, config: BaseConfig):
    if "bert" in config.model_name.lower():
        cls.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        cls.maxpool = nn.MaxPool1d(int(config.batch_size / 2) if int(config.batch_size / 2) > 0 else 1)
        cls.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size, config.num_classes)
    else:
        cls.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                           bidirectional=True, batch_first=True, dropout=config.dropout)

        cls.maxpool = nn.MaxPool1d(int(config.batch_size / 2) if int(config.batch_size / 2) > 0 else 1)
        cls.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)
    print(f"int(config.batch_size / 2) =  {int(config.batch_size / 2) if int(config.batch_size / 2) > 0 else 1}")
    return cls
