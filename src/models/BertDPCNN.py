import argparse
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput

from src.models import Bert
from src.models.plugs import add_lstm


class Config(Bert.Config):

    def __init__(self, args: argparse.ArgumentParser):
        super(Config, self).__init__(args)
        self.num_filters = 250                                          # 卷积核数量(channels数)


class Model(Bert.Model):

    def __init__(self, config: Config):
        super(Model, self).__init__(config)
        # self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.hidden_size), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        out = self.bert(context, attention_mask=mask)
        encoder_out = out[0]
        out = torch.unsqueeze(encoder_out, dim=1)
        # x = encoder_out.unsqueeze(1)  # [batch_size, 1, seq_len, embed]
        out = self.conv_region(out)  # [batch_size, 250, seq_len-3+1, 1]
        out = self.padding1(out)  # [batch_size, 250, seq_len, 1]
        out = self.relu(out)
        out = self.conv(out)  # [batch_size, 250, seq_len-3+1, 1]
        out = self.padding1(out)  # [batch_size, 250, seq_len, 1]
        out = self.relu(out)
        out = self.conv(out)  # [batch_size, 250, seq_len-3+1, 1]
        while out.size()[2] > 2:
            out = self._block(out)
        out = out.squeeze()  # [batch_size, num_filters(250)]
        out = self.fc(out)
        return out

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
        x = x + px  # short cut
        return x