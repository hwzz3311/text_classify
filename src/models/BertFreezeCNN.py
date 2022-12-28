import argparse
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer

from src.models import Bert, BertFreeze
from src.models.config import BaseConfig
from src.models.plugs import add_cnn, conv_and_pool


class Config(BertFreeze.Config):

    def __init__(self, args: argparse.ArgumentParser):
        super(Config, self).__init__(args)
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256

class Model(BertFreeze.Model):

    def __init__(self, config: Config):
        super(Model, self).__init__(config)
        # 添加cnn
        self.cnn = types.MethodType(add_cnn, self)
        self.cnn(config)
        self.conv_and_pool = types.MethodType(conv_and_pool, self)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        out = self.bert(context, attention_mask=mask)
        out = out[0]
        out = torch.unsqueeze(out, dim=1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

