import argparse
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, AutoModelForMaskedLM
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# from pytorch_pretrained_bert import BertTokenizer, BertModel
from transformers.modeling_outputs import MaskedLMOutput
from src.models import Bert, BertFreeze

from src.models.config import BaseConfig
from src.models.plugs import add_rcnn


class Config(BertFreeze.Config):
    """
    bert_type：
    """

    def __init__(self, args: argparse.ArgumentParser):
        super(Config, self).__init__(args)
        self.rnn_hidden = 256
        self.num_layers = 2


class Model(BertFreeze.Model):

    def __init__(self, config: Config):
        super(Model, self).__init__(config)
        # 添加rcnn
        self.rcnn = types.MethodType(add_rcnn, self)
        self.rcnn(config)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        out = self.bert(context, attention_mask=mask)
        encoder_out = out[0]
        out, _ = self.lstm(encoder_out)
        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
