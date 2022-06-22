import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, AutoModelForMaskedLM
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# from pytorch_pretrained_bert import BertTokenizer, BertModel
from transformers.modeling_outputs import MaskedLMOutput

from src.models.config import BaseConfig


class Config(BaseConfig):
    """
    bert_type：
    """

    def __init__(self, args: argparse.ArgumentParser):
        super(Config, self).__init__(args)
        if self.local_model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.hidden_size = 1024 if "large" in str(self.bert_dir).lower() \
                                   or "large" in str(self.bert_type).lower() else 768


class Model(nn.Module):

    def __init__(self, config: Config):
        super(Model, self).__init__()
        if config.local_model:
            self.bert = AutoModel.from_pretrained(config.bert_dir)
        else:
            self.bert = AutoModel.from_pretrained(config.bert_type)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

        out: MaskedLMOutput = self.bert(context, attention_mask=mask)
        out = self.fc(out.get("pooler_output"))
        return out
