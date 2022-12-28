# coding: UTF-8
import argparse
import os

import numpy
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertSelfAttention
import copy
import math

# 定义配置文件
from src.models.config import BaseConfig


class Config(BaseConfig):
    """
    bert_type：
    """

    def __init__(self, args: argparse.ArgumentParser):
        super(Config, self).__init__(args)
        if self.local_model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_dir)
            self.bert_model: BertModel = AutoModel.from_pretrained(self.bert_dir).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
            self.bert_model: BertModel = AutoModel.from_pretrained(self.bert_type).to(self.device)
        self.hidden_size = 1024 if "large" in str(self.bert_dir).lower() \
                                   or "large" in str(self.bert_type).lower() else 768
        self.head = 8
        self.embedding = self.hidden_size
        self.rnn_hidden = self.hidden_size
        self.num_layers = 2


# 定义多头自注意力机制的类
# class MultiHeadedAttention(nn.Module):
#     def __init__(self, head, embedding_dim, dropout):
#         super(MultiHeadedAttention, self).__init__()
#         # 每个头的特征尺寸
#         self.d_k = embedding_dim // head
#         # 多少个头
#         self.head = head
#         # 线性层的列表
#         self.linears = self.clones(nn.Linear(embedding_dim, embedding_dim), 4)
#         # 注意力权重分布
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, query, key, value):  # mask=None
#         # 求多少批次batch_size  # [32,256,768]
#         batch_size = query.size()[0]
#         query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
#                              for model, x in zip(self.linears, (query, key, value))]
#         x, self.attn = self.attention(query, key, value)
#         x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
#         x = self.linears[-1](x)
#         return x
#
#     def attention(self, Q, K, V, mask=None, dropout=None):
#         """
#         这里使用的是自注意力机制Q=K=V
#         Q：经过bert模型输出的pooler_output
#         return: 注意力结果表示， 注意力权重分数
#         """
#         # 求查询张量的特征尺寸大小
#         d_k = Q.size()[-1]
#         # 求查询张量的权重分布
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
#         if mask is not None:
#             attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
#
#         p_attn = F.softmax(attn_scores, dim=-1)
#         if dropout is not None:
#             p_attn = self.dropout(p_attn)
#         return torch.matmul(p_attn, V), p_attn
#
#     def clones(self, module, N):
#         """
#         module: 需要克隆的模型
#         N：克隆的个数
#         """
#         return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask=None):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)


class Model(nn.Module):
    """
    整个模型结构和数据加载方案的介绍 ：
    需求：为了解决多个topic 如果微调bert模型导致的显存占用大的问题
    解决方案：不进行微调bert，把bert当作特征提取器；将样本中的每一句都单独的输入到bert中，然后将整篇文本/段落进行合并；
            使用合并后的embedding 去训练/微调一个两层的全连接；
            首先将拼接合并的embedding送入到attention中去提取到多个句子的核心，在将提取核心之后的embedding送入到两层全连接中去训练，最后将调整后的两层全连接的参数保存；
            上线时只需要去上线一个bert + N个两层全连接的模型即可
    """

    def __init__(self, config: Config):
        super(Model, self).__init__()
        # self.bert = config.model
        # if config.adjust:
        #     for name, param in self.bert.named_parameters():
        #         param.requires_grad = True
        # self.tokenizer = config.tokenizer
        self.attn = MultiHeadedAttention(config.head, config.embedding, config.dropout)

        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)
        # self.fc_01 = nn.Linear(config.hidden_size, int(config.hidden_size / 2))
        # self.fc_02 = nn.Linear(int(config.hidden_size / 2), config.num_classes)

    def forward(self, x):
        out = x  # 1，256，768
        out_all = None
        for i in out:
            p_attn = self.attn(i, i, i)
            lstm_out, _ = self.lstm(p_attn)
            drop_out = self.dropout(out)
            fc_out = self.fc(drop_out[:, -1, :])

            if out_all is None:
                out_all = fc_out
            else:
                out_all = torch.cat([out_all, fc_out], 0)

        return out_all


if __name__ == '__main__':
    # config = Config('环境违规')
    # from src.utils import be_bert_deal_01
    # model = Model(config=config)
    # sentences = '济南市章丘区环境保护局依据《中华人民共和国大气污染防治法》第一百零八条第(一)项罚款2万元。白环罚字[2017]11号显示，甘肃华骏产生的漆渣、废油漆桶等危险物未进行危废申报、废油漆桶未进行分类规范贮存。白银市白银区环境保护局依据《中华人民共和国固体废物污染环境防治法》第五十三条第一款、第七十五条第二款罚款2.66万元。'
    #
    # test_data = be_bert_deal_01(sentences, config=config)
    # with torch.no_grad():
    #     outputs = model(test_data)
    #     pre_result = torch.max(outputs.data, dim=1)[1].cpu()
    # pre_result = pre_result.numpy().tolist()

    # data_path = '/Users/songhaoli/PycharmProjects/daily-practice/daily-practice/src/data'
    # class_list = [x.strip() for x in open(data_path + '/class.txt').readlines()]
    # print(class_list)
    # tokenzier = tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    # sent = '宁夏回族自治区生态环境厅督查组要求，企业要加快封闭式储煤棚建设，抓紧完善裸露空地及道路扬尘防控措施。'
    # print(dict(tokenzier(sent, max_length=512)))
    # config = AutoConfig.from_pretrained("hfl/chinese-bert-wwm-ext")
    # print(config)
    # config = Config('环境违规')
    # print(config.save_path)
    # 测试2

    # 测试1 bert模型 基础bert模型
    from src.utils import load_dataset_v2, build_iter

    #
    config = Config('环境违规')
    model = Model(config=config)
    eval_data = load_dataset_v2(path=config.eval_path)
    eval_iter = build_iter(eval_data, config=config)
    for i, v in enumerate(eval_iter):
        print(v[0])

        print('111111', type(v[0]))
        res = model(v[0])
        # loss = F.cross_entropy(res, v[1])
        # print(res.shape)  # torch.Size([32, 2])
        # print('aaaaa', type(v[1]))
        # print('bbbbbb', type(res))
        print(res.shape)
        #
        break
        # print(res.shape)
        # -----------------------  获得批次数据的预测标签
        # pre_label = torch.max(res.data, 1)[1]
        # print(pre_label)
        # --------------- 获得批次数据的预测标签
        # pre = pre_label.numpy().tolist()
        # print(pre[0])
        # print(pre_label)
        # print(pre_label.shape)
        # print(res.shape)
        # print(res.data)

        # print(res)
        # print('res的形状----------->', res.size())
        # print(res.logits.shape)
        # print(res.logits)
        # print(res.get('pooler_output'))
        # print(res)
