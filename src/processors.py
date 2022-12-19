import ast
import json
import os.path
import random
import re
from typing import List, Tuple

import numpy as np
import regex
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import DataProcessor, InputExample, BertModel
import pickle as pkl

from src.models.Bert import Config
from src.models.config import BaseConfig
from src.utils.model_utils import build_vocab

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
MAX_VOCAB_SIZE = 10000
UNK = '<UNK>'


def build_by_sentence(config: Config, sentences, vocab, label, pad_size=32):
    tokenizer = lambda x: [y for y in x]  # char-level

    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    contents = []
    for sentence in sentences:
        words_line = []
        if "bert" in config.model_name.lower():
            token = config.tokenizer.tokenize(sentence)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, label, seq_len, mask))
        else:
            token = tokenizer(sentence)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))

            # fasttext ngram
            if config.model_name == "FastText":
                buckets = config.n_gram_vocab
                bigram = []
                trigram = []
                # ------ngram------
                for i in range(pad_size):
                    bigram.append(biGramHash(words_line, i, buckets))
                    trigram.append(triGramHash(words_line, i, buckets))
                # -----------------
                contents.append((words_line, label, seq_len, bigram, trigram))
            else:
                contents.append((words_line, label, seq_len))

    return contents  # [([...], 0), ([...], 1), ...]


class build_dataset(Dataset):
    def __init__(self, config: BaseConfig, file_path, pad_size=32):
        self.config = config
        self.pad_size = pad_size
        self.data = self.load_file(file_path)
        self.tokenizer = lambda x: [y for y in x]  # char-level
        if os.path.exists(config.vocab_path):
            self.vocab = pkl.load(open(config.vocab_path, 'rb'))
        else:
            self.vocab = build_vocab(config.train_file, tokenizer=self.tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
            pkl.dump(self.vocab, open(config.vocab_path, 'wb'))
        print(f"Vocab size: {len(self.vocab)}")

    def biGramHash(self, sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(self, sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def load_file(self, file_path):
        res = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                json_data = ast.literal_eval(line)
                res.append(json_data)
        return res

    def __getitem__(self, index):
        json_data = self.data[index]
        try:
            content, label = json_data["text"], json_data["label"]
        except KeyError:
            print(json_data)
        words_line = []
        if "bert" in self.config.model_name.lower():
            token = self.config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = self.config.tokenizer.convert_tokens_to_ids(token)
            if self.pad_size:
                if len(token) < self.pad_size:
                    mask = [1] * len(token_ids) + [0] * (self.pad_size - len(token))
                    token_ids += ([0] * (self.pad_size - len(token)))
                else:
                    mask = [1] * self.pad_size
                    token_ids = token_ids[:self.pad_size]
                    seq_len = self.pad_size
            return (token_ids, self.config.class_list.index(label), seq_len, mask)
        else:
            token = self.tokenizer(content)
            seq_len = len(token)
            if self.pad_size:
                if len(token) < self.pad_size:
                    token.extend([PAD] * (self.pad_size - len(token)))
                else:
                    token = token[:self.pad_size]
                    seq_len = self.pad_size
            # word to id
            for word in token:
                words_line.append(self.vocab.get(word, self.vocab.get(UNK)))
            # fasttext ngram
            if self.config.model_name == "FastText":
                buckets = self.config.n_gram_vocab
                bigram = []
                trigram = []
                # ------ngram------
                for i in range(self.pad_size):
                    bigram.append(self.biGramHash(words_line, i, buckets))
                    trigram.append(self.triGramHash(words_line, i, buckets))
                # -----------------
                return (words_line, self.config.class_list.index(label), seq_len, bigram, trigram)
            else:
                return (words_line, self.config.class_list.index(label), seq_len)

    def __len__(self):
        return len(self.data)


def dataset_collate_fn(config: BaseConfig, datas: List):
    x = torch.LongTensor([_[0] for _ in datas]).to(config.device)
    y = torch.LongTensor([_[1] for _ in datas]).to(config.device)
    seq_len = torch.LongTensor([_[2] for _ in datas]).to(config.device)
    if config.model_name == "FastText":
        bigram = torch.LongTensor([_[3] for _ in datas]).to(config.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(config.device)
        return (x, seq_len, bigram, trigram), y
    elif "bert" in config.model_name.lower():
        mask = torch.LongTensor([_[3] for _ in datas]).to(config.device)
        return (x, seq_len, mask), y
    return (x, seq_len), y

# class build_dataset(object):
#     def __init__(self, config: BaseConfig):
#         self.config = config
#         self.tokenizer = lambda x: [y for y in x]  # char-level
#         if os.path.exists(config.vocab_path):
#             self.vocab = pkl.load(open(config.vocab_path, 'rb'))
#         else:
#             self.vocab = build_vocab(config.train_file, tokenizer=self.tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
#             pkl.dump(self.vocab, open(config.vocab_path, 'wb'))
#         print(f"Vocab size: {len(self.vocab)}")
#
#     def biGramHash(self, sequence, t, buckets):
#         t1 = sequence[t - 1] if t - 1 >= 0 else 0
#         return (t1 * 14918087) % buckets
#
#     def triGramHash(self, sequence, t, buckets):
#         t1 = sequence[t - 1] if t - 1 >= 0 else 0
#         t2 = sequence[t - 2] if t - 2 >= 0 else 0
#         return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets
#
#     def load_dataset(self, path, pad_size=32):
#         contents = []
#         with open(path, 'r', encoding='UTF-8') as f:
#             for line in tqdm(f):
#                 line = line.strip()
#                 json_data = ast.literal_eval(line)
#                 content, label = json_data["text"], json_data["label"]
#                 words_line = []
#                 if "bert" in self.config.model_name.lower():
#                     token = self.config.tokenizer.tokenize(content)
#                     token = [CLS] + token
#                     seq_len = len(token)
#                     mask = []
#                     token_ids = self.config.tokenizer.convert_tokens_to_ids(token)
#
#                     if pad_size:
#                         if len(token) < pad_size:
#                             mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
#                             token_ids += ([0] * (pad_size - len(token)))
#                         else:
#                             mask = [1] * pad_size
#                             token_ids = token_ids[:pad_size]
#                             seq_len = pad_size
#                     contents.append((token_ids, self.config.class_list.index(label), seq_len, mask))
#                 else:
#                     token = self.tokenizer(content)
#                     seq_len = len(token)
#                     if pad_size:
#                         if len(token) < pad_size:
#                             token.extend([PAD] * (pad_size - len(token)))
#                         else:
#                             token = token[:pad_size]
#                             seq_len = pad_size
#                     # word to id
#                     for word in token:
#                         words_line.append(self.vocab.get(word, self.vocab.get(UNK)))
#
#                     # fasttext ngram
#                     if self.config.model_name == "FastText":
#                         buckets = self.config.n_gram_vocab
#                         bigram = []
#                         trigram = []
#                         # ------ngram------
#                         for i in range(pad_size):
#                             bigram.append(self.biGramHash(words_line, i, buckets))
#                             trigram.append(self.triGramHash(words_line, i, buckets))
#                         # -----------------
#                         contents.append((words_line, self.config.class_list.index(label), seq_len, bigram, trigram))
#                     else:
#                         contents.append((words_line, self.config.class_list.index(label), seq_len))
#         return contents
#
#     def _jsonl_to_label_json(self, file_path):
#         res = {}
#         with open(file_path, "r") as f:
#             for line in f:
#                 line = line.strip()
#                 json_data = ast.literal_eval(line)
#                 label = json_data["category"]
#                 text = json_data["text"]
#                 if label not in res.keys():
#                     res[label] = [text]
#                 else:
#                     res[label].append(text)
#         return res
#
#     def get_train_dataset(self):
#
#         train = self.load_dataset(self.config.train_file, self.config.pad_size)
#         return train
#
#     def get_dev_dataset(self):
#         dev = self.load_dataset(self.config.eval_file, self.config.pad_size)
#         return dev
#
#     def get_test_dataset(self):
#         test = self.load_dataset(self.config.test_file, self.config.pad_size)
#         return test

def out_predict_dataset(predict_all_list, config: BaseConfig):
    bert_type = config.bert_type
    if "/" in bert_type:
        bert_type = config.bert_type.split("/")[1]
    out_file_name = f"{config.model_name}_predict_result.txt" \
        if "bert" not in config.model_name.lower() \
        else f"{config.model_name}_{bert_type}_predict_result.txt"
    predict_out_file = os.path.join(config.predict_out_dir, out_file_name)
    predict_out_file = open(predict_out_file, "w")
    with open(config.test_file, "r") as f:
        for index, line in enumerate(f):
            try:
                line = line.split("\t")[0]
                predict_out_file.write(line.replace("\n", "") + f"\t {predict_all_list[index]} \n")
            except:
                pass


# class DatasetIter(object):
#     def __init__(self, config: BaseConfig, batches, batch_size, device):
#         self.config = config
#         self.batch_size = batch_size
#         self.batches = batches
#         print(f"len(self.batches) = {len(self.batches)}")
#         self.n_batches = len(self.batches) // self.batch_size
#         # print(f"self.n_batches = {self.n_batches},self.batch_size = {self.batch_size},len(self.batches) = {len(self.batches)}")
#         self.residue = False
#         if len(batches) % self.n_batches != 0:
#             self.residue = True
#         self.index = 0
#         self.device = device
#
#     def _to_tensor(self, datas):
#         x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
#         y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
#         # pad 前的长度（超过pad_size的设为pad_size）
#         seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
#         # print(f"x:{x.shape}, seq_len:{seq_len.shape},y:{y.shape}")
#         if self.config.model_name == "FastText":
#             bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
#             trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)
#             return (x, seq_len, bigram, trigram), y
#         elif "bert" in self.config.model_name.lower():
#             mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
#             return (x, seq_len, mask), y
#
#         return (x, seq_len), y
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         # 当不够一个batch的时候
#         if self.residue and self.index == self.n_batches:
#             batches = self.batches[self.index * self.batch_size:]
#             self.index += 1
#             batches = self._to_tensor(batches)
#             return batches
#         # index 超出，则主动停止
#         elif self.index >= self.n_batches:
#             self.index = 0
#             raise StopIteration
#         else:
#             batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
#             # print("1.else __next__", len(batches))
#             self.index += 1
#             batches = self._to_tensor(batches)
#             # print("2.else __next__", len(batches))
#             return batches
#
#     def __len__(self):
#         return self.n_batches + 1 if self.residue else self.n_batches


# def build_iterator(config: BaseConfig, dataset, batch_size=None):
#     if batch_size is None:
#         batch_size = config.batch_size
#     iterator = DatasetIter(config, dataset, batch_size, config.device)
#     return iterator

class DatasetIterBertAtt:
    def __init__(self, dataset, batch_size, device, model_name, pad_size, bert_model, dropout, class_list, tokenizer):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = len(dataset) // batch_size
        self.device = device
        self.model_name = model_name
        self.pad_size = pad_size
        self.bert_model = bert_model
        self.dropout = dropout
        self.index = 0
        self.out = 0
        self.residue = False
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.config_classes = class_list
        self.config_tokenizer = tokenizer

    def cut_sent(self, para):
        patterns = ['([。;；！？\?])([^”’])', '(\.{6})([^”’])', '(\…{2})([^”’])', '([。！？\?][”’])([^，。！？\?])']

        if all([regex.search(p, para) is None for p in patterns]):
            para = para + '.'
        para = re.sub('([。;；！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 破折号、英文双引号等忽略，需要的再做些简单调整即可。
        sents = para.split("\n")
        sents = [s.strip() for s in sents]
        sents = [s for s in sents if s]
        return sents

    def be_bert_deal(self, dataset, class_list, tokenizer):
        # hl = md5()  # TODO SHL --md5
        # print('长度', len(dataset))
        token_dict = {}
        out = []
        labels = []
        for index, line in tqdm(enumerate(dataset), total=len(dataset)):
            context = line['text']
            label = line['label']
            label = class_list.index(label)
            sents = self.cut_sent(context)  # 对句子进行分句
            for sent in sents:
                # if md5(sent) not in token_dict.keys():
                # if sent not in token_dict.keys():

                    token = tokenizer.tokenize(sent)
                    token = [CLS] + token
                    seq_len = len(token)
                    token_ids = tokenizer.convert_tokens_to_ids(token)
                    mask = []
                    if self.pad_size:
                        if seq_len < self.pad_size:
                            mask = [1] * len(token_ids) + [0] * (self.pad_size - len(token))
                            mask = torch.LongTensor(mask).to(self.device)
                            token_ids += ([0] * (self.pad_size - len(token)))
                            token_ids = torch.LongTensor(token_ids).to(self.device)
                            seq_len = torch.LongTensor(seq_len).to(self.device)
                        else:
                            mask = [1] * self.pad_size
                            mask = torch.LongTensor(mask).to(self.device)
                            token_ids = token_ids[:self.pad_size]
                            token_ids = torch.LongTensor(token_ids).to(self.device)
                            seq_len = torch.LongTensor(self.pad_size).to(self.device)

                    # token_dict[md5(sent)] =
                    res = self.bert_model(token_ids.unsqueeze(0), attention_mask=mask.unsqueeze(0))
                    drop_data = nn.Dropout(self.dropout).to(self.device)
                    out_bert = drop_data(res.get('last_hidden_state')).to(self.device)
                    # contents.append((token_ids, label, seq_len, mask, out_bert))
                    # token_dict[sent] = (token_ids, label, seq_len, mask, out_bert)  # 字典中保存相应的结果，
                    self.out += out_bert
                # else:
                #     res_dict = token_dict[sent]
                #     out_bert = res_dict[4]
                #     self.out += out_bert
            out_1 = self.out
            out.append(out_1)
            labels.append(label)
            self.out = 0
        labels = torch.LongTensor(labels).to(self.device)
        return out, labels

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index += 1
            batches = self.be_bert_deal(dataset=batches, class_list=self.config_classes,
                                        tokenizer=self.config_tokenizer)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self.be_bert_deal(dataset=batches, class_list=self.config_classes,
                                        tokenizer=self.config_tokenizer)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches



def build_iter_bertatt(dataset: list,bert_model: BertModel, config: Config):
    iter_data = DatasetIterBertAtt(dataset,
                                   config.batch_size,
                                   config.device,
                                   config.model_name,
                                   config.pad_size,
                                   bert_model,
                                   config.dropout,
                                   config.class_list,
                                   config.tokenizer)
    return iter_data


def load_jsonl(path):
    res = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = ast.literal_eval(line)
            res.append(line)
    return res
