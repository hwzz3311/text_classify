import ast
import json
import os.path
import random
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import regex
import torch
from sklearn import metrics
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import DataProcessor, InputExample, BertModel
import pickle as pkl

from src.models.Bert import Config
from src.models.config import BaseConfig
from src.utils.data_utils import gen_pattern
from src.utils.model_utils import predict_res_merger, build_vocab

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
    def __init__(self, config: BaseConfig, dataset, is_predict=False):
        self.config = config
        self.pad_size = config.pad_size
        self.is_predict = is_predict

        self.data = dataset
        self.tokenizer = lambda x: [y for y in x]  # char-level
        if "bert" not in str(config.model_name).lower():
            if os.path.exists(config.vocab_path):
                self.vocab = pkl.load(open(config.vocab_path, 'rb'))
            else:
                self.vocab = build_vocab(config.train_file, tokenizer=self.tokenizer, max_size=MAX_VOCAB_SIZE,
                                         min_freq=1)
                pkl.dump(self.vocab, open(config.vocab_path, 'wb'))
            print(f"Vocab size: {len(self.vocab)}")

    def biGramHash(self, sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(self, sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def __getitem__(self, index):
        json_data = self.data[index]
        try:
            content, label = json_data["text"], json_data["label"]
        except KeyError:
            print(json_data)
        if self.is_predict:
            news_id = json_data["news_id"]
        else:
            news_id = ""
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
            return (token_ids, self.config.class_list.index(label), seq_len, mask, news_id, content)
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
                return (
                    words_line, self.config.class_list.index(label), seq_len, bigram, trigram, news_id,
                    content)
            else:
                return (words_line, self.config.class_list.index(label), seq_len, news_id, content)

    def __len__(self):
        return len(self.data)


def dataset_collate_fn(config: BaseConfig, datas: List):
    x = torch.LongTensor([_[0] for _ in datas]).to(config.device)
    y = torch.LongTensor([_[1] for _ in datas]).to(config.device)
    seq_len = torch.LongTensor([_[2] for _ in datas]).to(config.device)
    if config.model_name == "FastText":
        bigram = torch.LongTensor([_[3] for _ in datas]).to(config.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(config.device)
        news_ids = [_[5] for _ in datas]
        origin_data = [_[6] for _ in datas]
        return (x, seq_len, bigram, trigram), y, news_ids, origin_data
    elif "bert" in config.model_name.lower():
        mask = torch.LongTensor([_[3] for _ in datas]).to(config.device)
        news_ids = [_[4] for _ in datas]
        origin_data = [_[5] for _ in datas]
        return (x, seq_len, mask), y, news_ids, origin_data
    news_ids = [_[3] for _ in datas]
    origin_data = [_[4] for _ in datas]
    return (x, seq_len), y, news_ids, origin_data


def out_predict_dataset(predict_all_list, predict_result_score_all, news_ids_list, origin_text_all,
                        config: BaseConfig):
    assert len(predict_all_list) == len(news_ids_list) == len(origin_text_all), "predict_all_list not equal length input_tokens_all"
    bert_type = config.bert_type
    if "/" in bert_type:
        bert_type = config.bert_type.split("/")[1]
    out_file_name = f"{config.model_name}_predict_result.txt" \
        if "bert" not in config.model_name.lower() \
        else f"{config.model_name}_{bert_type}_predict_result.txt"
    predict_out_file = os.path.join(config.predict_out_dir, out_file_name)
    predict_res_dict = predict_res_merger(predict_all_list, predict_result_score_all, news_ids_list, origin_text_all,
                                          config)
    predict_datas = load_jsonl(config.predict_file)
    t_labels = []
    p_labels = []
    compute_acc_flag = False
    for e in predict_datas:
        news_id = e["news_id"]
        if "label" in e.keys():
            compute_acc_flag = True
            if e["label"] in config.class_list:
                t_labels.append(config.class_list.index(e["label"]))
            else:
                t_labels.append(int(e["label"]))
            p_labels.append(int(predict_res_dict[news_id]["label"]))
    if compute_acc_flag:
        acc = metrics.accuracy_score(t_labels, p_labels)
        print(f"Acc : {acc}")
        report = metrics.classification_report(t_labels, p_labels)
        print(report)
        tn, fp, fn, tp = metrics.confusion_matrix(t_labels, p_labels).ravel()
        print(f"(tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}) ")
    for e in predict_datas:
        e["p_label"] = predict_res_dict[e["news_id"]]["label"]
        e["support_sentence"] = predict_res_dict[e["news_id"]]["support_sentence"]
        e["all_result_score"] = predict_res_dict[e["news_id"]]["all_result_score"]
    out_data_to_jsonl(predict_datas, predict_out_file)
    # 同时生成csv 文件
    for e in predict_datas:
        if len(e["support_sentence"]) > 0:
            e["support_sentence"] = "\n\n".join(e["support_sentence"])
    out_file_name, etx = os.path.splitext(out_file_name)
    pd.DataFrame(predict_datas).to_csv(out_file_name + ".csv")


def out_test_dataset(predict_all_list, config: BaseConfig):
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
    def __init__(self, dataset, config: BaseConfig, is_predict=False):
        self.config = config
        self.is_predict = is_predict
        self.dataset = dataset
        self.batch_size = config.batch_size
        self.n_batches = max(1, len(dataset) // config.batch_size)
        self.device = config.device
        self.model_name = config.model_name
        self.pad_size = config.pad_size
        self.bert_model = config.bert_model
        self.dropout = config.dropout

        self.index = 0
        self.out = 0
        self.residue = False
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.config_classes = config.class_list
        self.config_tokenizer = config.tokenizer

    def be_bert_deal(self, dataset, class_list, tokenizer):
        # hl = md5()  # TODO SHL --md5
        # print('长度', len(dataset))
        token_dict = {}
        out = []
        labels = []
        news_ids = []
        content_list = []
        for index, line in enumerate(dataset):
            context = line['text']
            label = line['label']
            news_id = ""
            if self.is_predict:
                news_id = line["news_id"]
            label = class_list.index(label)
            sents = cut_sent(context)  # 对句子进行分句
            token_ids_tmp_list = []
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
                        token_ids += ([0] * (self.pad_size - len(token)))
                    else:
                        mask = [1] * self.pad_size
                        token_ids = token_ids[:self.pad_size]

                # token_dict[md5(sent)] =
                token_ids_tmp_list.extend(token_ids)
                token_ids = torch.LongTensor(token_ids).to(self.device)
                mask = torch.LongTensor(mask).to(self.device)

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
            content_list.append(context)
            out_1 = self.out
            out.append(out_1)
            labels.append(label)
            news_ids.append(news_id)
            self.out = 0
        labels = torch.LongTensor(labels).to(self.device)
        return out, labels, news_ids, content_list

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


def build_iter_bertatt(dataset: list, config: Config, is_predict=False):
    iter_data = DatasetIterBertAtt(dataset,
                                   config,
                                   is_predict)
    return iter_data


def load_jsonl(path):
    res = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = ast.literal_eval(line)
            res.append(line)
    return res


def cut_sent(para):
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


def para_clear(para):
    patterns = ['([。;；！？\?])([^”’])', '(\.{6})([^”’])', '(\…{2})([^”’])', '([。！？\?][”’])([^，。！？\?])']

    if all([regex.search(p, para) is None for p in patterns]) and not str(para).endswith("."):
        para = para + '.'
    para = re.sub('([。;；！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para


def cut_sentences(json_data, config):
    para = json_data.get("title", "") + "\n" + json_data.get("content", "")
    para = para_clear(para)
    # 破折号、英文双引号等忽略，需要的再做些简单调整即可。
    sentences = regex.split("([?？!！。.])", para)
    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    res = []
    for i in range(0, len(sentences)):
        text = "".join(sentences[i:i + config.cut_sen_len])  # 每steps 句拼接为一个text，进一次模型
        text = text.replace("\n", " ")
        res.append(text)
    return res


def be_bert_deal_by_sentecese(dataset, config: Config):
    outs = 0
    sents = cut_sent(dataset)
    for sent in sents:
        token = config.tokenizer.tokenize(sent)
        token = [CLS] + token
        seq_len = len(token)
        token_ids = config.tokenizer.convert_tokens_to_ids(token)
        mask = []
        if config.pad_size:
            if seq_len < config.pad_size:
                mask = [1] * len(token_ids) + [0] * (config.pad_size - len(token))
                mask = torch.LongTensor(mask).to(config.device)
                token_ids += ([0] * (config.pad_size - len(token)))
                token_ids = torch.LongTensor(token_ids).to(config.device)
                seq_len = torch.LongTensor(seq_len).to(config.device)
            else:
                mask = [1] * config.pad_size
                mask = torch.LongTensor(mask).to(config.device)
                token_ids = token_ids[:config.pad_size]
                token_ids = torch.LongTensor(token_ids).to(config.device)
                seq_len = torch.LongTensor(config.pad_size).to(config.device)
        # token_dict[md5(sent)] =
        res = config.bert_model(token_ids.unsqueeze(0), attention_mask=mask.unsqueeze(0))
        drop_data = nn.Dropout(config.dropout).to(config.device)
        out_bert = drop_data(res.get('last_hidden_state')).to(config.device)
        # contents.append((token_ids, label, seq_len, mask, out_bert))
        outs += out_bert
    out_1 = outs

    return out_1


def out_data_to_jsonl(datas, out_file_path, mode="w"):
    file_dir = os.path.dirname(out_file_path)
    os.makedirs(file_dir, exist_ok=True)
    with open(out_file_path, mode) as f:
        for e in datas:
            f.write(str(e) + "\n")
    print(f"write data len :{len(datas)} to file :{out_file_path}")


def batch_gen_dataiter_model_dict(models, news_datas: Optional[Dict]):
    # res_dict : {"data_iter" : [model_index]}
    res_dict = {
    }
    # sen_len_dataiter_dict : {"2" : data_iter,"3":data_iter}
    sen_len_dataiter_dict = {

    }
    data_iter_list = []
    for model_index, (config, model, _) in enumerate(models):
        if str(config.cut_sen_len) not in sen_len_dataiter_dict.keys():
            base_keywords: list = config.predict_base_keywords
            re_base_pattern = None
            if len(base_keywords) > 0:
                re_base_pattern = gen_pattern(base_keywords, expansion_num=0)
            sentences = []
            for query in news_datas:
                #  拆分成句子
                for text in cut_sentences(query, config):
                    # 如果是 predict 模式，则将所有的label都mask掉，因为预测模式下用不到 label
                    if len(str(text).strip()) < 1:
                        continue
                    # 进行一次关键词过滤，切记此模式不可用在训练中，只可以用在推理中，用于加速以及保证基础的P
                    if re_base_pattern is not None and regex.search(re_base_pattern, text) is None:
                        continue
                    sentences.append({"text": text, "label": "other", "news_id": query['news_id']})
            # 如果sentences 为空，进入DataLoader时会报错，所以添加一个空白的text
            if len(sentences) == 0:
                sentences.append({"text": "", "label": "other", "news_id": query['news_id']})
            if "BertAtt" in config.model_name:
                data_iter = build_iter_bertatt(sentences, config=config, is_predict=True)
            else:
                predict_dataset = build_dataset(config, sentences, is_predict=True)
                data_iter = DataLoader(dataset=predict_dataset,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       collate_fn=lambda x: dataset_collate_fn(config, x))
                sen_len_dataiter_dict[str(config.cut_sen_len)] = data_iter
        else:
            data_iter = sen_len_dataiter_dict[str(config.cut_sen_len)]

        if id(data_iter) not in res_dict.keys():
            res_dict[id(data_iter)] = [model_index]
        else:
            res_dict[id(data_iter)].append(model_index)
        data_iter_list.append(data_iter)
    sen_len_dataiter_dict.clear()
    return res_dict, data_iter_list
