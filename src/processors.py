import ast
import json
import os.path
import random
import re
from copy import deepcopy
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
from src.utils.data_utils import gen_pattern, para_sentences
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
    def __init__(self, config: BaseConfig, dataset, is_predict=False, att=False):
        self.config = config
        self.pad_size = config.pad_size
        self.is_predict = is_predict
        self.att = att
        self.original_dataset = dataset

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
        if att:
            # 当使用att时动态的加载不同的数据处理器
            self.data = self.gen_att_datas()
            # self.__getitem__ == self.__att__getitem__
            self.bert_model = config.bert_model
            self.dropout = config.dropout

    def biGramHash(self, sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(self, sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def gen_att_datas(self):
        self.att_datas = []
        # 这里添加对原始数据的处理办法
        # 重采样等等
        for json_data in self.original_dataset:
            title_content_sep = ". "
            topic_name = self.config.class_list[1]
            try:
                # 此时取到的是
                title, content = json_data["title"], json_data["content"]
            except KeyError:
                print(json_data)
            if self.is_predict or "news_id" in json_data.keys():
                news_id = json_data["news_id"]
            else:
                news_id = ""
            content = " " * len(news_id) + " " * 9 + title + " " * 9 + content
            title_len = len(title)
            offset = len('***$@$***') * 2 - len(title_content_sep) + len(news_id)
            offset = 0
            if topic_name in json_data.keys():
                for loc in json_data[topic_name]:
                    start, end = loc
                    if start >= title_len:
                        start -= offset
                        end -= offset
                    label_text = content[start: end]

                    self.att_datas.append({
                        "text": label_text,
                        "label": topic_name,
                        "news_id": news_id,
                    })
                    if end != len(content):
                        start_text = content[start:]
                        self.att_datas.append({
                            "text": start_text,
                            "label": topic_name,
                            "news_id": news_id,
                        })

                    if start != 0:
                        end_text = content[:end]
                        self.att_datas.append({
                            "text": end_text,
                            "label": topic_name,
                            "news_id": news_id,
                        })

                    if start != 0 and end != len(content):
                        before_text = content[:start]
                        after_text = content[end:]
                        curr_text = before_text + label_text + after_text
                        self.att_datas.append({
                            "text": curr_text,
                            "label": topic_name,
                            "news_id": news_id,
                        })
            # 加入不属于当前topic的数据作为负样本
            # for k, v in json_data.items():
            #     for start, end in json_data[k]:
            #         if start >= title_len:
            #             start -= offset
            #             end -= offset
            #         other_label_text = content[start:end]
            #         self.att_datas.append({
            #             "text": other_label_text,
            #             "label": self.config.class_list[0],
            #             "news_id": news_id,
            #         })
            else:
                # 随机从负样本中抽取 N个句子
                content_sents = para_sentences(content)
                if title:
                    content_sents.append(title)
                sampled_sents = random.sample(content_sents, min(8, len(content_sents)))
                for s in sampled_sents:
                    if s.strip():
                        self.att_datas.append({
                            "text": s.strip(),
                            "label": self.config.class_list[0],
                            "news_id": news_id,
                        })
                # if random.uniform(0.0, 1.0) > 0.5:
        return self.att_datas

    def __att__getitem__(self, index):
        att_data: dict = self.data[index]
        label = att_data["label"]
        text = att_data["text"]
        news_id = att_data["news_id"]
        max_length = self.config.max_seq_len
        token = self.config.tokenizer.tokenize(text)
        count = 0
        last_hidden_states = []
        all_token_ids = []
        all_seq_len = []
        all_mask = []
        seq_len_count = 0
        while len(token) > max_length or count < 1:
            curr_token = [CLS] + token[:max_length]

            seq_len = len(curr_token)
            mask = []
            token_ids = self.config.tokenizer.convert_tokens_to_ids(curr_token)
            if self.pad_size:
                if len(curr_token) < self.pad_size:
                    mask = [1] * len(token_ids) + [0] * (self.pad_size - len(curr_token))
                    token_ids += ([0] * (self.pad_size - len(curr_token)))
                else:
                    mask = [1] * self.pad_size
                    token_ids = token_ids[:self.pad_size]
                    seq_len = self.pad_size

            # token_ids = torch.LongTensor(token_ids).to(self.config.device)
            # mask = torch.LongTensor(mask).to(self.config.device)
            all_token_ids.append(token_ids)
            all_seq_len.append(seq_len)
            all_mask.append(mask)

            # res = self.bert_model(token_ids.unsqueeze(0), attention_mask=mask.unsqueeze(0))
            # drop_data = nn.Dropout(self.dropout).to(self.config.device)
            # out_bert = drop_data(res.get('last_hidden_state')).to(self.config.device)
            # last_hidden_states.append(res.get('last_hidden_state'))
            token = token[max_length:]
            count += 1
        # last_hidden_states = torch.cat(last_hidden_states, dim=1)

        # return last_hidden_states, self.config.class_list.index(label), None, None, news_id, text
        return (all_token_ids, self.config.class_list.index(label), all_seq_len, all_mask, news_id, text)

    def __normal__getitem__(self, index):
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

    def __getitem__(self, index):
        if self.att:
            return self.__att__getitem__(index)
        else:
            return self.__normal__getitem__(index)

    def __len__(self):
        return len(self.data)


def dataset_collate_fn(config: BaseConfig, datas: List):
    # if config.att:
    #     x = torch.LongTensor([_[0] for _ in datas]).to(config.device)
    #     seq_len = [None for _ in datas]
    #     mask = [None for _ in datas]
    #     y = torch.LongTensor([_[1] for _ in datas]).to(config.device)
    #     news_ids = [_[4] for _ in datas]
    #     origin_data = [_[5] for _ in datas]
    #     return (x, seq_len, mask), y, news_ids, origin_data
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


def batch_out_predict_dataset(predict_all_list, predict_result_score_all, news_ids_list, origin_text_all,
                              config: BaseConfig):
    assert len(predict_all_list) == len(news_ids_list) == len(
        origin_text_all), "predict_all_list not equal length input_tokens_all"

    predict_res_dict = predict_res_merger(predict_all_list, predict_result_score_all, news_ids_list, origin_text_all,
                                          config)
    return predict_res_dict


def out_predict_dataset(predict_all_list, predict_result_score_all, news_ids_list, origin_text_all,
                        config: BaseConfig):
    assert len(predict_all_list) == len(news_ids_list) == len(
        origin_text_all), "predict_all_list not equal length input_tokens_all"
    bert_type = config.bert_type
    if "/" in bert_type:
        bert_type = config.bert_type.split("/")[1]
    out_file_name = f"{config.model_name}_threshold{config.threshold}_predict_result.txt" \
        if "bert" not in config.model_name.lower() \
        else f"{config.model_name}_{bert_type}_threshold{config.threshold}_predict_result.txt"
    if not os.path.exists(config.predict_out_dir):
        os.makedirs(config.predict_out_dir, exist_ok=True)
    if config.predict_out_file is None:
        predict_out_file = os.path.join(config.data_dir, out_file_name)
    else:
        os.makedirs(os.path.dirname(config.predict_out_file), exist_ok=True)
        predict_out_file = config.predict_out_file
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


def out_test_dataset(predict_all_list, labels_all_list, original_text_all, softmax_score_all, config: BaseConfig):
    predict_results_dict = {}
    predict_results_list = []
    for p, t, text, softmax_score in zip(predict_all_list, labels_all_list, original_text_all, softmax_score_all):
        e = {"predict": p, "label": t, "text": text}
        for i, score in enumerate(softmax_score):
            e[f"score_{i}"] = score
        predict_results_dict[text] = e
        predict_results_list.append(e)
    bert_type = config.bert_type
    if "/" in bert_type:
        bert_type = config.bert_type.split("/")[1]
    out_file_name = f"{config.model_name}_{config.threshold}_predict_result.txt" \
        if "bert" not in config.model_name.lower() \
        else f"{config.model_name}_{bert_type}_{config.threshold}_predict_result.txt"
    out_csv_file_name = os.path.splitext(out_file_name)[0] + ".csv"
    predict_out_file = os.path.join(config.data_dir, out_file_name)
    predict_out_csv_file = os.path.join(config.data_dir, out_csv_file_name)
    predict_out_file = open(predict_out_file, "w")
    with open(config.test_file, "r") as f:
        for line in f:
            line = ast.literal_eval(line)
            text = line["text"]
            predict_results = predict_results_dict[text]
            predict_out_file.write(str(predict_results) + "\n")
    predict_out_file.close()
    pd.DataFrame(predict_results_list).to_csv(predict_out_csv_file, index=False)
    print(f"predict_out_file path: {predict_out_file}")
    print(f"predict_out_csv_file path: {predict_out_csv_file}")


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
            sents = para_sentences(context)  # 对句子进行分句
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
    sents = para_sentences(dataset)
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
