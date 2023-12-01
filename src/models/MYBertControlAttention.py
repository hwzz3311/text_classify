import argparse
import os

import torch.nn as nn
from transformers import BertModel, AutoModel, AutoTokenizer, AutoConfig, BertConfig

from src.models.config import BaseConfig
from src.models.custom_bert import MYBertModel, MYBertContinue
from src.utils.model_utils import split_model_layer
"""
此处主要实现为了解决
"""

mybert = None

class MyBertSingleton(object):
    __instance = None

    def __init__(self, bert_config: BertConfig, bert_layer_nums, bert_split_dir, device):
        print("mybert init")
        mybert = MYBertModel(bert_config, bert_layer_nums=bert_layer_nums, bert_split_dir=bert_split_dir)
        mybert.to(device)
        print(f"mybert to device: {device}")
        for param in mybert.parameters():
            param.requires_grad = False
        print("next(mybert.parameters()).is_cuda : ", next(mybert.parameters()).is_cuda)
        self.mybert = mybert

    def __new__(cls, *args, **kwargs):
        # TODO : 此处可以进再一步优化，由于目前只使用了一个bert的版本，所以可以放心的使用单例模式，但是后续可能会出现多个不同版本的bert的情况，所以后续可能需要进行修改
        if MyBertSingleton.__instance is None:
            MyBertSingleton.__instance = super().__new__(cls, *args, **kwargs)
        return MyBertSingleton.__instance

    def __call__(self, *args, **kwargs):
        # 将数据传入到mybert中进行预测
        return self.mybert(*args, **kwargs)

    def encoder_continue(self, *args, **kwargs):
        return self.mybert.encoder_continue(*args, **kwargs)


class Config(BaseConfig):
    """
    bert_type：
    nohup python -m src.run --model MYBert --bert_type nbroad/ESG-BERT --do_train true --do_predict_news True --data_dir assets/data/topic_en_greenwashing/ --gpu_ids 1 --num_epochs 40 --loss soft_bootstrapping_loss --batch_size 12 --bert_layer_nums 9 --save_model_name nbroad/ESG-BERT_9  > runoob.log 2>&1 &
    """

    def __init__(self, args: argparse.ArgumentParser):
        super(Config, self).__init__(args)
        if self.local_model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_dir)
            self.bert_config = AutoConfig.from_pretrained(self.bert_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
            self.bert_config = AutoConfig.from_pretrained(self.bert_type)
        self.hidden_size = 1024 if "large" in str(self.bert_dir).lower() \
                                   or "large" in str(self.bert_type).lower() else 768
        self.continue_layer_nums = self.bert_config.num_hidden_layers - self.bert_layer_nums
        # 需要先将模型保存到一个临时文件夹，再去做拆分的操作
        if self.bert_split_dir is None:
            self.bert_split_dir = os.path.join(os.path.dirname(__file__), f"../../assets/models/{self.bert_type}_split")
        if self.local_model:
            bin_file_path = os.path.join(self.bert_dir, "./pytorch_model.bin")
        else:
            bert: BertModel = AutoModel.from_pretrained(self.bert_type)
            bert_save_dir = os.path.join(os.path.dirname(__file__), f"../../assets/models/{self.bert_type}")
            bert.save_pretrained(bert_save_dir)
            bin_file_path = os.path.join(bert_save_dir, "./pytorch_model.bin")
            del bert
        if not os.path.exists(self.bert_split_dir) or len(os.listdir(self.bert_split_dir)) == 0:
            split_model_layer(bin_file_path, self.bert_split_dir)
        global mybert
        if mybert is None:
            mybert = MYBertModel(self.bert_config, bert_layer_nums=self.bert_layer_nums, bert_split_dir=self.bert_split_dir)
            mybert.to(self.device)
            print(f"mybert to device: {self.device}")
            for param in mybert.parameters():
                param.requires_grad = False
            mybert.eval()


class Model(nn.Module):

    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.bert = MYBertContinue(config.bert_config, user_config=config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.bert_layer_nums = config.bert_layer_nums
        assert mybert is not None

    def get_my_bert_res(self, x, bert_continue=False):
        if bert_continue:
            hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions, \
            extended_attention_mask, head_mask, encoder_hidden_states, \
            encoder_extended_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states = x
        else:
            context = x[0]  # 输入的句子
            mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
            x = mybert(context, attention_mask=mask, to_layer_num=self.bert_layer_nums)
            hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions, \
            extended_attention_mask, head_mask, encoder_hidden_states, \
            encoder_extended_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states = x

        out = self.bert(hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions,
                        all_cross_attentions, extended_attention_mask, head_mask, encoder_hidden_states,
                        encoder_extended_attention_mask, past_key_values, use_cache, output_attentions,
                        output_hidden_states)
        return out

    def forward(self, x, bert_continue=False):
        out = self.get_my_bert_res(x, bert_continue)
        # print(out.keys())
        # print(out[0].shape)
        out = self.dropout(out[1])
        out = self.fc(out)
        return out
