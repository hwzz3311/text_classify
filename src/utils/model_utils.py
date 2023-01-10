import ast
import importlib
import logging
import os
import pickle
import random
import time
from datetime import timedelta

import numpy as np
import shap
import torch
from torch import nn
from tqdm import tqdm

from src.compression.moefication.utils import bert_change_forward
from src.models import FastText
from src.models.config import BaseConfig

logger = logging.getLogger(name="flask_app")

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
MAX_VOCAB_SIZE = 10000
UNK = '<UNK>'


def explainer_init(config, model):
    softmax_fun = torch.nn.Softmax(dim=1)

    def shap_predict_fun(datas, device=config.device, tokenizer=config.tokenizer, pad_size=config.pad_size):
        datas_tokens = []
        for x in datas:

            token = tokenizer.tokenize(x)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = tokenizer.convert_tokens_to_ids(token)
            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            datas_tokens.append({
                "input_ids": token_ids,
                "seq_len": seq_len,
                "mask": mask
            })

        input_ids = torch.LongTensor([_["input_ids"] for _ in datas_tokens]).to(device)
        seq_len = torch.LongTensor([_["seq_len"] for _ in datas_tokens]).to(device)
        mask = torch.LongTensor([_["mask"] for _ in datas_tokens]).to(device)
        X = (input_ids, seq_len, mask)

        with torch.no_grad():
            outputs = model(X)
            _outputs = outputs.detach().cpu().numpy()
            scores = (np.exp(_outputs).T / np.exp(_outputs).sum(-1)).T
            try:
                predict_result = torch.max(outputs.data, dim=1)[1].cpu()
            except:
                outputs = torch.unsqueeze(outputs, 0)
                predict_result = torch.max(outputs.data, dim=1)[1].cpu()

            softmax_output = softmax_fun(outputs)

            # outputs = softmax_output
            # predict = torch.where(outputs > config.threshold, torch.ones_like(outputs),
            #                       torch.zeros_like(outputs))

            # print(sigmoid_output.data.cpu())
            # sigmoid_output_list = sigmoid_output.data.cpu().numpy().tolist()
            predict = softmax_output.detach().cpu().numpy()
            return predict

    if "topic_en_" in str(config.data_dir):
        masker = shap.maskers.Text(r" ")
    else:
        masker = shap.maskers.Text(r".")

    explainer = shap.Explainer(shap_predict_fun, masker, output_names=config.class_list)
    return explainer


def models_init(args):
    logger.info(args.__dict__)
    mode_name = args.model
    x: FastText = importlib.import_module(f"src.models.{mode_name}")
    config: FastText.Config = x.Config(args)

    # 设置随机种子
    set_seed(config)
    logger.info(f'Process info : {config.device} , gpu_ids : {config.gpu_ids}, model_name : {config.model_name}')

    # 加载模型
    vocab = get_vocab(config, use_word=False)
    config.n_vocab = len(vocab)

    model: FastText.Model = x.Model(config)

    if config.model_name != "Transformer" and "bert" not in config.model_name.lower():
        init_network(model)

    if args.check_point_path is None or len(args.check_point_path) == 0:
        args.check_point_path = config.save_path
    assert os.path.exists(args.check_point_path), "check point file not find !"
    config.save_path = args.check_point_path
    # 先统一加载到cpu，再转到gpu上
    model.load_state_dict(torch.load(config.save_path, map_location="cpu"))
    print("torch.load_state_dict loaded successfully for " + config.save_path)
    print(model)
    if config.MOE_model:
        bert_type = config.bert_type
        if "/" in config.bert_type:
            bert_type = config.bert_type.split("/")[1]
        param_split_dir = os.path.join(os.path.dirname(__file__), "../", config.data_dir,
                                       f"saved_dict/{config.model_name}/MOE/{bert_type}")
        assert os.path.exists(param_split_dir), "MOE model file not found"
        bert_change_forward(model, param_split_dir, config.device, 20)

    model.to(config.device)
    model.eval()
    print('"bert" in config.model_name.lower() ： ', "bert" in config.model_name.lower())
    logger.info(config.__dict__)
    return config, model





def predict_res_merger(predict_all_list, predict_result_score_all, news_ids_list, origin_text_all, config):
    predict_res_dict = {}
    for predict_res, result_score, news_id, text in zip(predict_all_list, predict_result_score_all, news_ids_list,
                                                        origin_text_all):

        if news_id not in predict_res_dict.keys():
            predict_res_dict[news_id] = {
                "predict_res": [predict_res],
                "texts": [text],
                "support_sentence": [text] if predict_res == config.class_list[1] else [],
                "result_score": [result_score] if predict_res == config.class_list[1] else [],
                "label": True if predict_res == config.class_list[1] else False
            }
        else:
            predict_res_dict[news_id]["predict_res"].append(predict_res)
            predict_res_dict[news_id]["texts"].append(text)
            if predict_res == config.class_list[1]:
                predict_res_dict[news_id]["support_sentence"].append(text)
                predict_res_dict[news_id]["result_score"].append(result_score)
                predict_res_dict[news_id]["label"] = True

    for news_id in predict_res_dict.keys():
        predict_res_dict[news_id].pop("predict_res")
        predict_res_dict[news_id].pop("texts")
    return predict_res_dict


def get_all_models() -> list:
    filedir = os.path.dirname(__file__)
    models_dir = os.path.join(filedir, "../models/")
    model_files = [file_name.replace(".py", "") for file_name in os.listdir(models_dir) if file_name[0].isupper()]
    return model_files


def get_bert_types():
    e = {
        "bert_type": {
            "bert": ["bert-base-chinese", "hfl/chinese-bert-wwm-ext", ],
            "robert": ["hfl/chinese-roberta-wwm-ext", ],
            "albert": ["albert-base-v2", "albert-base-v1"],
            "xlnet": ["hfl/chinese-xlnet-base", "hfl/chinese-xlnet-mid"],
            "electr": ["hfl/chinese-electra-180g-base-discriminator"],
            "distilbert": ["distilbert-base-uncased", "ASCCCCCCCC/distilbert-base-chinese-amazon_zh_20000", ],
            "ernie": ["nghuyong/ernie-gram-zh", "nghuyong/ernie-1.0"]
        }
    }
    return e


def set_seed(config: BaseConfig):
    """
    设置随机种子
    :param config:
    :return:
    """
    # random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    # TODO 检查是否将代码放在此处
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


def init_network(model: nn.Module, method="xavier", exclude="embedding", seed=123):
    """
    初始化网络参数
    :param model:
    :param method:
    :param exclude: 需要排除的层
    :param seed:
    :return:
    """
    for name, w in model.named_parameters():
        if exclude not in name:
            if "weight" in name:
                if method == "xavier":
                    nn.init.xavier_normal_(w)
                elif method == "kaiming":
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif "bias" in name:
                nn.init.constant_(w, 0)
            else:
                pass


def get_vocab(config: BaseConfig, use_word=False):
    if "bert" in str(config.model_name).lower():
        config.vocab = {}
        return {}
    if use_word:
        tokenizer = lambda x: x.split(" ")  # 空格分割
    else:
        tokenizer = lambda x: [y for y in x]  # char分类
    # if os.path.exists(config.vocab_path):
    #     vocab = pickle.load(open(config.vocab_path, "rb"))
    # else:
    vocab = build_vocab(config.train_file, tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    pickle.dump(vocab, open(config.vocab_path, "wb"))
    print(f"Vocab size:{len(vocab)}")
    config.vocab = vocab
    return vocab


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def load_and_cache_examples():
    pass
    # glue_convert_examples_to_features


def get_time_dif(start_time: float):
    time_dif = time.time() - start_time
    return timedelta(seconds=int(round(time_dif)))


def modelsize(model, input, type_size=4):
    """
    # 模型显存占用监测函数
    # model：输入的模型
    # input：实际中需要输入的Tensor变量
    # type_size 默认为 4 默认类型为 float32
    :param model:
    :param input:
    :param type_size:
    :return:
    """
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))


def split_model_layer(model_file_path, save_folder):
    # 此处有一个奇怪的bug 在mac 上 layer_name 是bert. 开头；而linux系统上却并没有bert.
    model = torch.load(model_file_path, map_location='cpu')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for file_index, layer_name in tqdm(enumerate(model.keys()), total=len(model.keys()), desc="split_model_layer ing"):
        ffn_weight = model[layer_name].numpy()
        filename = os.path.join(save_folder, f"{layer_name}")
        torch.save(ffn_weight, filename)
    print(f"model layer split save to {save_folder}.")


if __name__ == "__main__":
    pass
    # print(all_model_type())
