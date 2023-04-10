import argparse
import importlib
import json
import logging
import os
import pickle as pkl
import time
import uuid
from functools import wraps

import numpy as np
import shap

import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from importlib import import_module

from flask_cors import CORS
from flask_socketio import SocketIO
from markupsafe import Markup
from torch.utils.data import DataLoader

# from batch_predict_job import batch_gen_dataiter_model_dict
from src.compression.moefication.utils import bert_change_forward
from src.models import FastText
from src.options import RunArgs
from src.processors import build_by_sentence, CLS, be_bert_deal_by_sentecese, build_iter_bertatt, build_dataset, \
    dataset_collate_fn
from src.train_eval import predict_batch
from src.utils.data_utils import para_sentences, pre_cut_sentences, pre_cut_by_sentences
from src.utils.model_utils import set_seed, get_vocab, init_network, build_vocab
from src.utils.shap_plots_text import text as shap_plots_text

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)

args = RunArgs().get_parser()
# runargs = RunArgs()
# parser = runargs.parser()
# parser = runargs.initialize(parser)
#
# args = parser.parse_args(
#     ["--model", "Bert", "--data_dir", "assets/data/topic_en_greenwashing/", "--gpu_ids", "1", "--bert_type",
#      "nbroad/ESG-BER"])

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
if config.device.type == "cpu":
    model.load_state_dict(torch.load(config.save_path, map_location="cpu"))
else:
    model.load_state_dict(torch.load(config.save_path))
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
softmax_fun = torch.nn.Softmax(dim=1)

topic_name_dict = {
    "Greenwashing": "Greenwashing",
    "Customer_Privacy_Incidents": "Customer_Privacy_Incidents",
    "Safety_Accidents": "Safety_Accidents",
    "other": "其他"
}

# 使用字典缓存 输入news的预测结果
cache_topic_res = {}
cache_topic_max_len = 5000
cache_topic_md5s = []

app = Flask(__name__)
CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins='*')


def log_filter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = 1000 * time.time()
        app.logger.info(f"=============  Begin: {func.__name__}  =============")
        app.logger.info(f"Args: {kwargs}")
        try:
            rsp = func(*args, **kwargs)
            app.logger.info(f"Response: {rsp}")
            end = 1000 * time.time()
            app.logger.info(f"Time consuming: {end - start}ms")
            app.logger.info(f"=============   End: {func.__name__}   =============\n")
            return rsp
        except Exception as e:
            app.logger.error(repr(e))
            raise e

    return wrapper


def _to_tensor(config, datas):
    x = torch.LongTensor([_[0] for _ in datas]).to(config.device)
    # pad 前的长度（超过pad_size的设为pad_size）
    seq_len = torch.LongTensor([_[2] for _ in datas]).to(config.device)
    # print(f"x:{x.shape}, seq_len:{seq_len.shape},y:{y.shape}")
    if config.model_name == "FastText":
        bigram = torch.LongTensor([_[3] for _ in datas]).to(config.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(config.device)
        return x, seq_len, bigram, trigram
    elif "bert" in config.model_name.lower():
        mask = torch.LongTensor([_[3] for _ in datas]).to(config.device)
        # print(f"mask :{mask.shape}")
        return x, seq_len, mask
    return x, seq_len


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
        # print(sigmoid_output.data.cpu())
        # sigmoid_output_list = sigmoid_output.data.cpu().numpy().tolist()
        softmax_output = softmax_output.detach().cpu().numpy()
        return softmax_output



if "topic_en_" in str(config.data_dir):
    masker = shap.maskers.Text(r" ")
else:
    masker = shap.maskers.Text(r".")

explainer = shap.Explainer(shap_predict_fun, masker, output_names=config.class_list)


@app.route("/", methods=["POST", "GET"])
@log_filter
def hi():
    return jsonify({
        [
            {"api_name": "text_classify",
             "parameter": {
                 "content": "str"
             },
             "MOE": {
                 "type": "json",
                 "parameter": {
                     "content": "str",
                     "predict_result": "list"
                 }
             }}
        ]
    })


@app.route("/shap_analysis", methods=["POST", "GET"])
@log_filter
def shap_analysis():
    if request.method == 'GET':
        return render_template('base.html', models=[])
    if request.method == 'POST':
        query = None
        if query is None:
            query = json.loads(request.get_data())

        if query is None:
            query = request.form.to_dict()
        # 由于分词的时候会莫名的删除一个字符，所以这里就添加一个空白字符
        content = query['content'] + " "
        shap_values = explainer([content])
        s = shap_plots_text(shap_values)
        # with open("./save.html", "w") as f:
        #     f.write(s)
        return jsonify({"select_model": config.data_dir, "shap_result": s})

    # return send_from_directory(os.path.dirname(__file__), "save.html")


@app.route("/text_classify", methods=["POST"])
@log_filter
def text_classify_predict():
    query = request.get_json()
    content = query['content']
    threshold = query.get('threshold', 0.7)
    sentences = [content]

    if "BertAtt" in config.model_name:
        test_data = be_bert_deal_by_sentecese(content, config=config)
    else:
        test_data = build_by_sentence(config, sentences, vocab, 0, config.pad_size)
        test_data = _to_tensor(config, test_data)
    sigmoid_output_list = [-1, -1]
    with torch.no_grad():
        outputs = model(test_data)
        try:
            predict_result = torch.max(outputs.data, dim=1)[1].cpu()
        except:
            outputs = torch.unsqueeze(outputs, 0)
            predict_result = torch.max(outputs.data, dim=1)[1].cpu()

        softmax_output = softmax_fun(outputs)
        print(softmax_output.data.cpu())
        softmax_output_list = softmax_output.data.cpu().numpy().tolist()
        outputs = softmax_output
        predict = torch.where(outputs > threshold, torch.ones_like(outputs), torch.zeros_like(outputs))
        predict_result = predict.cpu().numpy().tolist()
    print(predict_result)
    print(np.argmax(predict_result[0]))
    class_score = {config.class_list[i]: x for i, x in enumerate(softmax_output_list[0])}

    return jsonify({"content": content,
                    "result": config.class_list[np.argmax(predict_result[0])],
                    "class_score": class_score,
                    })


@app.route("/auto_annotation", methods=["POST"])
@log_filter
def auto_annotation():
    query = request.get_json()
    text = query['text']
    sentences = query['text_list']
    threshold = query.get('threshold', 0.7)
    content_split = str(text).split("***$@$***")
    if len(content_split) == 3:
        news_id, title, content = content_split
    else:
        news_id = uuid.uuid1().hex
        title = ""
        content = content_split[0]
    news_datas = [
        {
            "news_id": news_id,
            "title": title,
            "content": content
        }
    ]
    # para = text
    # sentences: list = para_sentences(para)
    dataset = pre_cut_by_sentences(news_id, sentences, config)
    if str(config.model_name).startswith("BertAtt"):
        predict_iterator = build_iter_bertatt(dataset, config=config, is_predict=True)
    else:
        predict_dataset = build_dataset(config, dataset, is_predict=True)
        predict_iterator = DataLoader(dataset=predict_dataset,
                                      batch_size=config.batch_size,
                                      collate_fn=lambda x: dataset_collate_fn(config, x))
    models_predict_res = predict_batch(config, model, predict_iterator, news_datas)
    for news_id in models_predict_res.keys():
        predict_topics = models_predict_res[news_id]["result"]["topics"]

        models_mains_res = models_predict_res[news_id]["result"]["mains"]

        print(f"predict_topics : {predict_topics}")
        print(f"models_mains_res : {models_mains_res}")
    response_dict = {}
    result = models_predict_res[news_id]["result"]
    mains = result["mains"]
    res_sentence_indexs = []
    print(f"sentences : {sentences}")
    sentences = [str(x).strip() for x in sentences]
    print(f"sentences : {sentences}")
    if config.class_list[1] in mains:
        for sup_text_dict in mains[config.class_list[1]]:
            text = sup_text_dict["text"]
            text_seq = para_sentences(text, clear=False)
            v = text_seq[0].strip()
            sentence_index = sentences.index(v)
            if sentence_index > -1:
                for i in range(sentence_index, sentence_index + len(text_seq)):
                    res_sentence_indexs.append(i)
    response_dict["res_sentence_indexs"] = list(set(res_sentence_indexs))
    response_dict["topic_res"] = config.class_list[1] in result["topics"]
    response_dict["sentences"] = sentences
    return jsonify(response_dict)


@app.route("/train_model", methods=["POST"])
@log_filter
def train_model():
    # TODO
    query = request.get_json()
    text = query['text']
    sentences = query['text_list']
    threshold = query.get('threshold', 0.7)
    content_split = str(text).split("***$@$***")
    if len(content_split) == 3:
        news_id, title, content = content_split
    else:
        news_id = uuid.uuid1().hex
        title = ""
        content = content_split[0]
    news_datas = [
        {
            "news_id": news_id,
            "title": title,
            "content": content
        }
    ]
    # para = text
    # sentences: list = para_sentences(para)
    dataset = pre_cut_by_sentences(news_id, sentences, config)
    if str(config.model_name).startswith("BertAtt"):
        predict_iterator = build_iter_bertatt(dataset, config=config, is_predict=True)
    else:
        predict_dataset = build_dataset(config, dataset, is_predict=True)
        predict_iterator = DataLoader(dataset=predict_dataset,
                                      batch_size=config.batch_size,
                                      collate_fn=lambda x: dataset_collate_fn(config, x))


    models_predict_res = predict_batch(config, model, predict_iterator, news_datas)
    for news_id in models_predict_res.keys():
        predict_topics = models_predict_res[news_id]["result"]["topics"]

        models_mains_res = models_predict_res[news_id]["result"]["mains"]

        print(f"predict_topics : {predict_topics}")
        print(f"models_mains_res : {models_mains_res}")
    response_dict = {}
    result = models_predict_res[news_id]["result"]
    mains = result["mains"]
    res_sentence_indexs = []
    print(f"sentences : {sentences}")
    # sentences = [para_sentences(x) for x in sentences]
    if config.class_list[1] in mains:
        for sup_text_dict in mains[config.class_list[1]]:
            text = sup_text_dict["text"]
            text_seq = para_sentences(text, clear=False)
            v = text_seq[0]
            sentence_index = sentences.index(v)
            if sentence_index > -1:
                for i in range(sentence_index, sentence_index + len(text_seq)):
                    res_sentence_indexs.append(i)
    response_dict["res_sentence_indexs"] = list(set(res_sentence_indexs))
    response_dict["topic_res"] = config.class_list[1] in result["topics"]
    response_dict["sentences"] = sentences
    return jsonify(response_dict)


if __name__ == '__main__':
    # app.run(host="0.0.0.0",port="5005",debug=True)
    app.run(host="0.0.0.0", port="5006", debug=False)
