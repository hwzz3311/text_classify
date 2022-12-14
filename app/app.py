import argparse
import importlib
import json
import logging
import os
import pickle as pkl
import time
from functools import wraps

import numpy as np
import shap

import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from importlib import import_module

from markupsafe import Markup

from src.models import FastText
from src.options import RunArgs
from src.processors import build_by_sentence, CLS
from src.utils.model_utils import set_seed, get_vocab, init_network, build_vocab
from src.utils.shap_plots_text import text as shap_plots_text

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)

args = RunArgs().get_parser()

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
model.to(config.device)
if config.model_name != "Transformer" and "bert" not in config.model_name.lower():
    init_network(model)

if args.check_point_path is None or len(args.check_point_path) == 0:
    args.check_point_path = config.save_path
assert os.path.exists(args.check_point_path), "check point file not find !"
config.save_path = args.check_point_path
if config.device.type == "cpu":
    model.load_state_dict(torch.load(config.save_path, map_location="cpu"))
    # model = model.module
else:
    model.load_state_dict(torch.load(config.save_path))
model.eval()
print('"bert" in config.model_name.lower() ： ', "bert" in config.model_name.lower())
logger.info(config.__dict__)
softmax_fun = torch.nn.Softmax(dim=1)


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


app = Flask(__name__)

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
             "result": {
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
        return render_template('base.html')
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
        return jsonify({"shap_result": s})

    # return send_from_directory(os.path.dirname(__file__), "save.html")


@app.route("/text_classify", methods=["POST"])
@log_filter
def text_classify_predict():
    query = request.get_json()
    content = query['content']
    threshold = query.get('threshold', 0.5)
    sentences = [content]
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
        # print(outputs.data)

        # sigmoid_output = torch.sigmoid(outputs)
        softmax_output = softmax_fun(outputs)
        # print(sigmoid_output.data.cpu())
        print(softmax_output.data.cpu())
        # sigmoid_output_list = sigmoid_output.data.cpu().numpy().tolist()
        softmax_output_list = softmax_output.data.cpu().numpy().tolist()
        outputs = softmax_output
        # max_index = np.argmax(outputs.cpu().numpy().tolist())
        # softmax_max_index = np.argmax(softmax_output.cpu().numpy().tolist())
        predict = torch.where(outputs > threshold, torch.ones_like(outputs), torch.zeros_like(outputs))
        # print(predict)
        predict_result = predict.cpu().numpy().tolist()
    print(predict_result)
    print(np.argmax(predict_result[0]))
    # print("sigmoid_max_index", config.class_list[max_index])
    # print("softmax_max_index", config.class_list[softmax_max_index])
    class_score = {config.class_list[i]: x for i, x in enumerate(softmax_output_list[0])}
    # softmax_class_score = {config.class_list[i]: x for i, x in enumerate(softmax_output_list[0])}

    return jsonify({"content": content,
                    "result": config.class_list[np.argmax(predict_result[0])],
                    # "sigmoid_result": config.class_list[max_index],
                    # "softmax_result": config.class_list[softmax_max_index],
                    "class_score": class_score,
                    # "sigmoid_class_score": class_score,
                    # "softmax_class_score": softmax_class_score
                    })


if __name__ == '__main__':
    # app.run(host="0.0.0.0",port="5005",debug=True)
    app.run(host="0.0.0.0", port="5000", debug=True)
