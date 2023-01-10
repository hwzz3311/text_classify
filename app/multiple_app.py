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
from flask import Flask, request, jsonify, render_template

from torch.utils.data import DataLoader

from src.options import RunArgs
from src.processors import build_by_sentence, be_bert_deal_by_sentecese, cut_sentences, dataset_collate_fn, \
    build_dataset, build_iter_bertatt, para_clear
from src.utils.data_utils import text_to_md5
from src.utils.model_utils import models_init, explainer_init, \
    predict_res_merger
from src.utils.shap_plots_text import text as shap_plots_text


def model_init_by_parm(parm):
    runargs = RunArgs()
    parser = runargs.parser()
    parser = runargs.initialize(parser)

    args = parser.parse_args(parm)
    config, model = models_init(args)
    explainer = explainer_init(config, model)
    return config, model, explainer


app = Flask(__name__)
gunicorn_logger = logging.getLogger("gunicorn.error")
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)
logger = app.logger

args_list = [
    ["--model", "MYBert",
     "--data_dir", "assets/data/topic_en_greenwashing/",
     "--gpu_ids", "1",
     "--bert_type", os.path.join(os.path.dirname(__file__), "../assets/models/nbroad_ESG-BERT"),
     "--batch_size", "1",
     "--check_point_path",
     os.path.join(os.path.dirname(__file__), "../assets/data/topic_en_greenwashing/saved_dict/MYBert/ESG-BERT_230106_9.cpkt"),
     "--threshold", "0.8",
     "--cut_sen_len", "2",
     "--bert_layer_nums", "9",
     "--bert_split_dir", os.path.join(os.path.dirname(__file__), "../assets/models/nbroad/ESG-BERT_split")],

    # ["--model", "MYBert",
    #  "--data_dir", "assets/data/topic_en_Customer_Privacy_Incidents/",
    #  "--gpu_ids", "0",
    #  "--bert_type", os.path.join(os.path.dirname(__file__), "../assets/models/nbroad_ESG-BERT"),
    #  "--batch_size", "1",
    #  "--check_point_path", os.path.join(os.path.dirname(__file__),
    #                                     "../assets/data/topic_en_Customer_Privacy_Incidents/saved_dict/MYBert/ESG-BERT_230106_10.cpkt"),
    #  "--threshold", "0.8",
    #  "--cut_sen_len", "2",
    #  "--bert_layer_nums", "10",
    #  "--bert_split_dir", os.path.join(os.path.dirname(__file__), "../assets/models/nbroad/ESG-BERT_split")],
    #
    # ["--model", "MYBert",
    #  "--data_dir", "assets/data/topic_en_Safety_Accidents_v1/",
    #  "--gpu_ids", "0",
    #  "--bert_type", os.path.join(os.path.dirname(__file__), "../assets/models/nbroad_ESG-BERT"),
    #  "--batch_size", "1",
    #  "--check_point_path", os.path.join(os.path.dirname(__file__),
    #                                     "../assets/data/topic_en_Safety_Accidents_v1/saved_dict/MYBert/ESG-BERT_230106_10.cpkt"),
    #  "--threshold", "0.8",
    #  "--cut_sen_len", "2",
    #  "--bert_layer_nums", "10",
    #  "--bert_split_dir", os.path.join(os.path.dirname(__file__), "../assets/models/nbroad/ESG-BERT_split")]
]

# TODO 先将 所有的参数列表 进行判断，并且按照 bert_layer_nums 进行从小到大进行排序，并记录下最大的bert_layer_nums，使用最大的bert_layer_nums 对mybert进行实例化


# 不同模型对应的 bert_layer_nums，key 为模型的index，bert_layer_nums  为模型的层数;
# mybert_model_layers_indexs :[(0,10)]
mybert_model_layers_indexs = []

for model_index, args in enumerate(args_list):
    for pam_index, arg in enumerate(args):
        if "bert_layer_nums" in arg:
            # 取出对应的参数
            bert_layer_nums = int(args[pam_index + 1])
            mybert_model_layers_indexs.append((model_index, bert_layer_nums))
            break
mybert_max_layer_nums = max([x[1] for x in mybert_model_layers_indexs])
# 进行排序, 按照最小的在最前面的方式进行排序
mybert_model_layers_indexs = sorted(mybert_model_layers_indexs, key=lambda x: x[1])
sorted_args_list = []
for args_index, bert_layer_nums in mybert_model_layers_indexs:
    sorted_args_list.append(args_list[args_index])

# 先取出 最后（bert_layer_nums最大）的参数对mybert进行初始化
last_parm = sorted_args_list[-1]
last_x = model_init_by_parm(last_parm)
# 再去初始化其他的参数模型
models = []
for parm in sorted_args_list[:-1]:
    config, model, explainer = model_init_by_parm(parm)
    models.append((config, model, explainer))
# 再将最后的模型添加进来
models.append(last_x)

softmax_fun = torch.nn.Softmax(dim=1)

topic_name_dict = {
    "Greenwashing": "漂绿指控_ENG",
    "Customer_Privacy_Incidents": "客户隐私事件_ENG",
    "Safety_Accidents": "安全事故_ENG",
}

# 使用字典缓存 输入news的预测结果
cache_topic_res = {}
cache_topic_max_len = 5000
cache_topic_md5s = []

# 将 import 放在后面 为了是在mybert被初始化之后再导入
from src.models.MYBert import mybert

mybert.eval()

for config, model, _ in models:
    model.eval()


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


# def _to_tensor(config, datas):
#     x = torch.LongTensor([_[0] for _ in datas]).to(config.device)
#     # pad 前的长度（超过pad_size的设为pad_size）
#     seq_len = torch.LongTensor([_[2] for _ in datas]).to(config.device)
#     # print(f"x:{x.shape}, seq_len:{seq_len.shape},y:{y.shape}")
#     if config.model_name == "FastText":
#         bigram = torch.LongTensor([_[3] for _ in datas]).to(config.device)
#         trigram = torch.LongTensor([_[4] for _ in datas]).to(config.device)
#         return x, seq_len, bigram, trigram
#     elif "bert" in config.model_name.lower():
#         mask = torch.LongTensor([_[3] for _ in datas]).to(config.device)
#         # print(f"mask :{mask.shape}")
#         return x, seq_len, mask
#     return x, seq_len


# 临时关闭
@app.route("/shap_analysis", methods=["POST", "GET"])
@log_filter
def shap_analysis():
    if request.method == 'GET':
        select_models = [config.data_dir for config, model, explainer in models]
        return render_template('base.html', models=select_models)
    if request.method == 'POST':
        query = None
        if query is None:
            query = json.loads(request.get_data())
        if query is None:
            query = request.form.to_dict()
        select_model = query["select_model"]
        # 由于分词的时候会莫名的删除一个字符，所以这里就添加一个空白字符
        content = str(query['content'])
        content = para_clear(content)
        content = content.replace("\\n", " ")
        content += " "
        # TODO :base html中添加模型选择，一次只分析一个模型
        for config, model, explainer in models:
            if config.data_dir != select_model:
                continue
            shap_values = explainer([content])
            s = shap_plots_text(shap_values)
        # with open("./save.html", "w") as f:
        #     f.write(s)
        return jsonify({"select_model": select_model, "shap_result": s})

    # return send_from_directory(os.path.dirname(__file__), "save.html")


#
# 临时关闭
# @app.route("/text_classify", methods=["POST"])
# @log_filter
# def text_classify_predict():
#     query = request.get_json()
#     content = query['content']
#     threshold = query.get('threshold', 0.5)
#     sentences = [content]
#
#     if "BertAtt" in config.model_name:
#         test_data = be_bert_deal_by_sentecese(content, config=config)
#     else:
#         test_data = build_by_sentence(config, sentences, config.vocab, 0, config.pad_size)
#         test_data = _to_tensor(config, test_data)
#     sigmoid_output_list = [-1, -1]
#     with torch.no_grad():
#         outputs = model(test_data)
#         try:
#             predict_result = torch.max(outputs.data, dim=1)[1].cpu()
#         except:
#             outputs = torch.unsqueeze(outputs, 0)
#             predict_result = torch.max(outputs.data, dim=1)[1].cpu()
#
#         softmax_output = softmax_fun(outputs)
#         softmax_output_list = softmax_output.data.cpu().numpy().tolist()
#         outputs = softmax_output
#         predict = torch.where(outputs > threshold, torch.ones_like(outputs), torch.zeros_like(outputs))
#         predict_result = predict.cpu().numpy().tolist()
#     class_score = {config.class_list[i]: x for i, x in enumerate(softmax_output_list[0])}
#
#     return jsonify({"content": content,
#                     "result": config.class_list[np.argmax(predict_result[0])],
#                     "class_score": class_score,
#                     })
#

def gen_dataiter_model_dict(models, query):
    # res_dict : {"data_iter" : [model_index]}
    res_dict = {
    }
    # sen_len_dataiter_dict : {"2" : data_iter,"3":data_iter}
    sen_len_dataiter_dict = {

    }
    data_iter_list = []
    for model_index, (config, model, _) in enumerate(models):
        if str(config.cut_sen_len) not in sen_len_dataiter_dict.keys():
            sentences = []
            #  拆分成句子
            for text in cut_sentences(query, config):
                # 如果是 predict 模式，则将所有的label都mask掉，因为预测模式下用不到 label
                if len(str(text).strip()) < 1:
                    continue
                sentences.append({"text": text, "label": "other", "news_id": query['news_id']})
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


@app.route("/", methods=["POST"])
@log_filter
def topic():
    query = request.get_json()

    try:
        title = query.get('title')
        content = query.get('content')
        text_md5 = text_to_md5(title + content)
        if text_md5 in cache_topic_res.keys():
            res = cache_topic_res[text_md5]
        else:
            res = models_predict(query, text_md5, models)
        topic_res = {
            "status_code": 1,
            "result": res["result"]["topics"]
        }
        return jsonify(topic_res)
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"status": 0, "error": str(e), "result": []})


@app.route("/mains", methods=["POST"])
@log_filter
def mains():
    query = request.get_json()
    try:
        title = query.get('title')
        content = query.get('content')
        labels = query.get('labels')
        text_md5 = text_to_md5(title + content)
        if text_md5 in cache_topic_res.keys():
            models_res = cache_topic_res[text_md5]
        else:
            models_res = models_predict(query, text_md5, models)
        models_mains_res = models_res["result"]["mains"]
        # 由于 之前状态码 不规范，导致不同位置的状态码判断不一致，将错就错吧
        mains_res = {
            "detail": {},
            "status": 0
        }
        for label in labels:
            if label in models_mains_res.keys():
                mains_res["detail"][label] = models_mains_res[label]
        return jsonify(mains_res)
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"status": 1, "error": str(e), "detail": {}})


def models_predict(query, text_md5, models):
    news_id = query['news_id']
    res = {
        "news_id": news_id,
        "result": {
            "topics": [],
            "mains": {}
        }
    }
    topic_mains_sentence_dict = {
        config.class_list[1]: [] for (config, model, _) in models
    }

    # 通过不同的 data_iter 划分出不同的model；同一个data_iter 对应多个model
    dataiter_model_dict, data_iter_list = gen_dataiter_model_dict(models, query)
    app.logger.info(f"dataiter_model_dict : {dataiter_model_dict}")
    # 遍历不同的data_iter
    for data_iter_id, model_index_list in dataiter_model_dict.items():
        # for 选择对应的 data_iter
        data_iter = []
        for data_iter_cls in data_iter_list:
            if id(data_iter_cls) == data_iter_id:
                data_iter = data_iter_cls
                break

        for i, _data in enumerate(data_iter):
            # 当切换 data 就将上层的 中间参数都 清空一下; 同一批数据 的 参数都是可以通用的；
            last_layer_num = -1
            hidden_states = None
            next_decoder_cache = None
            all_hidden_states = None
            all_self_attentions = None
            all_cross_attentions = None
            extended_attention_mask = None
            head_mask = None
            encoder_hidden_states = None
            encoder_extended_attention_mask = None
            past_key_values = None
            use_cache = None
            output_attentions = None
            output_hidden_states = None
            for model_index in model_index_list:
                news_ids_all = []
                origin_text_all = []
                predict_result_all = []
                predict_result_score_all = []

                config, model, _ = models[model_index]
                sigmoid_output_list = [-1, -1]
                with torch.no_grad():
                    trains, labels, news_ids,origin_text = _data[0], _data[1], _data[2], _data[3]
                    input_tokens = trains[0]
                    # 先将input_tokens、 送到 mybert中
                    mask = trains[2]
                    if last_layer_num != config.bert_layer_nums:
                        if hidden_states is None:
                            x = mybert(input_tokens, attention_mask=mask, to_layer_num=config.bert_layer_nums)
                        else:
                            from_layer_num = last_layer_num
                            to_layer_num = config.bert_layer_nums
                            # TODO from_layer_num、to_layer_num 的这个数字可能有问题，由于目前没有调用到这个fun，就没办法调试，
                            x = mybert.encoder_continue(
                                hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions,
                                all_cross_attentions,
                                from_layer_num,
                                to_layer_num,
                                extended_attention_mask, head_mask, encoder_hidden_states,
                                encoder_extended_attention_mask, past_key_values, use_cache, output_attentions,
                                output_hidden_states
                            )
                        hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions, \
                        extended_attention_mask, head_mask, encoder_hidden_states, \
                        encoder_extended_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states = x
                        last_layer_num = config.bert_layer_nums
                    outputs = model(x, bert_continue=True)
                    try:
                        predict_result = torch.max(outputs.data, dim=1)[1].cpu()
                    except:
                        outputs = torch.unsqueeze(outputs, 0)
                        predict_result = torch.max(outputs.data, dim=1)[1].cpu()

                    softmax_output = softmax_fun(outputs)
                    outputs = softmax_output
                    predict = torch.where(outputs > config.threshold, torch.ones_like(outputs),
                                          torch.zeros_like(outputs))
                    predict_results = []
                    predict_result_score = []
                    softmax_output_list = softmax_output.data.cpu().numpy().tolist()
                    predict_list = predict.cpu().numpy().tolist()
                    for predict_result, softmax_output in zip(predict_list, softmax_output_list):
                        predict_results.append(config.class_list[np.argmax(predict_result)])
                        predict_result_score.append(softmax_output[np.argmax(predict_result)])
                    news_ids_all.extend(news_ids)
                    # 保存token id，在后续的结果中进行还原
                    origin_text_all.append(origin_text)
                    predict_result_all.extend(predict_results)
                    predict_result_score_all.extend(predict_result_score)
                    # app.logger.info(f"predict_result_all : {predict_result_all}\n predict_result_score_all : {predict_result_score_all}\n, news_ids_all : {news_ids_all}")
                    predict_res_dict = predict_res_merger(predict_result_all, predict_result_score_all, news_ids_all,
                                                          origin_text_all, config)
                    # app.logger.info(f"predict_res_dict : {predict_res_dict}")
                    if predict_res_dict[news_id]["label"]:
                        res["result"]["topics"].append(topic_name_dict[config.class_list[1]])

                        for sen, score in zip(predict_res_dict[news_id]["support_sentence"],
                                              predict_res_dict[news_id]["result_score"]):
                            topic_mains_sentence_dict[config.class_list[1]].append({
                                "probability": score,
                                "text": sen
                            })
    res["result"]["topics"] = list(set(res["result"]["topics"]))
    for k, v in topic_mains_sentence_dict.items():
        res["result"]["mains"][topic_name_dict[k]] = v

    # if len(cache_topic_md5s) > cache_topic_max_len:
    #     cache_topic_res.pop(cache_topic_md5s.pop(0))
    # cache_topic_res[text_md5] = res.copy()
    # cache_topic_md5s.append(text_md5)
    return res


if __name__ == '__main__':
    # app.run(host="0.0.0.0",port="5005",debug=True)
    app.run(host="0.0.0.0", port="5005", debug=False)
