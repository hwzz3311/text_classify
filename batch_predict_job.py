"""
主要是实现 BI 需要批量跑数据的需求
"""

import argparse
import os
from copy import deepcopy
from typing import Optional, Dict

import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.options import RunArgs
from src.processors import cut_sentences, dataset_collate_fn, \
    build_dataset, build_iter_bertatt
from src.utils.data_utils import load_jsonl_file
from src.utils.model_utils import models_init, explainer_init, \
    predict_res_merger


def model_init_by_parm(parm):
    runargs = RunArgs()
    parser = runargs.parser()
    parser = runargs.initialize(parser)

    args = parser.parse_args(parm)
    config, model = models_init(args)
    explainer = explainer_init(config, model)
    return config, model, explainer



models = []
mybert = None


def init_models(args_list):
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
    global models
    models = []
    for parm in sorted_args_list[:-1]:
        config, model, explainer = model_init_by_parm(parm)
        models.append((config, model, explainer))
    # 再将最后的模型添加进来
    models.append(last_x)

    # 将 import 放在后面 为了是在mybert被初始化之后再导入
    from src.models.MYBert import mybert as MyBert
    global mybert
    mybert = MyBert


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
            sentences = []
            for query in news_datas:
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


def batch_models_predict(news_datas: list, models):
    tmp_dict = {
        "result": {
            "topics": [],
            "mains": {}}
    }
    res = {
        f"{e['news_id']}": deepcopy(tmp_dict) for e in news_datas
    }
    tmp_dict = {
        config.class_list[1]: [] for (config, model, _) in models
    }

    topic_mains_sentence_dict = {
        f"{e['news_id']}": deepcopy(tmp_dict) for e in news_datas
    }

    # 通过不同的 data_iter 划分出不同的model；同一个data_iter 对应多个model
    dataiter_model_dict, data_iter_list = batch_gen_dataiter_model_dict(models, news_datas)
    print(dataiter_model_dict)
    print({k: [models[i][0].data_dir for i in v] for k, v in dataiter_model_dict.items()})
    # 遍历不同的data_iter
    for data_iter_id, model_index_list in dataiter_model_dict.items():
        # for 选择对应的 data_iter
        data_iter = []
        for data_iter_cls in data_iter_list:
            if id(data_iter_cls) == data_iter_id:
                data_iter = data_iter_cls
                break

        for i, _data in tqdm(enumerate(data_iter), total=len(data_iter), desc="batch_models_predict ing"):
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
                input_tokens_all = []
                predict_result_all = []
                predict_result_score_all = []

                config, model, _ = models[model_index]
                sigmoid_output_list = [-1, -1]
                with torch.no_grad():
                    trains, labels, news_ids = _data[0], _data[1], _data[2]
                    if str(config.model_name).startswith("BertAtt"):
                        input_tokens = _data[3]
                    else:
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
                    if str(config.model_name).startswith("BertAtt"):
                        input_tokens_all.extend(input_tokens)
                    else:
                        input_tokens_all.extend(input_tokens.data.cpu().numpy().tolist())
                    predict_result_all.extend(predict_results)
                    predict_result_score_all.extend(predict_result_score)
                    predict_res_dict = predict_res_merger(predict_result_all, predict_result_score_all, news_ids_all,
                                                          input_tokens_all, config)
                    for news_id in predict_res_dict.keys():
                        if predict_res_dict[news_id]["label"]:
                            res[news_id]["result"]["topics"].append(topic_name_dict[config.class_list[1]])

                            for sen, score in zip(predict_res_dict[news_id]["support_sentence"],
                                                  predict_res_dict[news_id]["result_score"]):
                                topic_mains_sentence_dict[news_id][config.class_list[1]].append({
                                    "probability": score,
                                    "text": sen
                                })
    for news_id in res.keys():
        res[news_id]["result"]["topics"] = list(set(res[news_id]["result"]["topics"]))
        for k, v in topic_mains_sentence_dict[news_id].items():
            res[news_id]["result"]["mains"][topic_name_dict[k]] = v

    return res


def main():
    #  先加载测试数据，测试集文件可以通过命令行指定
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_file_path", type=str, required=True, help="预测文件的路径")
    parser.add_argument("--out_file_path", type=str, required=True, help="输出文件的路径")
    parser.add_argument("--out_file_mode", type=str, required=True, choices=["a", "w"], help="将模型结果保存的方式 : 追加/覆盖 （a/w）")
    parser.add_argument("--batch_size", type=int, default=128,  help="batch size ")
    parser.add_argument("--cuda_id", type=int, default=1,  help="gpu_id 0/1")

    args = parser.parse_args()
    predict_file_path = args.predict_file_path
    out_file_path = args.out_file_path
    out_file_mode = args.out_file_mode
    batch_size = args.batch_size
    cuda_id = args.cuda_id

    args_list = [
        # 漂绿
        ["--model", "MYBert",
         "--data_dir", "assets/data/topic_en_greenwashing/",
         "--gpu_ids", f"{cuda_id}",
         "--bert_type", os.path.join(os.path.dirname(__file__), "../assets/models/nbroad_ESG-BERT"),
         "--batch_size", f"{batch_size}",
         "--check_point_path",
         os.path.join(os.path.dirname(__file__),
                      "../assets/data/topic_en_greenwashing/saved_dict/MYBert/ESG-BERT.cpkt"),
         "--threshold", "0.7",
         "--cut_sen_len", "2",
         "--bert_layer_nums", "10",
         "--bert_split_dir", os.path.join(os.path.dirname(__file__), "../assets/models/nbroad_ESG-BERT_split")],
        # 客户隐私事件
        ["--model", "MYBert",
         "--data_dir", "assets/data/topic_en_Customer_Privacy_Incidents/",
         "--gpu_ids", f"{cuda_id}",
         "--bert_type", os.path.join(os.path.dirname(__file__), "../assets/models/nbroad_ESG-BERT"),
         "--batch_size", f"{batch_size}",
         "--check_point_path", os.path.join(os.path.dirname(__file__),
                                            "../assets/data/topic_en_Customer_Privacy_Incidents/saved_dict/MYBert/ESG-BERT.cpkt"),
         "--threshold", "0.5",
         "--cut_sen_len", "2",
         "--bert_layer_nums", "10",
         "--bert_split_dir", os.path.join(os.path.dirname(__file__), "../assets/models/nbroad_ESG-BERT_split")],
        # 安全事故
        ["--model", "MYBert",
         "--data_dir", "assets/data/topic_en_Safety_Accidents/",
         "--gpu_ids", f"{cuda_id}",
         "--bert_type", os.path.join(os.path.dirname(__file__), "../assets/models/nbroad_ESG-BERT"),
         "--batch_size", f"{batch_size}",
         "--check_point_path", os.path.join(os.path.dirname(__file__),
                                            "../assets/data/topic_en_Safety_Accidents/saved_dict/MYBert/ESG-BERT.cpkt"),
         "--threshold", "0.5",
         "--cut_sen_len", "2",
         "--bert_layer_nums", "10",
         "--bert_split_dir", os.path.join(os.path.dirname(__file__), "../assets/models/nbroad_ESG-BERT_split")]
    ]

    assert os.path.exists(predict_file_path), "File does not exist!"
    # 初始化这些参数部分
    init_models(args_list)
    out_file_f = open(out_file_path, out_file_mode)
    need_predict_news = load_jsonl_file(predict_file_path)
    models_predict_res: Dict = batch_models_predict(need_predict_news, models)
    for news_id in models_predict_res.keys():
        predict_topics = models_predict_res[news_id]["result"]["topics"]

        models_mains_res = models_predict_res[news_id]["result"]["mains"]
        mains_res = {
            "detail": {},
            "status": 1
        }
        for label in predict_topics:
            if label in models_mains_res.keys():
                mains_res["detail"][label] = models_mains_res[label]
        e = {"news_id": news_id, "topics": predict_topics, "mains": mains_res}
        out_file_f.write(str(e) + "\n")
    out_file_f.close()


if __name__ == '__main__':
    main()
