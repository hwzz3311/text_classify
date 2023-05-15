import logging
import os
import time
from copy import deepcopy

# import neptune
import neptune
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score

from torch import nn
from torch.optim import AdamW, Adam
from torch.nn import functional as F
from tqdm import tqdm

from src.bootstrapping_loss.loss import SoftBootstrappingLoss, HardBootstrappingLoss, compute_kl_loss, \
    CustomCrossEntropyLoss

label_smoothing = 0.0005

from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from torchviz import make_dot

from src.models.config import BaseConfig
from src.processors import out_predict_dataset, out_test_dataset, batch_out_predict_dataset
from src.utils.model_utils import get_time_dif

α = 0


def train(config: BaseConfig, model: nn.Module, train_iter, dev_iter):
    start_time = time.time()
    run = neptune.init_run(
        project="zhengzhao134/text-classify",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmODJhNzQyZC1iODVjLTQxMGQtOTVmOC02YWY2YzdiZTgxNWUifQ==",
        tags=f"loss_fun:{config.loss_fun};\nmodel:{config.model_name};\nlr:{config.lr};\nbert_type:{config.bert_type}"
    )  # your credentials

    model.train()
    if "bert" in str(config.model_name).lower():
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)
    else:
        # optimizer = Adam(model.parameters(),
        #                  lr=config.lr,
        #                  eps=config.adam_epsilon,
        #                  weight_decay=config.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    if "bert" in config.model_name.lower():
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = 1
    dev_best_acc = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    if config.loss_fun == "cross_entropy":
        loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif config.loss_fun == "soft_bootstrapping_loss":
        loss_func = SoftBootstrappingLoss(beta=config.loss_beta, as_pseudo_label=True)
    elif config.loss_fun == "hard_bootstrapping_loss":
        loss_func = HardBootstrappingLoss(beta=config.loss_beta)
    elif config.loss_fun == "custom_cross_entropy":
        keywords = [x.strip() for x in open(os.path.join(os.path.dirname(__file__),
                                                         "../assets/data/topic_en_greenwashing/base_keywords.txt")).readlines()]
        vocab_weights = {
            x: 2.0 for x in keywords
        }
        tokenizer = AutoTokenizer.from_pretrained(config.bert_type)
        loss_func = CustomCrossEntropyLoss(vocab_weights, tokenizer)
    log_dir = os.path.join(config.log_path, './' + config.model_name, f"./{config.bert_type}",
                           time.strftime('%m-%d_%H.%M', time.localtime()))
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    print(f"train_iter length:{len(train_iter)}")
    eval_step = int(len(train_iter) * config.eval_scale)
    print(f"eval_step :{eval_step}")
    train_log_step = int(eval_step / 2)
    print(f"train_log_step :{train_log_step}")
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        print(f"time_dif : {get_time_dif(start_time)}")
        for i, _data in enumerate(train_iter):
            optimizer.zero_grad()
            trains, labels = _data[0], _data[1]
            if config.R_drop:
                try:
                    ##R-drop
                    outputs1 = model(trains)
                    outputs2 = model(trains)
                    outputs = torch.div((outputs1+outputs2),0.5)
                    ce_loss = 0.5 * (loss_func(outputs1, labels) + loss_func(outputs2, labels))
                    kl_loss = compute_kl_loss(outputs1, outputs2)
                    loss = ce_loss + α * kl_loss

                    #loss = loss_func(outputs, labels)
                except:
                    outputs = model(trains)
                    outputs = torch.unsqueeze(outputs, 0)
                    loss = loss_func(outputs, labels)
            else:
                outputs = model(trains)
                # print(f"train - outputs:{outputs.shape}, labels : {labels.shape}")
                model.zero_grad()
                try:
                    if config.loss_fun == "custom_cross_entropy":
                        loss = loss_func(outputs, labels, trains[0])
                    else:
                        loss = loss_func(outputs, labels)
                except:
                    outputs = torch.unsqueeze(outputs, 0)
                    if config.loss_fun == "custom_cross_entropy":
                        loss = loss_func(outputs, labels, trains[0])
                    else:
                        loss = loss_func(outputs, labels)
            run["loss"].log(loss.item(), step=total_batch)
            loss.backward()

            optimizer.step()
            make_dot(outputs, params=dict(model.named_parameters())).render(config.model_name, format="pdf")

            if total_batch != 0 and total_batch % eval_step == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss, labels_all, predict_all = evaluate(config, model, dev_iter)
                run["dev_loos"].log(dev_loss, step=total_batch)
                run["dev_acc"].log(dev_acc, step=total_batch)
                run["precision"].log(precision_score(labels_all, predict_all))
                run["recall"].log(recall_score(labels_all, predict_all))
                run["f1"].log(f1_score(labels_all, predict_all))
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                # output_dir = os.path.join(os.path.dirname(config.save_path), "checkpoint-{}".format(total_batch))
                # os.makedirs(output_dir, exist_ok=True)
                # print(f"saving checkpoint-{total_batch} to {output_dir}")
                # torch.save(model.state_dict(), os.path.join(output_dir, os.path.basename(config.save_path)))
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.8},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.8},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch % train_log_step == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.8},  Train Acc: {2:>6.2%}, Time: {3}'
                print(msg.format(total_batch, loss.item(), train_acc, time_dif))
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if "bert" in config.model_name.lower():
            scheduler.step()  # 学习率衰减
        if flag:
            break
    writer.close()


def test(config: BaseConfig, model: nn.Module, data_iter):
    model.eval()
    start_time = time.time()
    use_threshold = config.test_by_threshold
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, data_iter, test_mode=True,
                                                                use_threshold=use_threshold)
    msg = f"Test loss : {test_loss},Test acc: {test_acc}"
    print(msg)
    print(test_report)
    print(test_confusion)
    print(f"Used time: {get_time_dif(start_time)}")


def predict(config: BaseConfig, model: nn.Module, data_iter):
    model.eval()
    start_time = time.time()
    news_ids_all = []
    origin_text_all = []
    predict_result_all = []
    predict_result_score_all = []
    softmax_fun = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for i, _data in tqdm(enumerate(data_iter), total=len(data_iter), desc="predict ing"):
            # trains, labels, news_ids = _data[0], _data[1], _data[2]
            trains, labels, news_ids, origin_text = _data[0], _data[1], _data[2], _data[3]
            outputs = model(trains)
            try:
                predict_result = torch.max(outputs.data, dim=1)[1].cpu()
            except:
                outputs = torch.unsqueeze(outputs, 0)
                predict_result = torch.max(outputs.data, dim=1)[1].cpu()

            softmax_output = softmax_fun(outputs)
            outputs = softmax_output
            predict = torch.where(outputs > config.threshold, torch.ones_like(outputs), torch.zeros_like(outputs))
            predict_results = []
            predict_result_score = []
            softmax_output_list = softmax_output.data.cpu().numpy().tolist()
            predict_list = predict.cpu().numpy().tolist()
            for predict_result, softmax_output in zip(predict_list, softmax_output_list):
                predict_results.append(config.class_list[np.argmax(predict_result)])
                predict_result_score.append(softmax_output[np.argmax(predict_result)])
            news_ids_all.extend(news_ids)
            # 保存token id，在后续的结果中进行还原
            origin_text_all.extend(origin_text)
            predict_result_all.extend(predict_results)
            predict_result_score_all.extend(predict_result_score)
            # print(f"{len(predict_result_all)=} , {len(news_ids_all)=}, {len(origin_text_all)=}")
            assert len(predict_result_all) == len(news_ids_all) == len(
                origin_text_all), f"{len(predict_result_all)=} , {len(news_ids_all)=}, " \
                                  f"{len(origin_text_all)=},origin_text : {origin_text} , news_ids :{news_ids}"

    out_predict_dataset(predict_result_all, predict_result_score_all, news_ids_all, origin_text_all, config)

    print(f"Used time: {get_time_dif(start_time)}")


def predict_batch(config: BaseConfig, model: nn.Module, data_iter, news_datas, debug=False):
    model.eval()
    start_time = time.time()
    news_ids_all = []
    origin_text_all = []
    predict_result_all = []
    predict_result_score_all = []
    softmax_fun = torch.nn.Softmax(dim=1)
    tmp_dict = {
        "result": {
            "topics": [],
            "mains": {}}
    }
    res = {
        f"{e['news_id']}": deepcopy(tmp_dict) for e in news_datas
    }
    tmp_dict = {
        config.class_list[1]: []
    }

    topic_mains_sentence_dict = {
        f"{e['news_id']}": deepcopy(tmp_dict) for e in news_datas
    }
    if debug:
        res["result"]["debug_info"] = {
            config.class_list[1]: []
        }

    with torch.no_grad():
        for i, _data in tqdm(enumerate(data_iter), total=len(data_iter), desc="predict batch ing"):
            trains, labels, news_ids, origin_text = _data[0], _data[1], _data[2], _data[3]
            outputs = model(trains)
            try:
                predict_result = torch.max(outputs.data, dim=1)[1].cpu()
            except:
                outputs = torch.unsqueeze(outputs, 0)
                predict_result = torch.max(outputs.data, dim=1)[1].cpu()

            softmax_output = softmax_fun(outputs)
            outputs = softmax_output
            predict = torch.where(outputs > config.threshold, torch.ones_like(outputs), torch.zeros_like(outputs))
            predict_results = []
            predict_result_score = []
            softmax_output_list = softmax_output.data.cpu().numpy().tolist()
            predict_list = predict.cpu().numpy().tolist()
            for predict_result, softmax_output in zip(predict_list, softmax_output_list):
                predict_results.append(config.class_list[np.argmax(predict_result)])
                predict_result_score.append(softmax_output[np.argmax(predict_result)])
            news_ids_all.extend(news_ids)
            # 保存token id，在后续的结果中进行还原
            origin_text_all.extend(origin_text)
            predict_result_all.extend(predict_results)
            predict_result_score_all.extend(predict_result_score)
            assert len(predict_result_all) == len(news_ids_all) == len(
                origin_text_all), f"{len(predict_result_all)=} , {len(news_ids_all)=}, " \
                                  f"{len(origin_text_all)=},origin_text : {origin_text} , news_ids :{news_ids}"

    predict_res_dict = batch_out_predict_dataset(predict_result_all, predict_result_score_all, news_ids_all,
                                                 origin_text_all, config)
    for news_id in predict_res_dict.keys():
        if predict_res_dict[news_id]["label"]:
            res[news_id]["result"]["topics"].append(config.class_list[1])

            for sen, score in zip(predict_res_dict[news_id]["support_sentence"],
                                  predict_res_dict[news_id]["result_score"]):
                topic_mains_sentence_dict[news_id][config.class_list[1]].append({
                    "probability": score,
                    "text": sen
                })
    if debug:
        for text, softmax_output in zip(origin_text, softmax_output_list):
            res["result"]["debug_info"][config.class_list[1]].append({
                "text": text,
                "softmax_output": {
                    label: score for label, score in
                    zip(config.class_list, softmax_output)
                }
            })

    print(f"Used time: {get_time_dif(start_time)}")
    for news_id in res.keys():
        res[news_id]["result"]["topics"] = list(set(res[news_id]["result"]["topics"]))
        for k, v in topic_mains_sentence_dict[news_id].items():
            res[news_id]["result"]["mains"][k] = v
    if debug:
        for k in res["result"]["debug_info"].copy().keys():
            res["result"]["debug_info"][k] = res["result"]["debug_info"].pop(k)
    return res


def evaluate(config: BaseConfig, model: nn.Module, data_iter, test_mode=False, predict_mode=False, use_threshold=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype="int")
    labels_all = np.array([], dtype="int")
    news_ids_all = []
    input_tokens_all = []
    original_text_all = []
    softmax_score_all = []
    softmax_fun = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for i, _data in enumerate(data_iter):
            trains, labels, news_ids, original_text = _data[0], _data[1], _data[2], _data[3]
            input_tokens = trains[0]
            outputs = model(trains)
            # print(f"evaluate - outputs : {outputs.shape} , labels : {labels.shape}")
            try:
                loss = F.cross_entropy(outputs, labels, label_smoothing=label_smoothing)
            except:
                outputs = torch.unsqueeze(outputs, 0)
                loss = F.cross_entropy(outputs, labels, label_smoothing=label_smoothing)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            softmax_output = softmax_fun(outputs.data)
            if use_threshold:
                if config.threshold_for_positive > -1:
                    outputs = softmax_output
                    softmax_output_list = softmax_output.data.cpu().numpy().tolist()
                    predict_results = []
                    predict_result_score = []
                    for _softmax_output in softmax_output_list:
                        if _softmax_output[1] > config.threshold_for_positive:
                            predict_results.append(1)
                        else:
                            predict_results.append(0)
                        predict_result_score.append(_softmax_output)
                    predict_all = np.append(predict_all, np.array(predict_results))
                    softmax_score_all.extend(predict_result_score)
                else:
                    outputs = softmax_output
                    predict = torch.where(outputs > config.threshold, torch.ones_like(outputs),
                                          torch.zeros_like(outputs))
                    predict_results = []
                    predict_result_score = []
                    softmax_output_list = softmax_output.data.cpu().numpy().tolist()
                    predict_list = predict.cpu().numpy().tolist()
                    for predict_result, softmax_output in zip(predict_list, softmax_output_list):
                        predict_results.append(np.argmax(predict_result))
                        predict_result_score.append(softmax_output)
                    # print("predict_result_score : ", predict_results)
                    predict_all = np.append(predict_all, np.array(predict_results))
                    softmax_score_all.extend(predict_result_score)
            else:
                softmax_score = softmax_output.data.cpu().numpy().tolist()
                predict = torch.max(outputs.data, dim=1)[1].cpu().numpy()

                predict_all = np.append(predict_all, predict)
                softmax_score_all.extend(softmax_score)
            labels_all = np.append(labels_all, labels)
            original_text_all.extend(original_text)
            # news_ids_all = news_ids_all.append(news_ids)
            # 保存token id，在后续的结果中进行还原
            # input_tokens_all = input_tokens_all.append(input_tokens.data.cpu().numpy().tolist())
    if not predict_mode:
        acc = metrics.accuracy_score(labels_all, predict_all)
        print("***** eval report start")
        print(metrics.classification_report(labels_all, predict_all))
        tn, fp, fn, tp = metrics.confusion_matrix(labels_all, predict_all).ravel()
        print(f"(tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}) ")
        print("***** eval report end")
    if test_mode:
        report = None
        confusion = None
        try:
            report = metrics.classification_report(labels_all, predict_all)
            print(report)
            tn, fp, fn, tp = metrics.confusion_matrix(labels_all, predict_all).ravel()
            print(f"(tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}) ")
        except:
            pass
        # 将预测结果写到文件
        predict_all_list = predict_all.tolist()
        labels_all_list = labels_all.tolist()
        out_test_dataset(predict_all_list, labels_all_list, original_text_all, softmax_score_all, config)
        return acc, loss_total / len(data_iter), report, confusion
    if predict_mode:
        predict_all_list = predict_all.tolist()
        return predict_all_list, news_ids_all, input_tokens_all
    return acc, loss_total / len(data_iter), labels_all, predict_all
