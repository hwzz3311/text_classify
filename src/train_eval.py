import logging
import os
import time

import numpy as np
import torch
from sklearn import metrics

from torch import nn
from torch.optim import AdamW, Adam
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from src.models.config import BaseConfig
from src.processors import out_predict_dataset
from src.utils.model_utils import get_time_dif

def train(config: BaseConfig, model: nn.Module, train_iter, dev_iter):
    start_time = time.time()

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

    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        print(f"time_dif : {get_time_dif(start_time)}")
        for i, _data in enumerate(train_iter):
            optimizer.zero_grad()
            trains, labels = _data[0], _data[1]
            outputs = model(trains)
            # print(f"train - outputs:{outputs.shape}, labels : {labels.shape}")
            model.zero_grad()
            try:
                loss = F.cross_entropy(outputs, labels)
            except:
                outputs = torch.unsqueeze(outputs, 0)
                loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.8},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.8},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
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
    # 加载模型
    if config.check_point_path is not None and len(config.check_point_path) and config.do_train is False:
        assert os.path.exists(config.check_point_path), "check point file not find !"
        model.load_state_dict(torch.load(config.check_point_path))
    else:
        model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, data_iter, test_mode=True)
    msg = f"Test loss : {test_loss},Test acc: {test_acc}"
    print(msg)
    print(test_report)
    print(test_confusion)
    print(f"Used time: {get_time_dif(start_time)}")


def evaluate(config: BaseConfig, model: nn.Module, data_iter, test_mode=False):

    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype="int")
    labels_all = np.array([], dtype="int")
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            # print(f"evaluate - outputs : {outputs.shape} , labels : {labels.shape}")
            try:
                loss = F.cross_entropy(outputs, labels)
            except:
                outputs = torch.unsqueeze(outputs, 0)
                loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, dim=1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test_mode:
        report = None
        confusion = None
        try:
            report = metrics.classification_report(labels_all, predict_all)
            print(report)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            print(confusion)
        except:
            pass
        # 将预测结果写到文件
        predict_all_list = predict_all.tolist()
        out_predict_dataset(predict_all_list, config)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
