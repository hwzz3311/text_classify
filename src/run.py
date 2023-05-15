import importlib
import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.compression.moefication.utils import bert_change_forward
from src.compression.quantize import scale_quant_model
from src.models.config import BaseConfig
from src.options import RunArgs
from src.processors import build_dataset, dataset_collate_fn, load_jsonl, build_iter_bertatt
from src.train_eval import train, test, predict_batch, predict
from src.utils.data_utils import load_jsonl_file, pre_cut_sentences
from src.utils.model_utils import set_seed, init_network, get_vocab

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)

if __name__ == "__main__":
    args = RunArgs().get_parser()
    logger.info(args.__dict__)
    mode_name = args.model
    x = importlib.import_module(f"src.models.{mode_name}")
    config: BaseConfig = x.Config(args)
    # 设置随机种子
    set_seed(config)
    logger.info(f'Process info : {config.device} , gpu_ids : {config.gpu_ids}, model_name : {config.model_name}')
    logger.info(config.__dict__)

    vocab = get_vocab(config, use_word=False)
    config.n_vocab = len(vocab)

    model = x.Model(config)
    if config.model_name != "Transformer" and \
            "bert" not in config.model_name.lower():
        init_network(model)
    if config.check_point_path is not None and len(config.check_point_path):
        assert os.path.exists(config.check_point_path), "check point file not find !"
        model.load_state_dict(torch.load(config.check_point_path, map_location="cpu"))
    print(f"model to : {config.device}")
    model.to(config.device)
    att = True if str(config.model_name).startswith("BertAtt") else False
    config.att = att
    if config.do_train:
        #  only train
        # 加载数据

        # if str(config.model_name).startswith("BertAtt"):
        #     train_dataset = load_jsonl_file(config.train_file)
        #     train_iterator = build_iter_bertatt(train_dataset, config=config, att=att)
        #     dev_dataset = load_jsonl_file(config.eval_file)
        #     dev_iterator = build_iter_bertatt(dev_dataset, config=config)
        #     train(config, model, train_iterator, dev_iterator)
        # else:
        dataset = load_jsonl_file(config.train_file)
        train_dataset = build_dataset(config, dataset, att=att)
        train_iterator = DataLoader(dataset=train_dataset,
                                    batch_size=config.batch_size,
                                    shuffle=config.shuffle,
                                    collate_fn=lambda x: dataset_collate_fn(config, x))
        dataset = load_jsonl_file(config.eval_file)
        dev_dataset = build_dataset(config, dataset, att=att)
        dev_iterator = DataLoader(dataset=dev_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=config.shuffle,
                                  collate_fn=lambda x: dataset_collate_fn(config, x))

        train(config, model, train_iterator, dev_iterator)
    if config.do_test:
        # if str(config.model_name).startswith("BertAtt"):
        #     test_dataset = load_jsonl_file(config.test_file)
        #     test_iterator = build_iter_bertatt(test_dataset, config=config)
        # else:
        dataset = load_jsonl_file(config.test_file)
        test_dataset = build_dataset(config, dataset, att)
        # do test
        test_iterator = DataLoader(dataset=test_dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   collate_fn=lambda x: dataset_collate_fn(config, x))
        # 加载模型
        if config.check_point_path is not None and len(config.check_point_path) and config.do_train is False:
            assert os.path.exists(config.check_point_path), "check point file not find !"
            model.load_state_dict(torch.load(config.check_point_path, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(config.save_path, map_location="cpu"))
        model.to(config.device)
        test(config, model, test_iterator)
    if config.do_predict_news:
        # TODO test for batch
        if config.check_point_path is not None and len(config.check_point_path) and config.do_train is False:
            assert os.path.exists(config.check_point_path), "check point file not find !"
            model.load_state_dict(torch.load(config.check_point_path, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(config.save_path, map_location="cpu"))
        if config.MOE_model:
            bert_type = config.bert_type
            if "/" in config.bert_type:
                bert_type = config.bert_type.split("/")[1]
            param_split_dir = os.path.join(os.path.dirname(__file__), "../", config.data_dir,
                                           f"saved_dict/{config.model_name}/MOE/{bert_type}")
            assert os.path.exists(param_split_dir), "MOE model file not found"
            bert_change_forward(model, param_split_dir, config.device, 20)
        model.to(config.device)
        need_predict_news = load_jsonl_file(config.predict_file, is_predict=False, config=config)
        dataset = pre_cut_sentences(need_predict_news, config)
        if str(config.model_name).startswith("BertAtt"):
            predict_iterator = build_iter_bertatt(dataset, config=config, is_predict=True)
        else:
            predict_dataset = build_dataset(config, dataset, is_predict=True)
            predict_iterator = DataLoader(dataset=predict_dataset,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          collate_fn=lambda x: dataset_collate_fn(config, x))
        predict(config, model, predict_iterator)
    if config.do_batch_infer_news:
        # 批量的 infer 未标注的原始数据集
        if config.check_point_path is not None and len(config.check_point_path) and config.do_train is False:
            assert os.path.exists(config.check_point_path), "check point file not find !"
            model.load_state_dict(torch.load(config.check_point_path, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(config.save_path, map_location="cpu"))
        if config.MOE_model:
            bert_type = config.bert_type
            if "/" in config.bert_type:
                bert_type = config.bert_type.split("/")[1]
            param_split_dir = os.path.join(os.path.dirname(__file__), "../", config.data_dir,
                                           f"saved_dict/{config.model_name}/MOE/{bert_type}")
            assert os.path.exists(param_split_dir), "MOE model file not found"
            bert_change_forward(model, param_split_dir, config.device, 20)
        model.to(config.device)
        need_predict_news = load_jsonl_file(config.predict_file, is_predict=False, config=config)
        need_predict_news_id_dict = {
            x["news_id"]: x for x in need_predict_news
        }
        news_batch = 2000
        t_news_count = 0
        assert config.predict_out_file is not None, "Need to specify the output path of forecast news; 需要指定输出文件的路径"
        os.makedirs(os.path.dirname(config.predict_out_file), exist_ok=True)
        out_file_f = open(config.predict_out_file, "a")
        for i in tqdm([i for i in range(0, len(need_predict_news), news_batch)], desc="分段 batch ing"):
            datas = need_predict_news[i:i + news_batch]
            dataset = pre_cut_sentences(datas, config)
            # TODO 更新 BertAtt 在推理阶段的代码
            if str(config.model_name).startswith("BertAtt"):
                predict_iterator = build_iter_bertatt(dataset, config=config, is_predict=True)
            else:
                predict_dataset = build_dataset(config, dataset, is_predict=True)
                predict_iterator = DataLoader(dataset=predict_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              collate_fn=lambda x: dataset_collate_fn(config, x))
            models_predict_res = predict_batch(config, model, predict_iterator, datas)
            for news_id in models_predict_res.keys():
                predict_topics = models_predict_res[news_id]["result"]["topics"]

                models_mains_res = models_predict_res[news_id]["result"]["mains"]
                mains_res = {
                    "detail": {},
                    "status": 0
                }
                for label in predict_topics:
                    if label in models_mains_res.keys():
                        mains_res["detail"][label] = models_mains_res[label]
                if len(predict_topics) > 0:
                    t_news_count += 1
                    news_info = need_predict_news_id_dict[news_id]
                    title = news_info["title"]
                    content = news_info["content"]
                    e = {"news_id": news_id, "title": title, "content": content, "topics": predict_topics,
                         "mains": mains_res}
                    out_file_f.write(str(e) + "\n")
            print(f"t_news_count : {t_news_count}")
            out_file_f.flush()
        out_file_f.close()
