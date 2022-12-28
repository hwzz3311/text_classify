import importlib
import logging
import os

import torch
from torch.utils.data import DataLoader

from src.compression.moefication.utils import bert_change_forward
from src.compression.quantize import scale_quant_model
from src.models.config import BaseConfig
from src.options import RunArgs
from src.processors import build_dataset, dataset_collate_fn, load_jsonl, build_iter_bertatt
from src.train_eval import train, test, predict_batch
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
    if config.model_name != "Transformer" and\
            "bert" not in config.model_name.lower():
        init_network(model)
    if config.check_point_path is not None and len(config.check_point_path):
        assert os.path.exists(config.check_point_path), "check point file not find !"
        model.load_state_dict(torch.load(config.check_point_path))
    if config.do_train:
        #  only train
        # 加载数据
        if str(config.model_name).startswith("BertAtt"):
            train_dataset = load_jsonl(path=config.train_file)
            train_iterator = build_iter_bertatt(train_dataset, config=config)
            dev_dataset = load_jsonl(path=config.eval_file)
            dev_iterator = build_iter_bertatt(dev_dataset, config=config)
            train(config, model, train_iterator, dev_iterator)
        else:
            train_dataset = build_dataset(config, config.train_file)
            train_iterator = DataLoader(dataset=train_dataset,
                                        batch_size=config.batch_size,
                                        shuffle=config.shuffle,
                                        collate_fn=lambda x: dataset_collate_fn(config, x))

            dev_dataset = build_dataset(config, config.eval_file)
            dev_iterator = DataLoader(dataset=dev_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=config.shuffle,
                                      collate_fn=lambda x: dataset_collate_fn(config, x))

            train(config, model, train_iterator, dev_iterator)
    if config.do_test:
        if str(config.model_name).startswith("BertAtt"):
            test_dataset = load_jsonl(path=config.test_file)
            test_iterator = build_iter_bertatt(test_dataset, config=config)
        else:
            test_dataset = build_dataset(config, config.test_file)
            # do test
            test_iterator = DataLoader(dataset=test_dataset,
                                       batch_size=1,
                                       shuffle=True,
                                       collate_fn=lambda x: dataset_collate_fn(config, x))
        # 加载模型
        if config.check_point_path is not None and len(config.check_point_path) and config.do_train is False:
            assert os.path.exists(config.check_point_path), "check point file not find !"
            model.load_state_dict(torch.load(config.check_point_path))
        else:
            model.load_state_dict(torch.load(config.save_path))
        model.to(config.device)
        test(config, model, test_iterator)
    if config.do_predict_news:
        # TODO test for batch

        if config.check_point_path is not None and len(config.check_point_path) and config.do_train is False:
            assert os.path.exists(config.check_point_path), "check point file not find !"
            model.load_state_dict(torch.load(config.check_point_path))
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
        if str(config.model_name).startswith("BertAtt"):
            predict_dataset = load_jsonl(path=config.predict_file)
            predict_iterator = build_iter_bertatt(predict_dataset, config=config, is_predict=True)
        else:
            predict_dataset = build_dataset(config, config.predict_file, is_predict=True)
            predict_iterator = DataLoader(dataset=predict_dataset,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          collate_fn=lambda x: dataset_collate_fn(config, x))
        predict_batch(config, model, predict_iterator)
