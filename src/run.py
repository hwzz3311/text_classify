import importlib
import logging
import os

import torch
from torch.utils.data import DataLoader

from src.models.config import BaseConfig
from src.options import RunArgs
from src.processors import build_dataset, dataset_collate_fn
from src.train_eval import train, test
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
    model.to(config.device)
    if config.model_name != "Transformer" and\
            "bert" not in config.model_name.lower():
        init_network(model)
    if config.check_point_path is not None and len(config.check_point_path):
        assert os.path.exists(config.check_point_path), "check point file not find !"
        model.load_state_dict(torch.load(config.check_point_path))
    if config.do_train:
        #  only train
        # 加载数据
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
        test_dataset = build_dataset(config, config.test_file)
        # do test
        test_iterator = DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  collate_fn=lambda x: dataset_collate_fn(config, x))
        test(config, model, test_iterator)






