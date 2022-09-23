import argparse
import os

import torch


class BaseConfig(object):
    def __init__(self, args: argparse.ArgumentParser):
        print(args.__dict__)
        self.model_name = args.model

        self.bert_type = args.bert_type
        self.batch_size = args.batch_size
        self.n_vocab = 0
        self.local_model = True if os.path.exists(args.bert_type) else False
        self.bert_dir = args.bert_type if self.local_model else None
        dir_list = [dir_name for dir_name in os.listdir(os.path.join(os.path.dirname(__file__), "../../assets/data/"))]
        assert os.path.exists(os.path.join(os.path.dirname(__file__), "../../", args.data_dir)), f"choose a dataset : {', '.join(dir_list)}"
        self.data_dir = args.data_dir
        args.data_dir = os.path.join(os.path.dirname(__file__), "../../", args.data_dir)
        self.train_file = os.path.join(args.data_dir, "train.json")
        self.eval_file = os.path.join(args.data_dir, "eval.json")
        self.test_file = args.test_file if args.test_file is not None and len(args.test_file) and\
                                           os.path.exists(args.test_file) else os.path.join(args.data_dir, "test.json")
        self.class_list = [x.strip() for x in open(os.path.join(args.data_dir, "labels.txt")).readlines()]
        self.class_ids = [i for i in range(len(self.class_list))]
        self.vocab_path = os.path.join(args.data_dir, "vocab.pkl")
        if "bert" in self.model_name.lower():
            bert_type = self.bert_type
            if "/" in bert_type:
                bert_type = self.bert_type.split("/")[1]
            checkpoint_file_name = f"saved_dict/{self.model_name}/{bert_type}.cpkt"
            log_dir = f"{self.model_name}/{self.bert_type}/"
        else:
            checkpoint_file_name = f"saved_dict/{self.model_name}.cpkt"
            log_dir = f"{self.model_name}/"
        self.save_path = os.path.join(args.data_dir, checkpoint_file_name)
        self.log_path = os.path.join(args.data_dir, log_dir)
        self.gpu_ids = str(args.gpu_ids).split(",")
        # TODO 待添加多卡的支持
        if self.gpu_ids[0] == '-1':
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{int(self.gpu_ids[0])}")
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.dropout = args.dropout
        self.seed = args.seed
        self.pad_size = args.pad_size
        self.max_seq_len = args.max_seq_len
        self.num_classes = len(self.class_list)
        self.lr = args.lr
        if "bert" in self.model_name.lower() and self.lr == 1e-3:
            self.lr = 5e-5
        self.require_improvement = args.require_improvement
        self.num_epochs = args.num_epochs
        self.other_lr = args.other_lr
        self.max_grad_norm = args.max_grad_norm
        self.warmup_proportion = args.warmup_proportion
        self.weight_decay = args.weight_decay
        self.adam_epsilon = args.adam_epsilon
        self.require_improvement = args.require_improvement
        self.embedding = args.embedding
        self.check_point_path = args.check_point_path
        self.shuffle = args.shuffle

        self.do_train = args.do_train
        self.do_dev = args.do_dev
        self.do_test = args.do_test

        # self.eval_model_dir = args.eval_model_dir if hasattr(args, "eval_model_dir") else None
        # self.test_model_dir = args.test_model_dir if hasattr(args, "test_model_dir") else None
        self.predict_out_dir = args.predict_out_dir if hasattr(args, "predict_out_dir") else args.data_dir


