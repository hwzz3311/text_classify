# data/dev/test相关参数
import argparse

from src.utils.model_utils import get_all_models, get_bert_types


class BaseArgs(object):

    @staticmethod
    def parser():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        parser.add_argument("--model", required=True, type=str, choices=get_all_models(), help=f"choices from :{'、'.join(get_all_models())}")

        parser.add_argument("--do_train", default=False, type=bool, help="do train ?")
        parser.add_argument("--do_dev", default=True, type=bool)
        parser.add_argument("--do_test", default=False, type=bool)
        parser.add_argument("--do_predict_news", default=False, type=bool)

        parser.add_argument("--test_file", default=None, type=str, help="do test file path!")
        parser.add_argument("--predict_file", default=None, type=str, help="do test file path!")

        parser.add_argument("--pad_size", default=512, type=int)

        parser.add_argument("--bert_type",
                            default="hfl/chinese-bert-wwm-ext",
                            type=str,
                            help=f"{get_bert_types()}")

        parser.add_argument("--batch_size", default=64, type=int, help="batch_size")
        parser.add_argument("--eval_scale", default=0.2, type=float, help="模型训练多少步进入eval")
        parser.add_argument('--data_dir', required=True, help='the data dir for train/dev/test')
        parser.add_argument('--max_seq_len', default=512, type=int, help="max seq len")
        parser.add_argument('--seed', default=42, type=int, help="random seed for initialization")
        parser.add_argument('--gpu_ids', type=str, default='-1',
                            help='gpu ids to use, -1 for cpu, "1, 3" for multi gpu')
        parser.add_argument('--embedding', default='random', type=str,
                            help='random or embedding_SougouNews.npz / embedding_Tencent.npz')
        parser.add_argument("--check_point_path", default=None, help="predict models check point path")
        parser.add_argument("--shuffle", default=True, action="store_true", help="dataloader shuffle ?")
        parser.add_argument("--loss", type=str, default="cross_entropy", help="loss function",
                            choices=["cross_entropy", "soft_bootstrapping_loss", "hard_bootstrapping_loss"], )
        parser.add_argument("--loss_beta", type=float, help="loss beta")
        parser.add_argument("--cut_sen_len", type=int, default=2, help="predict 模型下 将新闻content划分的成句子，每句的长度")
        parser.add_argument("--threshold", type=float, default=0.5, help="test / predict模型下的阈值，默认0.5")
        parser.add_argument("--gen_bert_emb_file", type=bool, default=True, help="重新生成gen_bert_emb_file 可以加快训练速度")
        parser.add_argument("--bert_layer_nums", type=int, default=10, help="保留多少层的bert")
        parser.add_argument("--bert_split_dir", default=None, help="被分层的bert模型 存储位置")
        parser.add_argument("--save_model_name", default=None, help="被保存模型的名字")
        parser.add_argument("--predict_base_keywords_file", default=None, type=str, help="预测模型使用的关键词文件")
        parser.add_argument("--R_drop", default=False, type=bool, help="是否需要R_drop")
        return parser

    def get_parser(self):
        parser = self.parser()
        parser = self.initialize(parser)
        return parser.parse_args()


class RunArgs(BaseArgs):
    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        parser = BaseArgs.initialize(parser)
        parser.add_argument("--num_epochs", default=6, type=int, help="epochs of train")

        parser.add_argument("--dropout", default=0.5, type=float, help="drop out probability")
        parser.add_argument("--lr", default=1e-3, type=float, help="learning rate for the models")
        parser.add_argument("--other_lr", default=2e-4, type=float, help="learning rate for the module except")

        parser.add_argument('--max_grad_norm', default=1.0, type=float,help='max grad clip')
        parser.add_argument('--warmup_proportion', default=0.05, type=float)
        parser.add_argument('--weight_decay', default=0., type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)

        # parser.add_argument('--eval_model', default=False, help='whether to eval models after training')
        parser.add_argument("--require_improvement", default=2000, type=int, help="end training early")
        parser.add_argument("--MOE_model", default=False, type=bool, help="使用MOE模式划分出的小模型去加载模型结构")

        return parser


