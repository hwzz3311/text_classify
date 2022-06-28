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
        parser.add_argument("--do_test", default=True, type=bool)

        parser.add_argument("--test_file", default=None, type=str, help="do test file path!")

        parser.add_argument("--pad_size", default=512, type=int)

        parser.add_argument("--bert_type",
                            default="hfl/chinese-bert-wwm-ext",
                            type=str,
                            help=f"{get_bert_types()}")

        parser.add_argument("--batch_size", default=64, type=int, help="batch_size")
        parser.add_argument('--data_dir', required=True, help='the data dir for train/dev/test')
        parser.add_argument('--max_seq_len', default=512, type=int, help="max seq len")
        parser.add_argument('--seed', default=42, type=int, help="random seed for initialization")
        parser.add_argument('--gpu_ids', type=str, default='-1',
                            help='gpu ids to use, -1 for cpu, "1, 3" for multi gpu')
        parser.add_argument('--embedding', default='random', type=str,
                            help='random or embedding_SougouNews.npz / embedding_Tencent.npz')
        parser.add_argument("--check_point_path", default=None, help="predict models check point path")
        parser.add_argument("--shuffle", action="store_true", help="dataloader shuffle ?")


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

        return parser


