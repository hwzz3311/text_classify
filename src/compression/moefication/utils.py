import types
from typing import DefaultDict
import sys
import torch
import os
import tqdm
from collections import Counter
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from k_means_constrained import KMeansConstrained
from transformers.models.bert.modeling_bert import BertIntermediate


def get_layer_num(filename):
    model = torch.load(filename, map_location='cpu')['module']
    enc_keys = [x for x in model.keys() if 'ff.dense_relu_dense.wi.weight' in x and 'encoder' in x]
    dec_keys = [x for x in model.keys() if 'ff.dense_relu_dense.wi.weight' in x and 'decoder' in x]

    enc_nums = [int(x.split('.')[2]) for x in enc_keys]
    dec_nums = [int(x.split('.')[2]) for x in dec_keys]

    return max(enc_nums)+1, max(dec_nums)+1

def load_ffn_weight(filename, template, layer):

    model = torch.load(filename, map_location='cpu')
    key = template.format(layer)

    return model[key].numpy()

def load_hidden_states(folder, filename):
    target = os.path.join(folder, filename)
    vecs = torch.load(target)
    return vecs

class ModelConfig:

    def __init__(self, filename, folder, split_num):
        self.filename = filename
        self.folder = folder
        self.split_num = split_num

class LayerSplit:

    def __init__(self, config : ModelConfig, template, layer=0):
        self.config = config
        self.layer = layer
        self.template = template

    def split(self):
        pass

    def save(self):
        save_folder = os.path.join(self.config.folder, self.type)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        filename = os.path.join(save_folder, self.template.format(self.layer))
        torch.save(self.labels, filename)

    def cnt(self):
        print(Counter(self.labels))

    def load_param(self):
        self.ffn_weight = load_ffn_weight(self.config.filename, self.template, self.layer)
        self.neuron_num = self.ffn_weight.shape[0]
        self.split_size = self.neuron_num // self.config.split_num
        assert self.split_size * self.config.split_num == self.neuron_num

class RandomSplit(LayerSplit):

    def __init__(self, config: ModelConfig, layer=0, is_encoder=True):
        super().__init__(config, layer=layer, is_encoder=is_encoder)
        self.type = 'random_split'

    def split(self):
        self.load_param()

        self.labels = [i // self.split_size for i in range(self.neuron_num)]

class ParamSplit(LayerSplit):

    def __init__(self, config: ModelConfig, template, layer=0):
        super().__init__(config, template=template, layer=layer)
        self.type = 'param_split'

    def split(self):
        self.load_param()
        ffn_weight_norm = sklearn.preprocessing.normalize(self.ffn_weight)

        kmeans = KMeansConstrained(n_clusters=self.config.split_num, size_min=self.split_size, size_max=self.split_size, random_state=0).fit(ffn_weight_norm, None)

        self.labels = [x for x in kmeans.labels_]

class BlockCenter:

    def __init__(self, config, template, filename, layer):
        self.config = config
        self.filename = filename
        self.labels = torch.load(filename)
        self.template = template

        self.layer = layer

    def cal_center(self):
        pass

    def save(self):
        print(self.centers.shape)
        torch.save(self.centers, "{}_{}".format(self.filename, self.type))
        self.save_acc()

    def save_acc(self):
        with open("{}_{}_acc".format(self.filename, self.type), 'w') as fout:
            fout.write(str(self.acc))

class RandomCenter(BlockCenter):

    def __init__(self, config, filename):
        super().__init__(config, filename)
        self.type = "random"

    def cal_center(self):
        ffn_weight = load_ffn_weight(self.config.filename, self.layer, self.is_encoder)
        ffn_weight_norm = ffn_weight

        d = {}
        for i, x in enumerate(self.labels):
            if x not in d:
                d[x] = ffn_weight_norm[i, :]
        centers = sorted(list(d.items()), key=lambda x: x[0])

        self.centers = sklearn.preprocessing.normalize(np.array([x[1] for x in centers]))
        self.acc = 0

class ParamCenter(BlockCenter):

    def __init__(self, config, filename, layer):
        super().__init__(config, filename, layer)
        self.type = "param"

    def cal_center(self):
        ffn_weight = load_ffn_weight(self.config.filename, self.layer, self.is_encoder)
        ffn_weight_norm = sklearn.preprocessing.normalize(ffn_weight)

        centers = []
        num_blocks = max(self.labels) + 1
        for i in range(num_blocks):
            centers.append(ffn_weight_norm[np.array(self.labels) == i, :].mean(0))

        centers = np.array(centers)
        self.centers = centers

        centers = torch.tensor(centers).cuda().unsqueeze(0)

        patterns = []
        for i in range(num_blocks):
            patterns.append(np.array(self.labels) == i)
        patterns = torch.Tensor(patterns).cuda().float().transpose(0, 1) # 4096, num_blocks

        acc = []
        hiddens = load_hidden_states(self.config.folder, self.template.format(self.layer))
        hiddens = torch.cat(hiddens, 0).float()
        hiddens = hiddens.view(-1, hiddens.shape[-1])
        hiddens = hiddens / torch.norm(hiddens, dim=-1).unsqueeze(-1)
        num = hiddens.shape[0]

        ffn_weight = torch.tensor(ffn_weight).cuda().transpose(0, 1).float()
        for i in range(num // 10 * 9, num, 512):
            with torch.no_grad():
                input = hiddens[i:i+512, :].cuda()
                acts = torch.relu((torch.matmul(input, ffn_weight))) # 512, 4096
                scores = torch.matmul(acts, patterns) # 512, num_blocks, vary from 0 to 1
                labels = torch.topk(scores, k=25, dim=-1)[1]

                input = input / torch.norm(input, dim=-1).unsqueeze(-1)
                dist = -1 * torch.norm(input.unsqueeze(1).expand(-1, num_blocks, -1) - centers, dim=-1)
                pred = torch.topk(dist, k=25, dim=-1)[1]

                for x, y in zip(labels, pred):
                    x = set(x.cpu().numpy())
                    y = set(y.cpu().numpy())
                    acc.append(len(x & y) / 25)
        print("param acc", np.mean(acc))
        sys.stdout.flush()
        self.acc = np.mean(acc)

class MLPCenter(BlockCenter):
    def __init__(self, config, template, filename, layer):
        super().__init__(config, template, filename, layer)
        self.type = "input_compl"

    def cal_center(self):
        ffn_weight = load_ffn_weight(self.config.filename, self.template, self.layer)
        ffn_weight_norm_ = sklearn.preprocessing.normalize(ffn_weight)
        centers = []
        num_blocks = max(self.labels) + 1
        for i in range(num_blocks):
            centers.append(ffn_weight_norm_[np.array(self.labels) == i, :].mean(0))
        centers = np.array(centers) # num_blocks, 1024

        ffn_weight = torch.tensor(ffn_weight).cuda().transpose(0, 1).float()
        patterns = []
        num_blocks = max(self.labels) + 1
        for i in range(num_blocks):
            patterns.append(np.array(self.labels) == i)
        patterns = torch.Tensor(patterns).cuda().float().transpose(0, 1)

        hiddens = load_hidden_states(self.config.folder, self.template.format(self.layer))

        hiddens = hiddens / torch.norm(hiddens, dim=-1).unsqueeze(-1)

        model = torch.nn.Sequential(torch.nn.Linear(hiddens.shape[-1], num_blocks, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(num_blocks, num_blocks, bias=False))

        def weights_init(m):
            if isinstance(m, torch.nn.Linear):
                if m.weight.shape[-1] == hiddens.shape[-1]:
                    m.weight.data = torch.from_numpy(centers).float()
                else:
                    m.weight.data = torch.eye(m.weight.data.shape[0])
                    #torch.nn.init.normal_(m.weight.data)
                #m.bias.data[:] = 0

        model.apply(weights_init)

        model.cuda()

        optim = torch.optim.Adam(model.parameters(), lr=0.01)

        loss_func = torch.nn.BCEWithLogitsLoss()

        save_acc = [0, 0]
        save_epoch = [-1, -1]

        self.centers = model

        train_hiddens = hiddens[:hiddens.shape[0] // 10 * 9, :]
        #pos_max = None

        last_epoch = -1

        for epoch in range(30):
            train_hiddens=train_hiddens[torch.randperm(train_hiddens.size()[0])]

            pbar = tqdm.tqdm(range(0, train_hiddens.shape[0], 512))
            for i in pbar:
                model.zero_grad()

                input = train_hiddens[i:i+512, :].float().clone().detach().cuda()
                with torch.no_grad():
                    acts = torch.relu((torch.matmul(input, ffn_weight))).float()
                    scores = torch.matmul(acts, patterns)
                    scores /= scores.max()
                pred = model(input)
                loss = loss_func(pred.view(-1), scores.view(-1))

                loss.backward()
                optim.step()

                pbar.set_description("loss: {:.4f}".format(loss.item()))

            acc = []

            for i in range(hiddens.shape[0] // 10 * 9, hiddens.shape[0], 512):
                with torch.no_grad():
                    input = hiddens[i:i+512, :].float().cuda()
                    acts = torch.relu((torch.matmul(input, ffn_weight))).float() # 512, 4096

                    scores = torch.matmul(acts, patterns) # 512, num_blocks, vary from 0 to 1
                    mask, labels = torch.topk(scores, k=int(num_blocks*0.2), dim=-1)
                    mask = mask > 0

                    pred = model(input)
                    pred = torch.topk(pred, k=int(num_blocks*0.2), dim=-1)[1]

                    for x, m, s in zip(pred, mask, scores):
                        if m.sum().item() == 0:
                            continue
                        x = sum([s[xx] for xx in x.cpu()]).item()
                        y = s.sum().item()
                        acc.append( x / y)

            cur_acc = np.mean(acc)
            if cur_acc > save_acc[0]:
                self.del_ckpt(save_epoch[1])
                save_acc = [cur_acc, save_acc[0]]
                save_epoch = [epoch, save_epoch[0]]
                print("input compl center acc", np.mean(acc))
                self.acc = save_acc[1]
                sys.stdout.flush()
                self.save(epoch)
            elif cur_acc > save_acc[1]:
                self.del_ckpt(save_epoch[1])
                save_acc = [save_acc[0], cur_acc]
                save_epoch = [save_epoch[0], epoch]
                print("input compl center acc", np.mean(acc))
                self.acc = save_acc[1]
                sys.stdout.flush()
                self.save(epoch)
        os.system("rm -rf {}_{}_{}".format(self.filename, self.type, save_epoch[0]))
        os.system("cp {0}_{1}_{2} {0}_{1}".format(self.filename, self.type, save_epoch[1]))
        os.system("rm {0}_{1}_{2}".format(self.filename, self.type, save_epoch[1]))

    def del_ckpt(self, epoch):
        os.system("rm -rf {}_{}_{}".format(self.filename, self.type, epoch))

    def save(self, epoch):
        print("input compl center save")
        torch.save(self.centers, "{}_{}_{}".format(self.filename, self.type, epoch))
        self.save_acc()

def bert_change_forward(model, param_split_dir, device, k=20):
    for x in model.bert.encoder.layer:
        x.intermediate.dense.bias = None

    def _forward(ffn_self, hidden_states):
        hidden_states = ffn_self.forward_old(hidden_states)

        if ffn_self.patterns is not None:
            # golden
            k = ffn_self.k
            bsz, seq_len, hidden_size = hidden_states.shape
            hidden_states_relu = hidden_states.clone()
            hidden_states_relu = hidden_states_relu.view(-1, hidden_size)
            score = torch.matmul(hidden_states_relu, ffn_self.patterns.transpose(0, 1))
            labels = torch.topk(score, k=k, dim=-1)[1].view(bsz, seq_len, k)
            cur_mask = torch.nn.functional.embedding(labels, ffn_self.patterns).sum(-2)
            hidden_states[cur_mask == False] = 0

        return hidden_states

    def modify_ffn(ffn, path):
        assert type(ffn) == BertIntermediate
        labels = torch.load(path)
        cluster_num = max(labels)+1
        patterns = []
        for i in range(cluster_num):
            patterns.append(np.array(labels) == i)
        # ffn.patterns = torch.Tensor(patterns).cuda()
        ffn.patterns = torch.Tensor(patterns).to(device)
        ffn.k = k
        ffn.forward_old = ffn.forward
        ffn.forward = types.MethodType(_forward, ffn)

    # encoder
    for layer_idx, layer in tqdm.tqdm(enumerate(model.bert.encoder.layer),total=len(model.bert.encoder.layer), desc="bert_change_forward ing"):
        ffn = layer.intermediate
        path = os.path.join(param_split_dir, 'param_split', 'bert.encoder.layer.{}.intermediate.dense.weight'.format(layer_idx))
        modify_ffn(ffn, path)


