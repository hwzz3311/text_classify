import torch
import os
from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt


# 量化权重
def signed_quantize(x, bits, bias=None):
    min_val, max_val = x.min(), x.max()
    n = 2.0 ** (bits -1)
    scale = max(abs(min_val), abs(max_val)) / n
    qx = torch.floor(x / scale)
    if bias is not None:
        qb = torch.floor(bias / scale)
        return qx, qb
    else:
        return qx

# 对模型整体进行量化
def scale_quant_model(model, bits):
    net = deepcopy(model)
    params_quant = OrderedDict()
    params_save = OrderedDict()

    for k, v in model.state_dict().items():
        if str(k) == "bert.embeddings.position_ids":
            print("")
        if 'classifier' not in k and 'num_batches' not in k and 'running' not in k:
            if 'weight' in k:
                weight = v
                bias_name = k.replace('weight', 'bias')
                try:
                    bias = model.state_dict()[bias_name]
                    w, b = signed_quantize(weight, bits, bias)
                    params_quant[k] = w
                    params_quant[bias_name] = b
                    if bits > 8 and bits <= 16:
                        params_save[k] = w.short()
                        params_save[bias_name] = b.short()
                    elif bits >1 and bits <= 8:
                        params_save[k] = w.char()
                        params_save[bias_name] = b.char()
                    elif bits == 1:
                        params_save[k] = w.bool()
                        params_save[bias_name] = b.bool()

                except:
                    w = signed_quantize(weight, bits)
                    params_quant[k] = w
                    params_save[k] = w.char()
            else:
                params_quant[k] = v
                params_save[k] = v
        else:
            params_quant[k] = v
            params_save[k] = v
    net.load_state_dict(params_quant)
    return net, params_save


if __name__ == '__main__':
    pass
    # pruned = False
    # if pruned:
    #     channels = [17, 'M', 77, 'M', 165, 182, 'M', 338, 337, 'M', 360, 373, 'M']
    #     net = VGG_11_prune(channels).to(device)
    #     net.load_state_dict(
    #         torch.load(
    #             os.path.join(checkpoint, 'best_retrain_model.pth'))['compressed_net'])
    # else:
    #     net = VGG_11_prune().to(device)
    #     net.load_state_dict(
    #         torch.load(
    #             os.path.join(checkpoint, 'best_model.pth'), map_location=torch.device('cpu')
    #         )['net']
    #     )
    #
    # validation(net, torch.nn.CrossEntropyLoss())
    #
    # accuracy_list = []
    # bit_list = [16, 12, 8, 6, 4, 3, 2, 1]
    # for bit in bit_list:
    #     print('{} bit'.format(bit))
    #     scale_quantized_model, params = scale_quant_model(net, bit)
    #     print('validation: ', end='\t')
    #     accuracy, _ = validation(scale_quantized_model, torch.nn.CrossEntropyLoss())
    #     accuracy_list.append(accuracy)
    #     torch.save(params,
    #                os.path.join(checkpoint, 'pruned_{}_{}_bits.pth'.format(pruned, bit)))
    #
    # plt.plot(bit_list, accuracy_list)
    # plt.savefig('img/quantize_pruned:{}.jpg'.format(pruned))
    # plt.show()
