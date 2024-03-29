#
# Training Deep Neural Networks on Noisy Labels with Bootstrapping
# http://www-personal.umich.edu/~reedscot/bootstrap.pdf
#

import torch
from torch.nn import Module
import torch.nn.functional as F


class SoftBootstrappingLoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * p) * log(p)``

    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.
        as_pseudo_label (bool): Stop gradient propagation for the term ``(1 - beta) * p``.
            Can be interpreted as pseudo-label.
    """
    def __init__(self, beta=0.95, reduce=True, as_pseudo_label=True):
        super(SoftBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce
        self.as_pseudo_label = as_pseudo_label

    def forward(self, y_pred, y):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')

        y_pred_a = y_pred.detach() if self.as_pseudo_label else y_pred
        # second term = - (1 - beta) * p * log(p)
        bootstrap = - (1.0 - self.beta) * torch.sum(F.softmax(y_pred_a, dim=1) * F.log_softmax(y_pred, dim=1), dim=1)

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap


class CustomCrossEntropyLoss(Module):
    def __init__(self, vocab_weights, tokenizer):
        super(CustomCrossEntropyLoss, self).__init__()
        self.vocab_weights = vocab_weights
        self.tokenizer = tokenizer

    def forward(self, y_pred, y, input_ids):
        loss = F.cross_entropy(y_pred, y, reduction='none')
        # Apply vocab_weights to specific words in the input sequences
        for word, weight in self.vocab_weights.items():
            word_id = self.tokenizer.convert_tokens_to_ids(word)
            weight_mask = (input_ids == word_id).float()
            weight_mask *= (weight - 1)
            loss += weight_mask.sum(dim=-1)
            continue
        return loss.mean()


class HardBootstrappingLoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)``
    where ``z = argmax(p)``

    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.

    """
    def __init__(self, beta=0.8, reduce=True):
        super(HardBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, y_pred, y):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')

        # z = argmax(p)
        z = F.softmax(y_pred.detach(), dim=1).argmax(dim=1)
        z = z.view(-1, 1)
        bootstrap = F.log_softmax(y_pred, dim=1).gather(1, z).view(-1)
        # second term = (1 - beta) * z * log(p)
        bootstrap = - (1.0 - self.beta) * bootstrap

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2
    return loss


