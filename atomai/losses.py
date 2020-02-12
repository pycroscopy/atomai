# TODO Add focal loss
"""
Custom pytorch loss functions
"""

import torch
import torch.nn.functional as F


class dice_loss(torch.nn.Module):
    """
    Computes the Sørensen–Dice loss.
    Adapted with changes from https://github.com/kevinzakka/pytorch-goodies
    """
    def __init__(self, eps=1e-7):
        super(dice_loss, self).__init__()
        self.eps = eps

    def forward(self, logits, labels):
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[labels.squeeze(1).to(torch.int64)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else: # performance on multi-class case has not been tested yet
            true_1_hot = torch.eye(num_classes)[labels.squeeze(1).to(torch.int64)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)

        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, labels.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)

