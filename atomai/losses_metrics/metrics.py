"""
metrics.py
==========

Accuracy metrics
"""

import numpy as np
import torch
import torch.nn.functional as F

from atomai.transforms import squeeze_channels
from atomai.utils import cv_thresh


class IoU:
    """
    Computes mean of the Intersection over Union.
    Adapted with changes from https://github.com/kevinzakka/pytorch-goodies

    Args:
        true: labels (aka ground truth)
        pred: model predictions
        activation: applies softmax/sigmoid to predictions
        thresh: image binary threshold level for predictions
    """
    def __init__(self,
                 true: torch.Tensor,
                 pred: torch.Tensor,
                 activation: bool = True,
                 thresh: float = 0.5):
        """
        Initializes IoU
        """
        self.thresh = thresh
        self.nb_classes = pred.shape[1]
        if activation:
            if self.nb_classes > 1:
                pred = F.softmax(pred, dim=1)
            else:
                pred = torch.sigmoid(pred)
        if self.nb_classes == 1:
            self.nb_classes += 1
        pred = pred.detach().cpu().numpy()
        true = true.detach().cpu().numpy()
        pred = self.threshold_(pred, thresh)
        if pred.ndim == 4 and pred.shape[-1] > 1:
            true, pred = squeeze_channels(true, pred, clip=True)
        pred = torch.from_numpy(pred).long()
        true = torch.from_numpy(true).long()
        if torch.cuda.is_available():
            pred = pred.cuda()
            true = true.cuda()
        self.pred = pred
        self.true = true

    @classmethod
    def threshold_(cls, xarr, thresh):
        """
        Thresholds image data
        """
        xarr = xarr.transpose(0, 2, 3, 1)
        xarr_ = np.zeros_like(xarr)
        for i, x in enumerate(xarr):
            x = cv_thresh(x, thresh)
            x = x[..., None] if x.ndim == 2 else x
            xarr_[i] = x
        return xarr_

    def compute_hist(self, true, pred):
        """
        Computes histogram for a single true-pred pair
        """
        mask = (true >= 0) & (true < self.nb_classes)
        hist = torch.bincount(
            self.nb_classes * true[mask] + pred[mask],
            minlength=self.nb_classes ** 2)
        hist = hist.reshape(self.nb_classes, self.nb_classes).float()
        return hist

    def evaluate(self):
        """
        Computes mean IoU score for a batch
        """
        hist = torch.zeros((self.nb_classes, self.nb_classes))
        if torch.cuda.is_available():
            hist = hist.cuda()
        for t, p in zip(self.true, self.pred):
            hist += self.compute_hist(t.flatten(), p.flatten())
        A_inter_B = torch.diag(hist)
        A = torch.sum(hist, dim=1)
        B = torch.sum(hist, dim=0)
        jcd = A_inter_B / (A + B - A_inter_B + 1e-10)
        avg_jcd = torch.mean(jcd[jcd == jcd])
        return avg_jcd.item()
