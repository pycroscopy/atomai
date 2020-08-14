"""
metrics.py
==========

Accuracy metrics
"""

from typing import Union

import numpy as np
import torch

from atomai.transforms import squeeze_channels
from atomai.utils import cv_thresh


class IoU:
    """
    Computes mean of the Intersection over Union.
    Adapted with changes from https://github.com/kevinzakka/pytorch-goodies
    """
    def __init__(self,
                 true: Union[torch.Tensor, np.ndarray],
                 pred: Union[torch.Tensor, np.ndarray],
                 nb_classes: int,
                 thresh: float = 0.5):

        self.thresh = thresh
        self.nb_classes = nb_classes
        if self.nb_classes == 1:
            self.nb_classes += 1
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(true, torch.Tensor):
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
            x -= x.min()
            x /= x.ptp()
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
