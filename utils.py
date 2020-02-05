"""Utility functions"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_model(model, weights_path):
    '''Loads weights saved in a pytorch format (.pt) into a model skeleton'''
    if torch.cuda.device_count() > 0:
        checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


def img_resize(image_data, rs):
    '''Image resizing'''
    if image_data.shape[1:3] == rs:
        return image_data.copy()
    image_data_r = np.zeros(
        (image_data.shape[0], rs[0], rs[1]))
    for i, img in enumerate(image_data):
        img = cv2.resize(img, (rs[0], rs[1]))
        image_data_r[i, :, :] = img
    return image_data_r


def img_pad(image_data, pooling):
    '''Pads the image if its size (w, h)
    is not divisible by 2**n, where n is a number
    of pooling layers in a network'''
    # Pad image rows (height)
    while image_data.shape[1] % pooling != 0:
        d0, _, d2 = image_data.shape
        image_data = np.concatenate(
            (image_data, np.zeros((d0, 1, d2))), axis=1)
    # Pad image columns (width)
    while image_data.shape[2] % pooling != 0:
        d0, d1, _ = image_data.shape
        image_data = np.concatenate(
            (image_data, np.zeros((d0, d1, 1))), axis=2)
    return image_data


def torch_format(image_data):
    '''Reshapes and normalizes (optionally) image data
    to make it compatible with pytorch format'''
    image_data = np.expand_dims(image_data, axis=1)
    image_data = (image_data - np.amin(image_data))/np.ptp(image_data)
    image_data = torch.from_numpy(image_data).float()
    return image_data


class Hook():
    """
    Returns the input and output of a
    layer during forward/backward pass
    see https://www.kaggle.com/sironghuang/
        understanding-pytorch-hooks/notebook
    """
    def __init__(self, module, backward=False):
        """
        Args:
            module: torch modul(single layer or sequential block)
            backward (bool): replace forward_hook with backward_hook
        """
        if backward is False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def mock_forward(model, dims=(1, 64, 64)):
    '''Passes a dummy variable throuh a network'''
    x = torch.randn(1, dims[0], dims[1], dims[2])
    if next(model.parameters()).is_cuda:
        x = x.cuda()
    out = model(x)
    return out

def cv_thresh(imgdata, threshold):
    """Wrapper for opencv binary threshold method"""
    _, thresh = cv2.threshold(
                    imgdata, 
                    threshold, 1, 
                    cv2.THRESH_BINARY)
    return thresh


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

def plot_coord(img, coord):
    """Plots coordinates (colored according to atom class)"""
    y, x, c = coord.T
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.scatter(x, y, c=c, cmap='RdYlGn', s=8)
    plt.show()