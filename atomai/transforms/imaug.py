"""
imaug.py
========

Module for image transformations relevant to data augmentation

Created by Maxim Ziatdinov (maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Optional, Callable, Union, List, Tuple

import numpy as np
import torch
import cv2
from scipy import stats, ndimage
from skimage import exposure
from skimage.util import random_noise


class datatransform:
    """
    Applies a sequence of pre-defined operations for data augmentation.

    Args:
        n_channels (int):
            Number of classes (channels) in the ground truth
        dim_order_in (str):
            Channel first or channel last ordering in the input masks
        dim_order_out (str):
            Channel first or channel last ordering in the output masks
        seed (int):
            Determenism
        **custom_transform (Callable):
            Python function that takes two ndarrays (images and masks) as
            input, applies a set of transformation to them, and returns the two
            transformed arrays
        **rotation (bool):
            Rotating image by +- 90 deg (if image is square)
            and horizontal/vertical flipping.
        **zoom (bool or int):
            Zooming-in by a specified zoom factor (Default: 2)
            Note that a zoom window is always square
        **gauss_noise (bool or list ot tuple):
            Gaussian noise. You can pass min and max values as a list/tuple
            (Default [min, max] range: [0, 50])
        **poisson_noise (bool or list ot tuple):
            Poisson noise. You can pass min and max values as a list/tuple
            (Default [min, max] range: [30, 40])
        **salt_and_pepper (bool or list ot tuple):
            Salt and pepper noise. You can pass min and max values as a list/tuple
            (Default [min, max] range: [0, 50])
        **blur (bool or list ot tuple):
            Gaussian blurring. You can pass min and max values as a list/tuple
            (Default [min, max] range: [1, 50])
        **contrast (bool or list ot tuple):
            Contrast level. You can pass min and max values as a list/tuple
            (Default [min, max] range: [5, 20])
        **background (bool):
            Adds/substracts asymmetric 2D gaussian of random width and intensity
            from the image
        **resize (tuple):
            Values for image resizing
            [downscale factor (default: 2), upscale factor (default:1.5)]
    """
    def __init__(self,
                 n_channels: int = None,
                 dim_order_in: str = 'channel_last',
                 dim_order_out: str = 'channel_first',
                 squeeze_channels: bool = False,
                 seed: Optional[int] = None,
                 **kwargs: Union[bool, Callable, List, Tuple]) -> None:
        """
        Initializes image transformation parameters
        """
        self.ch = n_channels
        self.dim_order_in = dim_order_in
        self.dim_order_out = dim_order_out
        self.squeeze = squeeze_channels
        self.custom_transform = kwargs.get('custom_transform')
        self.rotation = kwargs.get('rotation')
        self.background = kwargs.get('background')
        self.gauss = kwargs.get('gauss_noise')
        if self.gauss is True:
            self.gauss = [0, 50]
        self.jitter = kwargs.get('jitter')
        if self.jitter is True:
            self.jitter = [0, 50]
        self.poisson = kwargs.get('poisson_noise')
        if self.poisson is True:
            self.poisson = [30, 40]
        self.salt_and_pepper = kwargs.get('salt_and_pepper')
        if self.salt_and_pepper is True:
            self.salt_and_pepper = [0, 50]
        self.blur = kwargs.get('blur')
        if self.blur is True:
            self.blur = [1, 50]
        self.contrast = kwargs.get('contrast')
        if self.contrast is True:
            self.contrast = [5, 20]
        self.zoom = kwargs.get('zoom')
        if self.zoom is True:
            self.zoom = 2
        self.resize = kwargs.get('resize')
        if self.resize is True:
            self.resize = [2, 1.5]
        if seed is not None:
            np.random.seed(seed)

    def apply_gauss(self,
                    X_batch: np.ndarray,
                    y_batch: np.ndarray) -> Tuple[np.ndarray]:
        """
        Random application of gaussian noise to each training inage in a stack
        """
        n, h, w = X_batch.shape[0:3]
        X_batch_noisy = np.zeros((n, h, w))
        for i, img in enumerate(X_batch):
            gauss_var = np.random.randint(self.gauss[0], self.gauss[1])
            X_batch_noisy[i] = random_noise(
                img, mode='gaussian', var=1e-4*gauss_var)
        return X_batch_noisy, y_batch

    def apply_jitter(self,
                     X_batch: np.ndarray,
                     y_batch: np.ndarray) -> Tuple[np.ndarray]:
        """
        Random application of jitter noise to each training image in a stack
        """
        n, h, w = X_batch.shape[0:3]
        X_batch_noisy = np.zeros((n, h, w))
        for i, img in enumerate(X_batch):
            jitter_amount = np.random.randint(self.jitter[0], self.jitter[1]) / 10
            shift_arr = stats.poisson.rvs(jitter_amount, loc=0, size=h)
            X_batch_noisy[i] = np.array([np.roll(row, z) for row, z in zip(img, shift_arr)])
        return X_batch_noisy, y_batch

    def apply_poisson(self,
                      X_batch: np.ndarray,
                      y_batch: np.ndarray) -> Tuple[np.ndarray]:
        """
        Random application of poisson noise to each training inage in a stack
        """
        def make_pnoise(image, l):
            vals = len(np.unique(image))
            vals = (50/l) ** np.ceil(np.log2(vals))
            image_n_filt = np.random.poisson(image * vals) / float(vals)
            return image_n_filt
        n, h, w = X_batch.shape[0:3]
        X_batch_noisy = np.zeros((n, h, w))
        for i, img in enumerate(X_batch):
            poisson_l = np.random.randint(self.poisson[0], self.poisson[1])
            X_batch_noisy[i] = make_pnoise(img, poisson_l)
        return X_batch_noisy, y_batch

    def apply_sp(self,
                 X_batch: np.ndarray,
                 y_batch: np.ndarray) -> Tuple[np.ndarray]:
        """
        Random application of salt & pepper noise to each training inage in a stack
        """
        n, h, w = X_batch.shape[0:3]
        X_batch_noisy = np.zeros((n, h, w))
        for i, img in enumerate(X_batch):
            sp_amount = np.random.randint(
                self.salt_and_pepper[0], self.salt_and_pepper[1])
            X_batch_noisy[i] = random_noise(img, mode='s&p', amount=sp_amount*1e-3)
        return X_batch_noisy, y_batch

    def apply_blur(self,
                   X_batch: np.ndarray,
                   y_batch: np.ndarray) -> Tuple[np.ndarray]:
        """
        Random blurring of each training image in a stack
        """
        n, h, w = X_batch.shape[0:3]
        X_batch_noisy = np.zeros((n, h, w))
        for i, img in enumerate(X_batch):
            blur_amount = np.random.randint(self.blur[0], self.blur[1])
            X_batch_noisy[i] = ndimage.filters.gaussian_filter(img, blur_amount*5e-2)
        return X_batch_noisy, y_batch

    def apply_contrast(self,
                       X_batch: np.ndarray,
                       y_batch: np.ndarray) -> Tuple[np.ndarray]:
        """
        Randomly change level of contrast of each training image on a stack
        """
        n, h, w = X_batch.shape[0:3]
        X_batch_noisy = np.zeros((n, h, w))
        for i, img in enumerate(X_batch):
            clevel = np.random.randint(self.contrast[0], self.contrast[1])
            X_batch_noisy[i] = exposure.adjust_gamma(img, clevel/10)
        return X_batch_noisy, y_batch

    def apply_zoom(self,
                   X_batch: np.ndarray,
                   y_batch: np.ndarray) -> Tuple[np.ndarray]:
        """
        Zoom-in achieved by cropping image and then resizing
        to the original size. The zooming window is a square.
        """
        n, h, w = X_batch.shape[0:3]
        shortdim = min([w, h])
        zoom_values = np.arange(int(shortdim // self.zoom), shortdim + 8, 8)
        zoom_values = zoom_values[zoom_values <= shortdim]
        X_batch_z = np.zeros((n, shortdim, shortdim))
        y_batch_z = np.zeros((n, shortdim, shortdim, self.ch))
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            zv = np.random.choice(zoom_values)
            img = img[
                (h // 2) - (zv // 2): (h // 2) + (zv // 2),
                (w // 2) - (zv // 2): (w // 2) + (zv // 2)]
            gt = gt[
                (h // 2) - (zv // 2): (h // 2) + (zv // 2),
                (w // 2) - (zv // 2): (w // 2) + (zv // 2)]
            img = cv2.resize(
                img, (shortdim, shortdim), interpolation=cv2.INTER_CUBIC)
            gt = cv2.resize(
                gt, (shortdim, shortdim), interpolation=cv2.INTER_CUBIC)
            img = np.clip(img, 0, 1)
            gt = np.around(gt)
            if len(gt.shape) != 3:
                gt = np.expand_dims(gt, axis=2)
            X_batch_z[i] = img
            y_batch_z[i] = gt
        return X_batch_z, y_batch_z

    def apply_background(self,
                         X_batch: np.ndarray,
                         y_batch: np.ndarray) -> Tuple[np.ndarray]:
        """
        Emulates thickness variation in STEM or height variation in STM
        """
        def gauss2d(xy, x0, y0, a, b, fwhm):
            return np.exp(-np.log(2)*(a*(xy[0]-x0)**2 + b*(xy[1]-y0)**2) / fwhm**2)
        n, h, w = X_batch.shape[0:3]
        X_batch_b = np.zeros((n, h, w))
        x, y = np.meshgrid(
            np.linspace(0, h, h), np.linspace(0, w, w), indexing='ij')
        for i, img in enumerate(X_batch):
            x0 = np.random.randint(0, h - h // 4)
            y0 = np.random.randint(0, w - w // 4)
            a, b = np.random.randint(10, 20, 2) / 10
            fwhm = np.random.randint(min([h, w]) // 4, min([h, w]) - min([h, w]) // 2)
            Z = gauss2d([x, y], x0, y0, a, b, fwhm)
            img = img + 0.05 * np.random.randint(-10, 10) * Z
            X_batch_b[i] = img
        return X_batch_b, y_batch

    def apply_rotation(self,
                       X_batch: np.ndarray,
                       y_batch: np.ndarray) -> Tuple[np.ndarray]:
        """
        Flips and rotates training images and correponding ground truth images
        """
        n, h, w = X_batch.shape[0:3]
        X_batch_r = np.zeros((n, h, w))
        y_batch_r = np.zeros((n, h, w, self.ch))
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            flip_type = np.random.randint(-1, 3)
            if flip_type == 3 and h == w:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            elif flip_type == 2 and h == w:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                img = cv2.flip(img, flip_type)
                gt = cv2.flip(gt, flip_type)
            if len(gt.shape) != 3:
                gt = np.expand_dims(gt, axis=2)
            X_batch_r[i] = img
            y_batch_r[i] = gt
        return X_batch_r, y_batch_r

    def apply_imresize(self,
                       X_batch: np.ndarray,
                       y_batch: np.ndarray) -> Tuple[np.ndarray]:
        """
        Resizes training images and corresponding ground truth images
        """
        rs_factor_d = 1 / self.resize[0]
        rs_factor_u = self.resize[1]
        n, h, w = X_batch.shape[0:3]
        s, p = 0.03, 8
        while (np.round((h * s), 7) % p != 0
               and np.round((w * s), 7) % p != 0):
            s += 1e-5
        rs_h = (np.arange(rs_factor_d, rs_factor_u, s) * h).astype(np.int64)
        rs_w = (np.arange(rs_factor_d, rs_factor_u, s) * w).astype(np.int64)
        rs_idx = np.random.randint(len(rs_h))
        if X_batch.shape[1:3] == (rs_h[rs_idx], rs_w[rs_idx]):
            return X_batch, y_batch
        X_batch_r = np.zeros((n, rs_h[rs_idx], rs_w[rs_idx]))
        y_batch_r = np.zeros((n, rs_h[rs_idx], rs_w[rs_idx], self.ch))
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            rs_method = cv2.INTER_AREA if rs_h[rs_idx] < h else cv2.INTER_CUBIC
            img = cv2.resize(img, (rs_w[rs_idx], rs_h[rs_idx]), rs_method)
            gt = cv2.resize(gt, (rs_w[rs_idx], rs_h[rs_idx]), rs_method)
            gt = np.around(gt)
            if len(gt.shape) < 3:
                gt = np.expand_dims(gt, axis=-1)
            X_batch_r[i] = img
            y_batch_r[i] = gt
        return X_batch_r, y_batch_r

    def run(self, images: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray]:
        """
        Applies a sequence of augmentation procedures to images
        and (except for noise) targets. Starts with user defined
        custom_transform if available. Then proceeds with
        rotation->zoom->resize->gauss->jitter->poisson->sp->blur->contrast->background.
        The operations that are not specified in kwargs are skipped.
        """
        same_dim = images.ndim + 1 == targets.ndim == 4 and self.ch is not None
        if self.dim_order_in == 'channel_first' and same_dim:
            targets = np.transpose(targets, [0, 2, 3, 1])
        elif self.dim_order_in == 'channel_last':
            pass
        else:
            raise NotImplementedError("Use 'channel_first' or 'channel_last'")
        images = (images - images.min()) / images.ptp()
        if self.custom_transform is not None:
            images, targets = self.custom_transform(images, targets)
        if self.rotation and same_dim:
            images, targets = self.apply_rotation(images, targets)
        if self.zoom and same_dim:
            images, targets = self.apply_zoom(images, targets)
        if isinstance(self.resize, list) or isinstance(self.resize, tuple):
            if same_dim:
                images, targets = self.apply_imresize(images, targets)
        if isinstance(self.gauss, list) or isinstance(self.gauss, tuple):
            images, targets = self.apply_gauss(images, targets)
        if isinstance(self.jitter, list) or isinstance(self.jitter, tuple):
            images, targets = self.apply_jitter(images, targets)
        if isinstance(self.poisson, list) or isinstance(self.poisson, tuple):
            images, targets = self.apply_poisson(images, targets)
        if isinstance(self.salt_and_pepper, list) or isinstance(self.salt_and_pepper, tuple):
            images, targets = self.apply_sp(images, targets)
        if isinstance(self.blur, list) or isinstance(self.blur, tuple):
            images, targets = self.apply_blur(images, targets)
        if isinstance(self.contrast, list) or isinstance(self.contrast, tuple):
            images, targets = self.apply_contrast(images, targets)
        if self.background:
            images, targets = self.apply_background(images, targets)
        if self.squeeze and same_dim:
            images, targets = squeeze_channels(images, targets)
        if self.dim_order_out == 'channel_first':
            images = np.expand_dims(images, axis=1)
            if same_dim:
                if self.squeeze is None or self.ch == 1:
                    targets = np.transpose(targets, (0, 3, 1, 2))
        elif self.dim_order_out == 'channel_last':
            images = np.expand_dims(images, axis=3)
        else:
            raise NotImplementedError("Use 'channel_first' or 'channel_last'")
        images = (images - images.min()) / images.ptp()
        return images, targets


def squeeze_channels(images: np.ndarray,
                     labels: np.ndarray,
                     clip: bool = False) -> Tuple[np.ndarray]:
    """
    Squeezes channels in each training image and
    filters out image-label pairs where some pixels have multiple values.
    As a result the number of image-label-pairs returned may be different
    from the number of image-label pairs in the original data.
    """

    def squeeze_channels_(label):
        """
        Squeezes multiple channel into a single channel for a single label
        """
        label_ = np.zeros((1, label.shape[0], label.shape[1]))
        for c in range(label.shape[-1]):
            label_ += label[:, :, c] * c
        return label_

    if labels.shape[-1] == 1:
        return images, labels
    images_valid, labels_valid = [], []
    for label, image in zip(labels, images):
        label = squeeze_channels_(label)
        if clip:
            label[label > labels.shape[-1] - 1] = 0
            labels_valid.append(label)
            images_valid.append(image[None, ...])
        else:
            if len(np.unique(label)) == labels.shape[-1]:
                labels_valid.append(label)
                images_valid.append(image[None, ...])
    return np.concatenate(images_valid), np.concatenate(labels_valid)


def unsqueeze_channels(labels: np.ndarray, n_channels: int) -> np.ndarray:
    """
    Separates pixels with different values into different channels
    """
    if n_channels == 1:
        return labels
    labels_ = np.eye(n_channels)[labels.astype(int)]
    return np.transpose(labels_, [0, 3, 1, 2])


def seg_augmentor(nb_classes: int,
                  **kwargs
                  ) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

    auglist = ["custom_transform", "zoom", "gauss_noise", "jitter",
               "poisson_noise", "contrast", "salt_and_pepper", "blur",
               "resize", "rotation", "background"]
    augdict = {k: kwargs[k] for k in auglist if k in kwargs.keys()}
    if len(augdict) == 0:
        return

    def augmentor(images, labels, seed):
        images = images.cpu().numpy().astype(np.float64)
        labels = labels.cpu().numpy().astype(np.float64)
        dt = datatransform(
                nb_classes, "channel_first", 'channel_first',
                True, seed, **augdict)
        images, labels = dt.run(
            images[:, 0, ...], unsqueeze_channels(labels, nb_classes))
        images = torch.from_numpy(images).float()
        if nb_classes == 1:
            labels = torch.from_numpy(labels).float()
        else:
            labels = torch.from_numpy(labels).long()
        return images, labels

    return augmentor


def imspec_augmentor(in_dim: Tuple[int],
                     out_dim: Tuple[int],
                     **kwargs
                     ) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    auglist = ["custom_transform", "gauss_noise", "jitter",
               "poisson_noise", "contrast", "salt_and_pepper", "blur",
               "background"]
    augdict = {k: kwargs[k] for k in auglist if k in kwargs.keys()}
    if len(augdict) == 0:
        return
    if len(in_dim) < len(out_dim):
        raise NotImplementedError("The built-in data augmentor works only" +
                                  " for img->spec models (i.e. input is image)")

    def augmentor(features, targets, seed):
        features = features.cpu().numpy().astype(np.float64)
        targets = targets.cpu().numpy().astype(np.float64)
        dt = datatransform(seed, **augdict)
        features, targets = dt.run(features[:, 0, ...], targets)
        features = torch.from_numpy(features).float()
        targets = torch.from_numpy(targets).float()
        return features, targets

    return augmentor
