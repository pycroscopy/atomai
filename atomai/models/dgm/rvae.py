"""
rvae.py
=======

Module for analysis of system "building blocks" with
rotationally-invariant variational autoencoders

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from copy import deepcopy as dc
from typing import Optional, Union, List

import numpy as np
import torch

from ...losses_metrics import rvae_loss
from ...utils import set_train_rng, to_onehot, transform_coordinates
from .vae import BaseVAE


class rVAE(BaseVAE):
    """
    Implements rotationally and translationally invariant
    Variational Autoencoder (VAE) based on the idea of "spatial decoder"
    by Bepler et al. in arXiv:1909.11663. In addition, this class allows
    implementating the class-conditioned VAE and skip-VAE (arXiv:1807.04863)
    with rotational and translational variance.

    Args:
        in_dim:
            Input dimensions for image data passed as (heigth, width)
            for grayscale data or (height, width, channels)
            for multichannel data
        latent_dim:
            Number of VAE latent dimensions associated with image content
        nb_classes:
            Number of classes for class-conditional rVAE
        translation:
            account for xy shifts of image content (Default: True)
        seed:
            seed for torch and numpy (pseudo-)random numbers generators
        **conv_encoder (bool):
            use convolutional layers in encoder
        **numlayers_encoder (int):
            number of layers in encoder (Default: 2)
        **numlayers_decoder (int):
            number of layers in decoder (Default: 2)
        **numhidden_encoder (int):
            number of hidden units OR conv filters in encoder (Default: 128)
        **numhidden_decoder (int):
            number of hidden units in decoder (Default: 128)
        **skip (bool):
            uses generative skip model with residual paths between
            latents and decoder layers (Default: False)

    Example:

    >>> input_dim = (28, 28)  # input dimensions
    >>> # Intitialize model
    >>> rvae = aoi.models.rVAE(input_dim)
    >>> # Train
    >>> rvae.fit(imstack_train, training_cycles=100,
                 batch_size=100, rotation_prior=np.pi/2)
    >>> rvae.manifold2d(origin="upper", cmap="gnuplot2")

    One can also pass labels to train a class-conditioned rVAE

    >>> # Intitialize model
    >>> rvae = aoi.models.rVAE(input_dim, nb_classes=10)
    >>> # Train
    >>> rvae.fit(imstack_train, labels_train, training_cycles=100,
    >>>          batch_size=100, rotation_prior=np.pi/2)
    >>> # Visualize learned manifold for class 1
    >>> rvae.manifold2d(label=1, origin="upper", cmap="gnuplot2")
    """

    def __init__(self,
                 in_dim: int = None,
                 latent_dim: int = 2,
                 nb_classes: int = 0,
                 translation: bool = True,
                 seed: int = 0,
                 **kwargs: Union[int, bool, str]
                 ) -> None:
        """
        Initializes rVAE model
        """
        coord = 3 if translation else 1  # xy translations and/or rotation
        args = (in_dim, latent_dim, nb_classes, coord)
        super(rVAE, self).__init__(*args, **kwargs)
        set_train_rng(seed)
        self.translation = translation
        self.dx_prior = None
        self.phi_prior = None
        self.kdict_ = dc(kwargs)
        self.kdict_["num_iter"] = 0

    def elbo_fn(self,
                x: torch.Tensor,
                x_reconstr: torch.Tensor,
                *args: torch.Tensor,
                **kwargs: Union[List, float, int]
                ) -> torch.Tensor:
        """
        Computes ELBO
        """
        return rvae_loss(self.loss, self.in_dim, x, x_reconstr, *args, **kwargs)

    def forward_compute_elbo(self,
                             x: torch.Tensor,
                             y: Optional[torch.Tensor] = None,
                             mode: str = "train"
                             ) -> torch.Tensor:
        """
        rVAE's forward pass with training/test loss computation
        """
        x_coord_ = self.x_coord.expand(x.size(0), *self.x_coord.size())
        if mode == "eval":
            with torch.no_grad():
                z_mean, z_logsd = self.encoder_net(x)
        else:
            z_mean, z_logsd = self.encoder_net(x)
            self.kdict_["num_iter"] += 1
        z_sd = torch.exp(z_logsd)
        z = self.reparameterize(z_mean, z_sd)
        phi = z[:, 0]  # angle
        if self.translation:
            dx = z[:, 1:3]  # translation
            dx = (dx * self.dx_prior).unsqueeze(1)
            z = z[:, 3:]  # image content
        else:
            dx = 0  # no translation
            z = z[:, 1:]  # image content

        if y is not None:
            targets = to_onehot(y, self.nb_classes)
            z = torch.cat((z, targets), -1)

        x_coord_ = transform_coordinates(x_coord_, phi, dx)
        if mode == "eval":
            with torch.no_grad():
                x_reconstr = self.decoder_net(x_coord_, z)
        else:
            x_reconstr = self.decoder_net(x_coord_, z)
        
        return self.elbo_fn(x, x_reconstr, z_mean, z_logsd, **self.kdict_)

    def fit(self,
            X_train: Union[np.ndarray, torch.Tensor],
            y_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
            X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            loss: str = "mse",
            **kwargs) -> None:
        """
        Trains rVAE model

        Args:
            X_train:
                3D or 4D stack of training images with dimensions
                (n_images, height, width) for grayscale data or
                or (n_images, height, width, channels) for multi-channel data
            y_train:
                Vector with labels of dimension (n_images,), where n_images
                is a number of training images
            X_test:
                3D or 4D stack of test images with the same dimensions
                as for the X_train (Default: None)
            y_test:
                Vector with labels of dimension (n_images,), where n_images
                is a number of test images
            loss:
                reconstruction loss function, "ce" or "mse" (Default: "mse")
            **translation_prior (float):
                translation prior
            **rotation_prior (float):
                rotational prior
            **capacity (list):
                List containing (max_capacity, num_iters, gamma) parameters
                to control the capacity of the latent channel.
                Based on https://arxiv.org/pdf/1804.03599.pdf
            **filename (str):
                file path for saving model aftereach training cycle ("epoch")
            **recording (bool):
                saves a learned 2d manifold at each training step
        """
        self._check_inputs(X_train, y_train, X_test, y_test)
        self.dx_prior = kwargs.get("translation_prior", 0.1)
        self.kdict_["phi_prior"] = kwargs.get("rotation_prior", 0.1)
        for k, v in kwargs.items():
            if k in ["capacity"]:
                self.kdict_[k] = v
        self.compile_trainer(
            (X_train, y_train), (X_test, y_test), **kwargs)
        self.loss = loss  # this part needs to be handled better
        if self.loss == "ce":
            self.sigmoid_out = True  # Use sigmoid layer for "prediction" stage
            self.metadict["sigmoid_out"] = True
        self.recording = kwargs.get("recording", False)

        for e in range(self.training_cycles):
            self.current_epoch = e
            elbo_epoch = self.train_epoch()
            self.loss_history["train_loss"].append(elbo_epoch)
            if self.test_iterator is not None:
                elbo_epoch_test = self.evaluate_model()
                self.loss_history["test_loss"].append(elbo_epoch_test)
            self.print_statistics(e)
            self.update_metadict()
            if self.recording and self.z_dim in [3, 5]:
                self.manifold2d(savefig=True, filename=str(e))
            self.save_model(self.filename)
        if self.recording and self.z_dim in [3, 5]:
            self.visualize_manifold_learning("./vae_learning")

    def update_metadict(self):
        self.metadict["num_epochs"] = self.current_epoch
        self.metadict["num_iter"] = self.kdict_["num_iter"]
