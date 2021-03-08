"""
jrvae.py
=======

Module for analysis of system "building blocks" with rotationally-invariant
variational autoencoders for joint continuous and discrete representations

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Optional, Union, List

from copy import deepcopy as dc
import numpy as np
import torch

from ...losses_metrics import joint_rvae_loss

from ...utils import set_train_rng, to_onehot, transform_coordinates
from .vae import BaseVAE


class jrVAE(BaseVAE):
    """
    Rotationally-invariant VAE for joint continuous and
    discrete latent representations.

    Args:
        in_dim:
            Input dimensions for image data passed as (heigth, width)
            for grayscale data or (height, width, channels)
            for multichannel data
        latent_dim:
            Number of latent dimensions associated with image content
        discrete_dim:
            List specifying dimensionalities of discrete (Gumbel-Softmax)
            latent variables associated with image content
        nb_classes:
            Number of classes for class-conditional VAE.
            (leave it at 0 to learn discrete latent reprenetations)
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

    >>> input_dim = (28, 28)  # intput dimensions
    >>> # Intitialize model
    >>> jrvae = aoi.models.jVAE(input_dim, latent_dim=2, discrete_dim=[10],
    >>>                         numlayers_encoder=3, numhidden_encoder=512,
    >>>                         numlayers_decoder=3, numhidden_decoder=512)
    >>> # Train
    >>> jrvae.fit(imstack_train, training_cycles=100,
                  batch_size=100, rotation_prior=np.pi/4)
    >>> jrvae.manifold2d(origin="upper", cmap="gnuplot2")

    """
    def __init__(self,
                 in_dim: int = None,
                 latent_dim: int = 2,
                 discrete_dim: List[int] = [2],
                 nb_classes: int = 0,
                 translation: bool = True,
                 seed: int = 0,
                 **kwargs: Union[int, bool, str]
                 ) -> None:
        """
        Initializes joint rVAE model (jrVAE)
        """
        coord = 3 if translation else 1  # xy translations and/or rotation
        args = (in_dim, latent_dim, nb_classes, coord, discrete_dim)
        super(jrVAE, self).__init__(*args, **kwargs)
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
                **kwargs: Union[List, int]
                ) -> torch.Tensor:
        """
        Computes ELBO
        """
        return joint_rvae_loss(self.loss, self.in_dim, x, x_reconstr, *args, **kwargs)

    def forward_compute_elbo(self,
                             x: torch.Tensor,
                             y: Optional[torch.Tensor] = None,
                             mode: str = "train"
                             ) -> torch.Tensor:
        """
        Joint rVAE's forward pass with training/test loss computation
        """
        tau = self.kdict_.get("temperature", .67)
        x_coord_ = self.x_coord.expand(x.size(0), *self.x_coord.size())
        x = x.to(self.device)
        if mode == "eval":
            with torch.no_grad():
                latent_ = self.encoder_net(x)
        else:
            latent_ = self.encoder_net(x)
            self.kdict_["num_iter"] += 1

        z_mean, z_logsd = latent_[:2]
        z_sd = torch.exp(z_logsd)
        z_cont = self.reparameterize(z_mean, z_sd)
        phi = z_cont[:, 0]  # angle
        if self.translation:
            dx = z_cont[:, 1:3]  # translation
            dx = (dx * self.dx_prior).unsqueeze(1)
            z_cont = z_cont[:, 3:]  # image content
        else:
            dx = 0  # no translation
            z_cont = z_cont[:, 1:]  # image content
        x_coord_ = transform_coordinates(x_coord_, phi, dx)

        alphas = latent_[2:]
        z_disc = [self.reparameterize_discrete(a, tau) for a in alphas]
        z_disc = torch.cat(z_disc, 1)

        z = torch.cat((z_cont, z_disc), dim=1)

        if y is not None:
            targets = to_onehot(y, self.nb_classes)
            z = torch.cat((z, targets), -1)

        if mode == "eval":
            with torch.no_grad():
                x_reconstr = self.decoder_net(x_coord_, z)
        else:
            x_reconstr = self.decoder_net(x_coord_, z)

        return self.elbo_fn(x, x_reconstr, z_mean, z_logsd, alphas, **self.kdict_)

    def fit(self,
            X_train: Union[np.ndarray, torch.Tensor],
            y_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
            X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            loss: str = "mse",
            **kwargs) -> None:
        """
        Trains joint rVAE model

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
            **temperature (float):
                Relaxation parameter for Gumbel-Softmax distribution
            **cont_capacity (list):
                List containing (max_capacity, num_iters, gamma) parameters
                to control the capacity of the continuous latent channel.
                Default values: [5.0, 25000, 30].
                Based on https://arxiv.org/pdf/1804.03599.pdf & https://arxiv.org/abs/1804.00104
            **disc_capacity (list):
                List containing (max_capacity, num_iters, gamma) parameters
                to control the capacity of the discrete latent channel(s).
                Default values: [5.0, 25000, 30].
                Based on https://arxiv.org/pdf/1804.03599.pdf & https://arxiv.org/abs/1804.00104
            **filename (str):
                file path for saving model after each training cycle ("epoch")
        """
        self._check_inputs(X_train, y_train, X_test, y_test)
        self.dx_prior = kwargs.get("translation_prior", 0.1)
        self.kdict_["phi_prior"] = kwargs.get("rotation_prior", 0.1)
        for k, v in kwargs.items():
            if k in ["cont_capacity", "disc_capacity", "temperature"]:
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
            self.save_model(self.filename)

    def update_metadict(self):
        self.metadict["num_epochs"] = self.current_epoch
        self.metadict["num_iter"] = self.kdict_["num_iter"]
