"""
imspec.py
=========

Encoder and decoder modules for im2spec and spec2im models

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)

"""

from typing import Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import convblock, dilated_block


class signal_encoder(nn.Module):
    """
    Encodes 1D/2D signal into a latent vector
    """
    def __init__(self, signal_dim: Tuple[int],
                 z_dim: int, nb_layers: int, nb_filters: int,
                 ) -> None:
        """
        Initialize NN parameters
        """
        super(signal_encoder, self).__init__()
        if isinstance(signal_dim, int):
            signal_dim = (signal_dim,)
        if not 0 < len(signal_dim) < 3:
            raise AssertionError("signal dimensionality must be to 1D or 2D")
        ndim = 2 if len(signal_dim) == 2 else 1
        n = np.product(signal_dim)
        self.reshape_ = nb_filters * n
        self.conv = convblock(
            ndim, nb_layers, 1, nb_filters,
            lrelu_a=0.1, use_batchnorm=True)
        self.fc = nn.Linear(nb_filters * n, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embeddes the input signal into a latent vector
        """
        x = self.conv(x)
        x = x.reshape(-1, self.reshape_)
        return self.fc(x)


class signal_decoder(nn.Module):
    """
    Decodes a ltent vector into 1D/2D signal
    """
    def __init__(self, signal_dim: Tuple[int],
                 z_dim: int, nb_layers: int, nb_filters: int,
                 **kwargs: bool) -> None:
        """
        """
        super(signal_decoder, self).__init__()
        self.upsampling = kwargs.get("upsampling", False)
        if isinstance(signal_dim, int):
            signal_dim = (signal_dim,)
        if not 0 < len(signal_dim) < 3:
            raise AssertionError("signal dimensionality must be to 1D or 2D")
        ndim = 2 if len(signal_dim) == 2 else 1
        if self.upsampling:
            signal_dim = [s // 4 for s in signal_dim]
        n = np.product(signal_dim)
        self.reshape_ = (nb_filters, *signal_dim)
        self.fc = nn.Linear(z_dim, nb_filters*n)
        if self.upsampling:
            self.deconv1 = convblock(
                ndim, 1, nb_filters, nb_filters,
                lrelu_a=0.1, use_batchnorm=True)
            self.deconv2 = convblock(
                ndim, 1, nb_filters, nb_filters,
                lrelu_a=0.1, use_batchnorm=True)
        self.dilblock = dilated_block(
            ndim, nb_filters, nb_filters,
            dilation_values=torch.arange(1, nb_layers + 1).tolist(),
            padding_values=torch.arange(1, nb_layers + 1).tolist(),
            lrelu_a=0.1, use_batchnorm=True)
        self.conv = convblock(
            ndim, 1, nb_filters, 1,
            lrelu_a=0.1, use_batchnorm=True)
        if ndim == 2:
            self.out = nn.Conv2d(1, 1, 1)
        else:
            self.out = nn.Conv1d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates a signal from embedded features (latent vector)
        """
        x = self.fc(x)
        x = x.reshape(-1, *self.reshape_)
        if self.upsampling:
            x = self.deconv1(x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.deconv2(x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.dilblock(x)
        x = self.conv(x)
        return self.out(x)


class signal_ed(nn.Module):
    """
    Transforms image into spectra (im2spec) and vice versa (spec2im)
    """
    def __init__(self, feature_dim: Tuple[int],
                 target_dim: Tuple[int], latent_dim: int,
                 *args: Type[nn.Module], **kwargs: Union[int, bool]
                 ) -> None:
        """
        Initializes im2spec/spec2im parameters
        """
        super(signal_ed, self).__init__()
        numlayers_e = kwargs.get("numlayers_encoder", 2)
        numlayers_d = kwargs.get("numlayers_decoder", 2)
        numfilters_e = kwargs.get("numhidden_encoder", 64)
        numfilters_d = kwargs.get("numhidden_decoder", 64)
        self.signal_encoder = signal_encoder(
            feature_dim, latent_dim, numlayers_e, numfilters_e)
        self.signal_decoder = signal_decoder(
            target_dim, latent_dim, numlayers_d, numfilters_d, **kwargs)

    def encode_signal(self, features: torch.Tensor) -> torch.Tensor:
        """
        Embeddes the input image into a latent vector
        """
        return self.signal_encoder(features)

    def decode_signal(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Generates signal from the embedded features
        """
        return self.signal_decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        x = self.encode_signal(x)
        return self.decode_signal(x)
