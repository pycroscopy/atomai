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
from .blocks import ConvBlock, DilatedBlock


class signal_encoder(nn.Module):
    """
    Encodes 1D/2D signal into a latent vector
    """
    def __init__(self, signal_dim: Tuple[int],
                 z_dim: int, nb_layers: int, nb_filters: int,
                 **kwargs: int) -> None:
        """
        Initialize NN parameters
        """
        super(signal_encoder, self).__init__()
        if isinstance(signal_dim, int):
            signal_dim = (signal_dim,)
        if not 0 < len(signal_dim) < 3:
            raise AssertionError("signal dimensionality must be to 1D or 2D")
        ndim = 2 if len(signal_dim) == 2 else 1
        self.downsample = kwargs.get("downsampling", 0)
        bn = kwargs.get('batch_norm', True)
        if self.downsample:
            signal_dim = [s // self.downsample for s in signal_dim]
        n = np.product(signal_dim)
        self.reshape_ = nb_filters * n
        self.conv = ConvBlock(
            ndim, nb_layers, 1, nb_filters,
            lrelu_a=0.1, batch_norm=bn)
        self.fc = nn.Linear(nb_filters * n, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embeddes the input signal into a latent vector
        """
        if self.downsample:
            if x.ndim == 3:
                x = F.avg_pool1d(
                    x, self.downsample, self.downsample)
            else:
                x = F.avg_pool2d(
                    x, self.downsample, self.downsample)
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
        bn = kwargs.get('batch_norm', True)
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
            self.deconv1 = ConvBlock(
                ndim, 1, nb_filters, nb_filters,
                lrelu_a=0.1, batch_norm=bn)
            self.deconv2 = ConvBlock(
                ndim, 1, nb_filters, nb_filters,
                lrelu_a=0.1, batch_norm=bn)
        self.dilblock = DilatedBlock(
            ndim, nb_filters, nb_filters,
            dilation_values=torch.arange(1, nb_layers + 1).tolist(),
            padding_values=torch.arange(1, nb_layers + 1).tolist(),
            lrelu_a=0.1, batch_norm=bn)
        self.conv = ConvBlock(
            ndim, 1, nb_filters, 1,
            lrelu_a=0.1, batch_norm=bn)
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
                 nblayers_encoder: int = 2, nblayers_decoder: int = 2,
                 nbfilters_encoder: int = 64, nbfilters_decoder: int = 2,
                 batch_norm: bool = True, encoder_downsampling: int = 0,
                 decoder_upsampling: bool = False) -> None:
        """
        Initializes im2spec/spec2im parameters
        """
        super(signal_ed, self).__init__()
        self.encoder = signal_encoder(
            feature_dim, latent_dim, nblayers_encoder, nbfilters_encoder,
            batch_norm=batch_norm, downsampling=encoder_downsampling)
        self.decoder = signal_decoder(
            target_dim, latent_dim, nblayers_decoder, nbfilters_decoder,
            batch_norm=batch_norm, upsampling=decoder_upsampling)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Embeddes the input image into a latent vector
        """
        return self.encoder(features)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Generates signal from the embedded features
        """
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        x = self.encode(x)
        return self.decode(x)


def init_imspec_model(in_dim, out_dim, latent_dim, **kwargs):
    """
    """
    nblayers_encoder = kwargs.get("nblayers_encoder", 3)
    nblayers_decoder = kwargs.get("nblayers_decoder", 4)
    nbfilters_encoder = kwargs.get("nbfilters_encoder", 64)
    nbfilters_decoder = kwargs.get("nbfilters_decoder", 64)
    batch_norm = kwargs.get("batch_norm", True)
    encoder_downsampling = kwargs.get("encoder_downsampling", 0)
    decoder_upsampling = kwargs.get("decoder_upsampling", False)
    net = signal_ed(
        in_dim, out_dim, latent_dim, nblayers_encoder, nblayers_decoder,
        nbfilters_encoder, nbfilters_decoder, batch_norm, encoder_downsampling,
        decoder_upsampling)
    meta_state_dict = {
        "model_type": "imspec",
        "in_dim": in_dim,
        "out_dim": out_dim,
        "latent_dim": latent_dim,
        "nblayers_encoder": nblayers_encoder,
        "nblayers_decoder": nblayers_decoder,
        "nbfilters_encoder": nbfilters_encoder,
        "nbfilters_decoder": nbfilters_decoder,
        "batchnorm": batch_norm,
        "encoder_downsampling": encoder_downsampling,
        "decoder_upsampling": decoder_upsampling
    }
    return net, meta_state_dict
