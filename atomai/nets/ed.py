"""
imspec.py
=========

Encoder and decoder modules for VAE/VED and im2spec/spec2im models

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)

"""

from typing import Tuple, Type, Union, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock, DilatedBlock


class SignalEncoder(nn.Module):
    """
    Encodes 1D/2D signal into a latent vector

    Args:
        signal_dim:
            Size of input signal. For images, it is (height, width).
            For spectra, it is (length,)
        z_dim:
            Number of fully-connected neurons in a "bottleneck layer"
            (latent dimensions)
        nb_layers:
            Number of convolutional layers
        nb_filters:
            Number of convolutional filters (aka "kernels") in each layer
        **batch_norm (bool):
            Apply batch normalization after each convolutional layer
            (Default: True)
        **downsampling (int):
            Downsamples input data by this factor before passing
            to convolutional layers (Default: no downsampling)

    """
    def __init__(self, signal_dim: Tuple[int],
                 z_dim: int, nb_layers: int, nb_filters: int,
                 **kwargs: int) -> None:
        """
        Initialize module parameters
        """
        super(SignalEncoder, self).__init__()
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


class SignalDecoder(nn.Module):
    """
    Decodes a latent vector into 1D/2D signal

    Args:
        signal_dim:
            Size of input signal. For images, it is (height, width).
            For spectra, it is (length,)
        z_dim:
            Number of fully-connected neurons in a "bottleneck layer"
            (latent dimensions)
        nb_layers:
            Number of convolutional layers
        nb_filters:
            Number of convolutional filters (aka "kernels") in each layer
        **batch_norm (bool):
            Apply batch normalization after each convolutional layer
            (Default: True)
        **upsampling (bool):
            Performs upsampling+convolution operation twice on the reshaped latent
            vector (starting from image/spectra dims 4x smaller than the target dims)
            before passing  to the decoder
    """
    def __init__(self, signal_dim: Tuple[int],
                 z_dim: int, nb_layers: int, nb_filters: int,
                 **kwargs: bool) -> None:
        """
        Initializes module parameters
        """
        super(SignalDecoder, self).__init__()
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


class SignalED(nn.Module):
    """
    Transforms image into spectra (im2spec) and vice versa (spec2im)

    Args:
        feature_dim:
            Input data dimensions.
            (height, width) for images or (length,) for spectra
        target_dim:
            Output dimensions.
            (length,) for spectra or (height, width) for images
        latent_dim:
            Dimensionality of the latent space
            (number of neurons in a fully connected "bottleneck" layer)
        nblayers_encoder:
            Number of convolutional layers in the encoder
        nblayers_decoder:
            Number of convolutional layers in the decoder
        nbfilters_encoder:
            number of convolutional filters in each layer of the encoder
        nbfilters_decoder:
            Number of convolutional filters in each layer of the decoder
        batch_norm:
            Apply batch normalization after each convolutional layer
            (Default: True)
        encoder_downsampling:
            Downsamples input data by this factor before passing
            to convolutional layers (Default: no downsampling)
        decoder_upsampling:
            Performs upsampling+convolution operation twice on the reshaped latent
            vector (starting from image/spectra dims 4x smaller than the target dims)
            before passing  to the decoder

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
        super(SignalED, self).__init__()
        self.encoder = SignalEncoder(
            feature_dim, latent_dim, nblayers_encoder, nbfilters_encoder,
            batch_norm=batch_norm, downsampling=encoder_downsampling)
        self.decoder = SignalDecoder(
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


class convEncoderNet(nn.Module):
    """
    Convolutional rncoder/inference network (for variational autoencoder)

    Args:
        in_dim:
            Input dimensions.
            For images, it is (height, width) or (height, width, channels).
            For spectra, it is (length,)
        latent_dim:
            number of latent dimensions
            (the first 3 latent dimensions are angle & translations by default)
        num_layers:
            number of NN layers
        hidden_dim:
            number of neurons in each fully connnected layer (for mlp=True)
            or number of filters in each convolutional layer (for mlp=False)
    """
    def __init__(self,
                 in_dim: Tuple[int],
                 latent_dim: int = 2,
                 num_layers: int = 2,
                 hidden_dim: int = 32,
                 **kwargs: float
                 ) -> None:
        """
        Initializes network parameters
        """
        super(convEncoderNet, self).__init__()
        if len(in_dim) not in (1, 2, 3):
            raise ValueError(
                "The input dimensions must be (length,) for 1D data and " +
                "(height, width) or (height, width, channel) for 2D data")
        dim = 2 if len(in_dim) > 1 else 1
        c = in_dim[-1] if len(in_dim) > 2 else 1
        self.conv = ConvBlock(
            dim, num_layers, c, hidden_dim,
            lrelu_a=kwargs.get("lrelu_a", 0.1))
        self.reshape_ = hidden_dim * np.product(in_dim[:2])
        self.fc11 = nn.Linear(self.reshape_, latent_dim)
        self.fc12 = nn.Linear(self.reshape_, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor with channel (> 1) as the last dimension
        """
        x = x.unsqueeze(1) if x.ndim in (2, 3) else x.permute(0, -1, 1, 2)
        x = self.conv(x)
        x = x.reshape(-1, self.reshape_)
        z_mu = self.fc11(x)
        z_logstd = self.fc12(x)
        return z_mu, z_logstd


class fcEncoderNet(nn.Module):
    """
    Encoder/inference network (for variational autoencoder)

    Args:
        in_dim:
            Input dimensions.
            For images, it is (height, width) or (height, width, channels).
            For spectra, it is (length,)
        latent_dim:
            number of latent dimensions
            (the first 3 latent dimensions are angle & translations by default)
        num_layers:
            number of NN layers
        hidden_dim:
            number of neurons in each fully connnected layer (for mlp=True)
            or number of filters in each convolutional layer (for mlp=False)
    """
    def __init__(self,
                 in_dim: Tuple[int],
                 latent_dim: int = 2,
                 num_layers: int = 2,
                 hidden_dim: int = 32,
                 ) -> None:
        """
        Initializes network parameters
        """
        super(fcEncoderNet, self).__init__()
        dense = []
        for i in range(num_layers):
            input_dim = np.product(in_dim) if i == 0 else hidden_dim
            dense.extend([nn.Linear(input_dim, hidden_dim), nn.Tanh()])
        self.dense = nn.Sequential(*dense)
        self.reshape_ = hidden_dim
        self.fc11 = nn.Linear(self.reshape_, latent_dim)
        self.fc12 = nn.Linear(self.reshape_, latent_dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass
        """
        x = x.reshape(-1, np.product(x.size()[1:]))
        x = self.dense(x)
        x = x.reshape(-1, self.reshape_)
        z_mu = self.fc11(x)
        z_logstd = self.fc12(x)
        return z_mu, z_logstd


class convDecoderNet(nn.Module):
    """
    Convolutional decoder network (for variational autoencoder)

    Args:
        out_dim:
            Output dimensions.
            For images, it is (height, width) or (height, width, channels).
            For spectra, it is (length,)
        latent_dim:
            number of latent dimensions associated with images content
        num_layers:
            number of fully connected layers
        hidden_dim:
            number of neurons in each fully connected layer
    """
    def __init__(self,
                 out_dim: Tuple[int],
                 latent_dim: int,
                 num_layers: int = 2,
                 hidden_dim: int = 32,
                 num_classes: int = 0,
                 **kwargs: float) -> None:
        """
        Initializes network parameters
        """
        super(convDecoderNet, self).__init__()
        if len(out_dim) not in (1, 2, 3):
            raise ValueError(
                "The output dimensions must be (length,) for 1D data and " +
                "(height, width) or (height, width, channel) for 2D data")
        dim = 2 if len(out_dim) > 1 else 1
        c = out_dim[-1] if len(out_dim) > 2 else 1
        self.fc_linear = nn.Linear(
            latent_dim + num_classes, hidden_dim * np.product(out_dim[:2]),
            bias=False)
        self.reshape_ = (hidden_dim, *out_dim[:2])
        self.decoder = ConvBlock(
            dim, num_layers, hidden_dim, hidden_dim,
            lrelu_a=kwargs.get("lrelu_a", 0.1))
        conv_1x1 = nn.Conv2d if dim == 2 else nn.Conv1d
        self.conv_1x1 = conv_1x1(hidden_dim, c, 1, 1, 0)
        self.out_dim = (c, *out_dim[:2])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        z = self.fc_linear(z)
        z = z.reshape(-1, *self.reshape_)
        h = self.decoder(z)
        h = self.conv_1x1(h)
        h = h.reshape(-1, *self.out_dim)
        if h.size(1) == 1:
            h = h.squeeze(1)
        else:
            h = h.permute(0, 2, 3, 1)
        return h


class fcDecoderNet(nn.Module):
    """
    Decoder network (for variational autoencoder)

    Args:
        out_dim:
            Output dimensions.
            For images, it is (height, width) or (height, width, channels).
            For spectra, it is (length,)
        latent_dim:
            number of latent dimensions associated with images content
        num_layers:
            number of fully connected layers
        hidden_dim:
            number of neurons in each fully connected layer
    """
    def __init__(self,
                 out_dim: Tuple[int],
                 latent_dim: int,
                 num_layers: int = 2,
                 hidden_dim: int = 32,
                 num_classes: int = 0) -> None:
        """
        Initializes network parameters
        """
        super(fcDecoderNet, self).__init__()
        if len(out_dim) not in (1, 2, 3):
            raise ValueError(
                "The output dimensions must be (length,) for 1D data and " +
                "(height, width) or (height, width, channel) for 2D data")
        c = out_dim[-1] if len(out_dim) > 2 else 1
        decoder = []
        for i in range(num_layers):
            hidden_dim_ = latent_dim + num_classes if i == 0 else hidden_dim
            decoder.extend([nn.Linear(hidden_dim_, hidden_dim), nn.Tanh()])
        self.decoder = nn.Sequential(*decoder)
        self.out = nn.Linear(hidden_dim, np.product(out_dim))
        self.out_dim = (c, *out_dim[:2])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        h = self.decoder(z)
        h = self.out(h)
        h = h.reshape(-1, *self.out_dim)
        if h.size(1) == 1:
            h = h.squeeze(1)
        else:
            h = h.permute(0, 2, 3, 1)
        return h


class rDecoderNet(nn.Module):
    """
    Spatial decoder network with (optional) skip connections

    Args:
        out_dim:
            output dimensions: (height, width) or (height, width, channels)
        latent_dim:
            number of latent dimensions associated with images content
        num_layers:
            number of fully connected layers
        hidden_dim:
            number of neurons in each fully connected layer
        skip:
            Use skip connections to propagate latent variables
            through decoder network (Default: False)
    """
    def __init__(self,
                 out_dim: Tuple[int],
                 latent_dim: int,
                 num_layers: int,
                 hidden_dim: int,
                 skip: bool = False,
                 num_classes: int = 0) -> None:
        """
        Initializes network parameters
        """
        super(rDecoderNet, self).__init__()
        if len(out_dim) == 2:
            c = 1
            self.reshape_ = (out_dim[0], out_dim[1])
        else:
            c = out_dim[-1]
            self.reshape_ = (out_dim[0], out_dim[1], c)
        self.skip = skip
        self.coord_latent = coord_latent(
            latent_dim+num_classes, hidden_dim, not skip)
        fc_decoder = []
        for i in range(num_layers):
            fc_decoder.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        self.fc_decoder = nn.Sequential(*fc_decoder)
        self.out = nn.Linear(hidden_dim, c)

    def forward(self, x_coord: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        batch_dim = x_coord.size()[0]
        h = self.coord_latent(x_coord, z)
        if self.skip:
            residual = h
            for i, fc_block in enumerate(self.fc_decoder):
                h = fc_block(h)
                if (i + 1) % 2 == 0:
                    h = h.add(residual)
        else:
            h = self.fc_decoder(h)
        h = self.out(h)
        h = h.reshape(batch_dim, *self.reshape_)
        return h


class coord_latent(nn.Module):
    """
    The "spatial" part of the rVAE's decoder that allows for translational
    and rotational invariance (based on https://arxiv.org/abs/1909.11663)

    Args:
        latent_dim:
            number of latent dimensions associated with images content
        out_dim:
            number of output dimensions
            (usually equal to number of hidden units
             in the first layer of the corresponding VAE's decoder)
        activation:
            Applies tanh activation to the output (Default: False)
    """
    def __init__(self,
                 latent_dim: int,
                 out_dim: int,
                 activation: bool = False) -> None:
        """
        Initiate parameters
        """
        super(coord_latent, self).__init__()
        self.fc_coord = nn.Linear(2, out_dim)
        self.fc_latent = nn.Linear(latent_dim, out_dim, bias=False)
        self.activation = nn.Tanh() if activation else None

    def forward(self,
                x_coord: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        batch_dim, n = x_coord.size()[:2]
        x_coord = x_coord.reshape(batch_dim * n, -1)
        h_x = self.fc_coord(x_coord)
        h_x = h_x.reshape(batch_dim, n, -1)
        h_z = self.fc_latent(z)
        h = h_x.add(h_z.unsqueeze(1))
        h = h.reshape(batch_dim * n, -1)
        if self.activation is not None:
            h = self.activation(h)
        return h


def init_imspec_model(in_dim: Tuple[int],
                      out_dim: Tuple[int],
                      latent_dim: int,
                      **kwargs: Union[int, bool]
                      ) -> Tuple[Type[nn.Module], Dict[str, Union[int, bool]]]:
    """
    Initializes ImSpec model
    """
    nblayers_encoder = kwargs.get("nblayers_encoder", 3)
    nblayers_decoder = kwargs.get("nblayers_decoder", 4)
    nbfilters_encoder = kwargs.get("nbfilters_encoder", 64)
    nbfilters_decoder = kwargs.get("nbfilters_decoder", 64)
    batch_norm = kwargs.get("batch_norm", True)
    encoder_downsampling = kwargs.get("encoder_downsampling", 0)
    decoder_upsampling = kwargs.get("decoder_upsampling", False)
    net = SignalED(
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


def init_VAE_nets(in_dim: Tuple[int],
                  latent_dim: int,
                  coord: int,
                  nb_classes: int,
                  **kwargs
                  ) -> Tuple[Type[nn.Module], Type[nn.Module], Dict[str, Union[int, bool]]]:
    """
    Initializes encoder and decoder for VAE
    """
    conv_e = kwargs.get("conv_encoder", False)
    if not coord:
        conv_d = kwargs.get("conv_decoder", False)
    numlayers_e = kwargs.get("numlayers_encoder", 2)
    numlayers_d = kwargs.get("numlayers_decoder", 2)
    numhidden_e = kwargs.get("numhidden_encoder", 128)
    numhidden_d = kwargs.get("numhidden_decoder", 128)
    skip = kwargs.get("skip", False)

    if not coord:
        dnet = convDecoderNet if conv_d else fcDecoderNet
        decoder_net = dnet(
            in_dim, latent_dim, numlayers_d, numhidden_d,
            nb_classes)
    else:
        decoder_net = rDecoderNet(
            in_dim, latent_dim, numlayers_d, numhidden_d,
            skip, nb_classes)
    enet = convEncoderNet if conv_e else fcEncoderNet
    encoder_net = enet(
        in_dim, latent_dim + coord, numlayers_e, numhidden_e)

    meta_state_dict = {
        "model_type": "vae",
        "in_dim": in_dim,
        "latent_dim": latent_dim,
        "coord": coord,
        "conv_encoder": conv_e,
        "numlayers_encoder": numlayers_e,
        "numlayers_decoder": numlayers_d,
        "numhidden_encoder": numhidden_e,
        "numhidden_decoder": numhidden_d,
        "skip": skip,
        "nb_classes": nb_classes
    }
    if not coord:
        meta_state_dict["conv_decoder"] = conv_d

    return encoder_net, decoder_net, meta_state_dict
