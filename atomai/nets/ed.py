"""
ed.py
=====

Encoder/decoder modules for variational autoencoders

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)

"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import conv2dblock


class EncoderNet(nn.Module):
    """
    Encoder/inference network (for variational autoencoder)

    Args:
        dim (tuple):
            image dimensions: (height, width) or (height, width, channels)
        latent_dim (int):
            number of latent dimensions
            (the first 3 latent dimensions are angle & translations by default)
        num_layers (int):
            number of NN layers
        hidden_dim (int):
            number of neurons in each fully connnected layer (for mlp=True)
            or number of filters in each convolutional layer (for mlp=False)
        mlp (bool):
            use a simple multi-layer perceptron instead of convolutional layers
            (Default: False)

    """
    def __init__(self,
                 dim: Tuple[int],
                 latent_dim: int = 5,
                 num_layers: int = 2,
                 hidden_dim: int = 32,
                 mlp: bool = False) -> None:
        """
        Initializes network parameters
        """
        super(EncoderNet, self).__init__()
        c = 1 if len(dim) == 2 else dim[-1]
        self.mlp = mlp
        if not self.mlp:
            self.econv = conv2dblock(num_layers, c, hidden_dim, lrelu_a=0.1)
            self.reshape_ = hidden_dim * dim[0] * dim[1]
        else:
            edense = []
            for i in range(num_layers):
                input_dim = np.product(dim) if i == 0 else hidden_dim
                edense.extend([nn.Linear(input_dim, hidden_dim), nn.Tanh()])
            self.edense = nn.Sequential(*edense)
            self.reshape_ = hidden_dim
        self.fc11 = nn.Linear(self.reshape_, latent_dim)
        self.fc12 = nn.Linear(self.reshape_, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass
        """
        if not self.mlp:
            x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, -1, 1, 2)
            x = self.econv(x)
        else:
            x = x.reshape(-1, np.product(x.size()[1:]))
            x = self.edense(x)
        x = x.reshape(-1, self.reshape_)
        z_mu = self.fc11(x)
        z_logstd = self.fc12(x)
        return z_mu, z_logstd


class rDecoderNet(nn.Module):
    """
    Spatial decoder network with skip connections

    Args:
        latent_dim (int):
            number of latent dimensions associated with images content
        num_layers (int):
            number of fully connected layers
        hidden_dim (int):
            number of neurons in each fully connected layer
        out_dim (tuple):
            output dimensions: (height, width) or (height, width, channels)
        skip (bool):
            Use skip connections to propagate latent variables
            through decoder network (Default: False)
    """
    def __init__(self,
                 latent_dim: int,
                 num_layers: int,
                 hidden_dim: int,
                 out_dim: Tuple[int],
                 skip: bool = False) -> None:
        """
        Initializes network parameters
        """
        super(rDecoderNet, self).__init__()
        if len(out_dim) == 2:
            c = 1
            self.reshape_ = (out_dim[0], out_dim[1])
            self.apply_softplus = True
        else:
            c = out_dim[-1]
            self.reshape_ = (out_dim[0], out_dim[1], c)
            self.apply_softplus = False
        self.skip = skip
        self.coord_latent = coord_latent(latent_dim, hidden_dim, not skip)
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
        if self.apply_softplus:
            return F.softplus(h)
        return h


class coord_latent(nn.Module):
    """
    The "spatial" part of the rVAE's decoder that allows for translational
    and rotational invariance (based on https://arxiv.org/abs/1909.11663)

    Args:
        latent_dim (int):
            number of latent dimensions associated with images content
        out_dim (int):
            number of output dimensions
            (usually equal to number of hidden units
             in the first layer of the corresponding VAE's decoder)
        activation (bool):
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


class DecoderNet(nn.Module):
    """
    Decoder network (for variational autoencoder)

    Args:
        latent_dim (int):
            number of latent dimensions associated with images content
        num_layers (int):
            number of fully connected layers
        hidden_dim (int):
            number of neurons in each fully connected layer
        out_dim (tuple):
            image dimensions: (height, width) or (height, width, channels)
        mlp (bool):
            using a simple multi-layer perceptron instead of convolutional layers
            (Default: False)
    """
    def __init__(self,
                 latent_dim: int,
                 num_layers: int,
                 hidden_dim: int,
                 out_dim: Tuple[int],
                 mlp: bool = False,) -> None:
        """
        Initializes network parameters
        """
        super(DecoderNet, self).__init__()
        c = 1 if len(out_dim) == 2 else out_dim[-1]
        self.mlp = mlp
        if not self.mlp:
            self.fc_linear = nn.Linear(
                latent_dim, hidden_dim * out_dim[0] * out_dim[1], bias=False)
            self.reshape_ = (hidden_dim, out_dim[0], out_dim[1])
            self.decoder = conv2dblock(
                num_layers, hidden_dim, hidden_dim, lrelu_a=0.1)
            self.out = nn.Conv2d(hidden_dim, c, 1, 1, 0)
        else:
            decoder = []
            for i in range(num_layers):
                hidden_dim_ = latent_dim if i == 0 else hidden_dim
                decoder.extend([nn.Linear(hidden_dim_, hidden_dim), nn.Tanh()])
            self.decoder = nn.Sequential(*decoder)
            self.out = nn.Linear(hidden_dim, np.product(out_dim))
        self.out_dim = (c, out_dim[0], out_dim[1])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        if not self.mlp:
            z = self.fc_linear(z)
            z = z.reshape(-1, *self.reshape_)
        h = self.decoder(z)
        h = self.out(h)
        h = h.reshape(-1, *self.out_dim)
        if h.size(1) == 1:
            h = h.squeeze(1)
        else:
            h = h.permute(0, 2, 3, 1)
        return h
