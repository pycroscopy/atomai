"""
fcnn.py
=========

Fully convolutional neural networks

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import conv2dblock, dilated_block, upsample_block


class dilUnet(nn.Module):
    """
    Builds a fully convolutional Unet-like neural network model

    Args:
        nb_classes (int):
            Number of classes in the ground truth
        nb_filters (int):
            Number of filters in 1st convolutional block
            (gets multibplied by 2 in each next block)
        use_dropout (bool):
            Use / not use dropout in the 3 inner layers
        batch_norm (bool):
            Use / not use batch normalization after each convolutional layer
        upsampling mode (str):
            Select between "bilinear" or "nearest" upsampling method.
            Bilinear is usually more accurate,but adds additional (small)
            randomness. For full reproducibility, consider using 'nearest'
            (this assumes that all other sources of randomness are fixed)
        **layers (list):
            List with a number of layers in each block.
            For U-Net the first 4 elements in the list
            are used to determine the number of layers
            in each block of the encoder (incluidng bottleneck layer),
            and the number of layers in the decoder  is chosen accordingly
            (to maintain symmetry between encoder and decoder)
    """
    def __init__(self,
                 nb_classes: int = 1,
                 nb_filters: int = 16,
                 use_dropout: bool = False,
                 batch_norm: bool = True,
                 upsampling_mode: str = "bilinear",
                 with_dilation: bool = True,
                 **kwargs: List[int]) -> None:
        """
        Initializes model parameters
        """
        super(dilUnet, self).__init__()
        nbl = kwargs.get("layers", [1, 2, 2, 3])
        dilation_values = torch.arange(2, 2*nbl[-1]+1, 2).tolist()
        padding_values = dilation_values.copy()
        dropout_vals = [.1, .2, .1] if use_dropout else [0, 0, 0]
        self.c1 = conv2dblock(
            nbl[0], 1, nb_filters,
            use_batchnorm=batch_norm
        )
        self.c2 = conv2dblock(
            nbl[1], nb_filters, nb_filters*2,
            use_batchnorm=batch_norm
        )
        self.c3 = conv2dblock(
            nbl[2], nb_filters*2, nb_filters*4,
            use_batchnorm=batch_norm,
            dropout_=dropout_vals[0]
        )
        if with_dilation:
            self.bn = dilated_block(
                nb_filters*4, nb_filters*8,
                dilation_values=dilation_values,
                padding_values=padding_values,
                use_batchnorm=batch_norm,
                dropout_=dropout_vals[1]
            )
        else:
            self.bn = conv2dblock(
                nbl[3], nb_filters*4, nb_filters*8,
                use_batchnorm=batch_norm,
                dropout_=dropout_vals[1]
            )
        self.upsample_block1 = upsample_block(
            nb_filters*8, nb_filters*4,
            mode=upsampling_mode)
        self.c4 = conv2dblock(
            nbl[2], nb_filters*8, nb_filters*4,
            use_batchnorm=batch_norm,
            dropout_=dropout_vals[2]
        )
        self.upsample_block2 = upsample_block(
            nb_filters*4, nb_filters*2,
            mode=upsampling_mode)
        self.c5 = conv2dblock(
            nbl[1], nb_filters*4, nb_filters*2,
            use_batchnorm=batch_norm
        )
        self.upsample_block3 = upsample_block(
            nb_filters*2, nb_filters,
            mode=upsampling_mode)
        self.c6 = conv2dblock(
            nbl[0], nb_filters*2, nb_filters,
            use_batchnorm=batch_norm
        )
        self.px = nn.Conv2d(nb_filters, nb_classes, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines a forward pass
        """
        # Contracting path
        c1 = self.c1(x)
        d1 = F.max_pool2d(c1, kernel_size=2, stride=2)
        c2 = self.c2(d1)
        d2 = F.max_pool2d(c2, kernel_size=2, stride=2)
        c3 = self.c3(d2)
        d3 = F.max_pool2d(c3, kernel_size=2, stride=2)
        # Bottleneck layer
        bn = self.bn(d3)
        # Expanding path
        u3 = self.upsample_block1(bn)
        u3 = torch.cat([c3, u3], dim=1)
        u3 = self.c4(u3)
        u2 = self.upsample_block2(u3)
        u2 = torch.cat([c2, u2], dim=1)
        u2 = self.c5(u2)
        u1 = self.upsample_block3(u2)
        u1 = torch.cat([c1, u1], dim=1)
        u1 = self.c6(u1)
        # Final layer used for pixel-wise convolution
        px = self.px(u1)
        return px


class dilnet(nn.Module):
    """
    Builds  a fully convolutional neural network model
    by utilizing a combination of regular and dilated convolutions

    Args:
        nb_classes (int):
            Number of classes in the ground truth
        nb_filters (int):
            Number of filters in 1st convolutional block
            (gets multiplied by 2 in each next block)
        use_dropout (bool):
            Use / not use dropout in the 3 inner layers
        batch_norm (bool):
            Use / not use batch normalization after each convolutional layer
        upsampling mode (str):
            Select between "bilinear" or "nearest" upsampling method.
            Bilinear is usually more accurate,but adds additional (small)
            randomness. For full reproducibility, consider using 'nearest'
            (this assumes that all other sources of randomness are fixed)
        **layers (list):
            List with a number of layers for each block.
    """

    def __init__(self,
                 nb_classes: int = 1,
                 nb_filters: int = 25,
                 use_dropout: bool = False,
                 batch_norm: bool = True,
                 upsampling_mode: str = "bilinear",
                 **kwargs: List[int]) -> None:
        """
        Initializes model parameters
        """
        super(dilnet, self).__init__()
        nbl = kwargs.get("layers", [3, 3, 3, 3])
        dilation_values_1 = torch.arange(2, 2*nbl[1]+1, 2).tolist()
        padding_values_1 = dilation_values_1.copy()
        dilation_values_2 = torch.arange(2, 2*nbl[2]+1, 2).tolist()
        padding_values_2 = dilation_values_2.copy()
        dropout_vals = [.3, .3] if use_dropout else [0, 0]
        self.c1 = conv2dblock(
                    nbl[0], 1, nb_filters,
                    use_batchnorm=batch_norm
        )
        self.at1 = dilated_block(
                    nb_filters, nb_filters*2,
                    dilation_values=dilation_values_1,
                    padding_values=padding_values_1,
                    use_batchnorm=batch_norm,
                    dropout_=dropout_vals[0]
        )
        self.at2 = dilated_block(
                    nb_filters*2, nb_filters*2,
                    dilation_values=dilation_values_2,
                    padding_values=padding_values_2,
                    use_batchnorm=batch_norm,
                    dropout_=dropout_vals[1]
        )
        self.up1 = upsample_block(
                    nb_filters*2, nb_filters,
                    mode=upsampling_mode
        )
        self.c2 = conv2dblock(
                    nbl[3], nb_filters*2, nb_filters,
                    use_batchnorm=batch_norm
        )
        self.px = nn.Conv2d(nb_filters, nb_classes, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines a forward pass
        """
        c1 = self.c1(x)
        d1 = F.max_pool2d(c1, kernel_size=2, stride=2)
        at1 = self.at1(d1)
        at2 = self.at2(at1)
        u1 = self.up1(at2)
        u1 = torch.cat([c1, u1], dim=1)
        u1 = self.c2(u1)
        px = self.px(u1)
        return px
