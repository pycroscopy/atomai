"""
blocks.py
=========

Customized NN blocks

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2dblock(nn.Module):
    """
    Creates block of layers each consisting of convolution operation,
    leaky relu and (optionally) dropout and batch normalization

    Args:
        nb_layers (int):
            Number of layers in the block
        input_channels (int):
            Number of input channels for the block
        output_channels (int):
            Number of the output channels for the block
        kernel_size (int):
            Size of convolutional filter (in pixels)
        stride (int):
            Stride of convolutional filter
        padding (int):
            Value for edge padding
        use_batchnorm (bool):
            Add batch normalization to each layer in the block
        lrelu_a (float)
            Value of alpha parameter in leaky ReLU activation
            for each layer in the block
        dropout_ (float):
            Dropout value for each layer in the block
    """
    def __init__(self, nb_layers: int, input_channels: int,
                 output_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1,
                 use_batchnorm: bool = False, lrelu_a: float = 0.01,
                 dropout_: float = 0) -> None:
        """
        Initializes module parameters
        """
        super(conv2dblock, self).__init__()
        block = []
        for idx in range(nb_layers):
            input_channels = output_channels if idx > 0 else input_channels
            block.append(nn.Conv2d(input_channels,
                                   output_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding))
            if dropout_ > 0:
                block.append(nn.Dropout(dropout_))
            block.append(nn.LeakyReLU(negative_slope=lrelu_a))
            if use_batchnorm:
                block.append(nn.BatchNorm2d(output_channels))
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines a forward pass
        """
        output = self.block(x)
        return output


class upsample_block(nn.Module):
    """
    Defines upsampling block performed using bilinear
    or nearest-neigbor interpolation followed by 1-by-1 convolution
    (the latter can be used to reduce a number of feature channels)

    Args:
        input_channels (int):
            Number of input channels for the block
        output_channels (int):
            Number of the output channels for the block
        scale_factor (positive int):
            Scale factor for upsampling
        mode (str):
            Upsampling mode. Select between "bilinear" and "nearest"
    """
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 scale_factor: int = 2,
                 mode: str = "bilinear") -> None:
        """
        Initializes module parameters
        """
        super(upsample_block, self).__init__()
        if not any([mode == 'bilinear', mode == 'nearest']):
            raise NotImplementedError(
                "use 'bilinear' or 'nearest' for upsampling mode")
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            input_channels, output_channels,
            kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines a forward pass
        """
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


class dilated_block(nn.Module):
    """
    Creates a "pyramid" with dilated convolutional
    layers (aka atrous convolutions)

    Args:
        input_channels (int):
            Number of input channels for the block
        output_channels (int):
            Number of the output channels for the block
        dilation_values (list of ints):
            List of dilation rates for each convolution layer in the block
            (for example, dilation_values = [2, 4, 6] means that the dilated
            block will 3 layers with dilation values of 2, 4, and 6).
        padding_values (list of ints):
            Edge padding for each dilated layer. The number of elements in this
            list should be equal to that in the dilated_values list and
            typically they can have the same values.
        kernel_size (int):
            Size of convolutional filter (in pixels)
        stride (int):
            Stride of convolutional filter
        use_batchnorm (bool):
            Add batch normalization to each layer in the block
        lrelu_a (float)
            Value of alpha parameter in leaky ReLU activation
            for each layer in the block
        dropout_ (float):
            Dropout value for each layer in the block

    """
    def __init__(self, input_channels: int, output_channels: int,
                 dilation_values: List[int], padding_values: List[int],
                 kernel_size: int = 3, stride: int = 1, lrelu_a: float = 0.01,
                 use_batchnorm: bool = False, dropout_: float = 0) -> None:
        """
        Initializes module parameters
        """
        super(dilated_block, self).__init__()
        atrous_module = []
        for idx, (dil, pad) in enumerate(zip(dilation_values, padding_values)):
            input_channels = output_channels if idx > 0 else input_channels
            atrous_module.append(nn.Conv2d(input_channels,
                                           output_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=pad,
                                           dilation=dil,
                                           bias=True))
            if dropout_ > 0:
                atrous_module.append(nn.Dropout(dropout_))
            atrous_module.append(nn.LeakyReLU(negative_slope=lrelu_a))
            if use_batchnorm:
                atrous_module.append(nn.BatchNorm2d(output_channels))
        self.atrous_module = nn.Sequential(*atrous_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines a forward pass
        """
        atrous_layers = []
        for conv_layer in self.atrous_module:
            x = conv_layer(x)
            atrous_layers.append(x.unsqueeze(-1))
        return torch.sum(torch.cat(atrous_layers, dim=-1), dim=-1)
