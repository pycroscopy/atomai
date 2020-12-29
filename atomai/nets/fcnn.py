"""
fcnn.py
=========

Fully convolutional neural networks

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import List, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock, ResModule, DilatedBlock, UpsampleBlock


class Unet(nn.Module):
    """
    Builds a fully convolutional Unet-like neural network model

    Args:
        nb_classes:
            Number of classes in the ground truth
        nb_filters:
            Number of filters in 1st convolutional block
            (gets multiplied by 2 in each next block)
        dropout:
            Use dropouts to the 3 inner layers
            (Default: False)
        batch_norm:
            Use batch normalization after each convolutional layer
            (Default: True)
        upsampling_mode:
            Select between "bilinear" or "nearest" upsampling method.
            Bilinear is usually more accurate,but adds additional (small)
            randomness. For full reproducibility, consider using 'nearest'
            (this assumes that all other sources of randomness are fixed)
        with_dilation:
            Use dilated convolutions instead of regular ones in the
            bottleneck layers (Default: False)
        **layers (list):
            List with a number of layers in each block.
            The first 4 elements in the list
            are used to determine the number of layers
            in each block of the encoder (incluidng bottleneck layers),
            and the number of layers in the decoder  is chosen accordingly
            (to maintain symmetry between encoder and decoder)
    """
    def __init__(self,
                 nb_classes: int = 1,
                 nb_filters: int = 16,
                 dropout: bool = False,
                 batch_norm: bool = True,
                 upsampling_mode: str = "bilinear",
                 with_dilation: bool = False,
                 **kwargs: List[int]) -> None:
        """
        Initializes model parameters
        """
        super(Unet, self).__init__()
        nbl = kwargs.get("layers", [1, 2, 2, 3])
        dilation_values = torch.arange(2, 2*nbl[-1]+1, 2).tolist()
        padding_values = dilation_values.copy()
        dropout_vals = [.1, .2, .1] if dropout else [0, 0, 0]
        self.c1 = ConvBlock(
            2, nbl[0], 1, nb_filters,
            batch_norm=batch_norm
        )
        self.c2 = ConvBlock(
            2, nbl[1], nb_filters, nb_filters*2,
            batch_norm=batch_norm
        )
        self.c3 = ConvBlock(
            2, nbl[2], nb_filters*2, nb_filters*4,
            batch_norm=batch_norm,
            dropout_=dropout_vals[0]
        )
        if with_dilation:
            self.bn = DilatedBlock(
                2, nb_filters*4, nb_filters*8,
                dilation_values=dilation_values,
                padding_values=padding_values,
                batch_norm=batch_norm,
                dropout_=dropout_vals[1]
            )
        else:
            self.bn = ConvBlock(
                2, nbl[3], nb_filters*4, nb_filters*8,
                batch_norm=batch_norm,
                dropout_=dropout_vals[1]
            )
        self.upsample_block1 = UpsampleBlock(
            2, nb_filters*8, nb_filters*4,
            mode=upsampling_mode)
        self.c4 = ConvBlock(
            2, nbl[2], nb_filters*8, nb_filters*4,
            batch_norm=batch_norm,
            dropout_=dropout_vals[2]
        )
        self.upsample_block2 = UpsampleBlock(
            2, nb_filters*4, nb_filters*2,
            mode=upsampling_mode)
        self.c5 = ConvBlock(
            2, nbl[1], nb_filters*4, nb_filters*2,
            batch_norm=batch_norm
        )
        self.upsample_block3 = UpsampleBlock(
            2, nb_filters*2, nb_filters,
            mode=upsampling_mode)
        self.c6 = ConvBlock(
            2, nbl[0], nb_filters*2, nb_filters,
            batch_norm=batch_norm
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
        nb_classes:
            Number of classes in the ground truth
        nb_filters:
            Number of filters in 1st convolutional block
            (gets multiplied by 2 in each next block)
        dropout:
            Use / not use dropout in the 3 inner layers
        batch_norm:
            Use / not use batch normalization after each convolutional layer
        upsampling_mode:
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
                 dropout: bool = False,
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
        dropout_vals = [.3, .3] if dropout else [0, 0]
        self.c1 = ConvBlock(
                    2, nbl[0], 1, nb_filters,
                    batch_norm=batch_norm
        )
        self.at1 = DilatedBlock(
                    2, nb_filters, nb_filters*2,
                    dilation_values=dilation_values_1,
                    padding_values=padding_values_1,
                    batch_norm=batch_norm,
                    dropout_=dropout_vals[0]
        )
        self.at2 = DilatedBlock(
                    2, nb_filters*2, nb_filters*2,
                    dilation_values=dilation_values_2,
                    padding_values=padding_values_2,
                    batch_norm=batch_norm,
                    dropout_=dropout_vals[1]
        )
        self.up1 = UpsampleBlock(
                    2, nb_filters*2, nb_filters,
                    mode=upsampling_mode
        )
        self.c2 = ConvBlock(
                    2, nbl[3], nb_filters*2, nb_filters,
                    batch_norm=batch_norm
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


class ResHedNet(nn.Module):
    """
    Holistic edge detector with residual connections in each block

    Args:
        nb_classes:
            Number of classes in the ground truth
        nb_filters:
            Number of filters in 1st residual block
            (gets multiplied by 2 in each next block)
        upsampling_mode:
            Select between "bilinear" or "nearest" upsampling method.
            Bilinear is usually more accurate,but adds additional (small)
            randomness. For full reproducibility, consider using 'nearest'
            (this assumes that all other sources of randomness are fixed)
        **layers (list):
            3-element list with a number of residual blocks
            in each segment (Default: [3, 4, 5])

    """
    def __init__(self,
                 nb_classes: int = 1,
                 nb_filters: int = 64,
                 upsampling_mode: str = "bilinear",
                 **kwargs: List[int]) -> None:
        """
        Initializes model's parameters
        """
        super(ResHedNet, self).__init__()
        nbl = kwargs.get("layers", [3, 4, 5])
        self.upsample = upsampling_mode
        self.net1 = ResModule(2, nbl[0], 1, nb_filters, True)
        self.net2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResModule(2, nbl[1], nb_filters, 2*nb_filters, True)
        )
        self.net3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResModule(2, nbl[2], 2*nb_filters, 4*nb_filters, True)
        )
        self.net1score = nn.Sequential(
            nn.Conv2d(nb_filters, nb_classes, 1, 1, 0),
            nn.BatchNorm2d(nb_classes)
        )
        self.net2score = nn.Sequential(
            nn.Conv2d(2*nb_filters, nb_classes, 1, 1, 0),
            nn.BatchNorm2d(nb_classes)
        )
        self.net3score = nn.Sequential(
            nn.Conv2d(4*nb_filters, nb_classes, 1, 1, 0),
            nn.BatchNorm2d(nb_classes)
        )
        self.out = torch.nn.Conv2d(3*nb_classes, nb_classes, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:4]
        net1out = self.net1(x)
        net2out = self.net2(net1out)
        net3out = self.net3(net2out)

        score1 = self.net1score(net1out)
        score2 = self.net2score(net2out)
        score3 = self.net3score(net3out)

        score2 = F.interpolate(score2, size=(h, w), mode=self.upsample)
        score3 = F.interpolate(score3, size=(h, w), mode=self.upsample)

        return self.out(torch.cat([score1, score2, score3], 1))


class SegResNet(nn.Module):
    '''
    Builds a fully convolutional neural network based on residual blocks
    for semantic segmentation

    Args:
        nb_classes:
            Number of classes in the ground truth
        nb_filters:
            Number of filters in 1st residual block
            (gets multiplied by 2 in each next block)
        batch_norm:
            Use batch normalization after each convolutional layer
            (Default: True)
        upsampling_mode:
            Select between "bilinear" or "nearest" upsampling method.
            Bilinear is usually more accurate,but adds additional (small)
            randomness. For full reproducibility, consider using 'nearest'
            (this assumes that all other sources of randomness are fixed)
        **layers (list):
            3-element list with a number of residual blocks
            in each residual segment (Default: [2, 2])

    '''
    def __init__(self,
                 nb_classes: int = 1,
                 nb_filters: int = 32,
                 batch_norm: bool = True,
                 upsampling_mode: bool = True,
                 **kwargs: List[int]
                 ) -> None:
        '''
        Initializes module parameters
        '''
        super(SegResNet, self).__init__()
        nbl = kwargs.get("layers", [2, 2, 2])
        self.c1 = ConvBlock(
            2, 1, 1, nb_filters, batch_norm=batch_norm
        )
        self.c2 = ResModule(
            2, nbl[0], nb_filters, nb_filters*2, batch_norm=batch_norm
        )
        self.bn = ResModule(
            2, nbl[1], nb_filters*2, nb_filters*4, batch_norm=batch_norm
        )
        self.upsample_block1 = UpsampleBlock(
            2, nb_filters*4, nb_filters*2, 2, upsampling_mode
        )
        self.c3 = ResModule(
            2, nbl[2], nb_filters*4, nb_filters*2, batch_norm=batch_norm
        )
        self.upsample_block2 = UpsampleBlock(
            2, nb_filters*2, nb_filters, 2, upsampling_mode
        )
        self.c4 = ConvBlock(
            2, 1, nb_filters*2, nb_filters, batch_norm=batch_norm
        )
        self.px = nn.Conv2d(nb_filters, nb_classes, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Defines a forward pass'''
        # Contracting path
        c1 = self.c1(x)
        d1 = F.max_pool2d(c1, kernel_size=2, stride=2)
        c2 = self.c2(d1)
        d2 = F.max_pool2d(c2, kernel_size=2, stride=2)
        # Bottleneck
        bn = self.bn(d2)
        # Expanding path
        u2 = self.upsample_block1(bn)
        u2 = torch.cat([c2, u2], dim=1)
        u2 = self.c3(u2)
        u1 = self.upsample_block2(u2)
        u1 = torch.cat([c1, u1], dim=1)
        u1 = self.c4(u1)
        # pixel-wise classification
        px = self.px(u1)
        return px


def init_fcnn_model(model: Union[Type[nn.Module], str],
                    nb_classes: int, **kwargs: [bool, int, List]
                    ) -> Type[nn.Module]:
    """
    Initializes a fully convolutional neural network
    """
    if not isinstance(model, str) and hasattr(model, "state_dict"):
        meta_state_dict = {
            'model_type': 'Seg', model: 'custom', 'nb_classes': nb_classes}
        return model, meta_state_dict
    batch_norm = kwargs.get('batch_norm', True)
    dropout = kwargs.get('dropout', False)
    upsampling = kwargs.get('upsampling', "bilinear")
    meta_state_dict = {
                'model_type': 'seg',
                'model': model,
                'nb_classes': nb_classes,
                'batch_norm': batch_norm,
                'dropout': dropout,
                'upsampling': upsampling,
            }
    if isinstance(model, str) and model == 'Unet':
        with_dilation = kwargs.get('with_dilation', False)
        nb_filters = kwargs.get('nb_filters', 16)
        layers = kwargs.get("layers", [1, 2, 2, 3])
        net = Unet(
            nb_classes, nb_filters, dropout,
            batch_norm, upsampling, with_dilation,
            layers=layers
        )
        meta_state_dict["with_dilation"] = with_dilation
    elif isinstance(model, str) and model == 'dilnet':
        nb_filters = kwargs.get('nb_filters', 25)
        layers = kwargs.get("layers", [1, 3, 3, 1])
        net = dilnet(
            nb_classes, nb_filters,
            dropout, batch_norm, upsampling,
            layers=layers
        )
    elif isinstance(model, str) and model == 'SegResNet':
        nb_filters = kwargs.get('nb_filters', 32)
        layers = kwargs.get("layers", [2, 2, 2])
        net = SegResNet(
            nb_classes, nb_filters,
            batch_norm, upsampling, layers=layers
        )
    elif isinstance(model, str) and model == 'ResHedNet':
        nb_filters = kwargs.get('nb_filters', 64)
        layers = kwargs.get("layers", [3, 4, 5])
        net = ResHedNet(
            nb_classes, nb_filters,
            upsampling, layers=layers
        )
    else:
        raise NotImplementedError(
            "Currently implemented models are 'Unet', 'dilnet', SegResNet', and 'ResHedNet'"
        )
    if model in ["ResHedNet", "SegResNet"]:
        meta_state_dict["dropout"] = None
    if model == ['ResHedNet']:
        meta_state_dict["batch_norm"] = True
    meta_state_dict["nb_filters"] = nb_filters
    meta_state_dict["layers"] = layers
    return net, meta_state_dict
