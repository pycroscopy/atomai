"""
Deep learning models and various custom NN blocks

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class dilUnet(nn.Module):
    '''
    Builds a fully convolutional Unet-like neural network model

    Args:
        nb_classes: int
            number of classes in the ground truth
        nb_filters: int
            number of filters in 1st convolutional block
            (gets multibplied by 2 in each next block)
        use_dropout: bool
            use / not use dropout in the 3 inner layers
        batch_norm: bool
            use / not use batch normalization after each convolutional layer
    '''
    def __init__(self,
                 nb_classes=1,
                 nb_filters=16,
                 with_dilation=True,
                 use_dropout=False,
                 batch_norm=True):
        super(dilUnet, self).__init__()
        dropout_vals = [.1, .2, .1] if use_dropout else [0, 0, 0]
        self.c1 = conv2dblock(
            1, 1, nb_filters,
            use_batchnorm=batch_norm
        )
        self.c2 = conv2dblock(
            2, nb_filters, nb_filters*2,
            use_batchnorm=batch_norm
        )
        self.c3 = conv2dblock(
            2, nb_filters*2, nb_filters*4,
            use_batchnorm=batch_norm,
            dropout_=dropout_vals[0]
        )
        if with_dilation:
            self.bn = dilated_block(
                nb_filters*4, nb_filters*8,
                dilation_values=[2, 4, 6],
                padding_values=[2, 4, 6],
                use_batchnorm=batch_norm,
                dropout_=dropout_vals[1]
            )
        else:
            self.bn = conv2dblock(
                3, nb_filters*4, nb_filters*8,
                use_batchnorm=batch_norm,
                dropout_=dropout_vals[1]
            )
        self.upsample_block1 = upsample_block(
            nb_filters*8, nb_filters*4)
        self.c4 = conv2dblock(
            2, nb_filters*8, nb_filters*4,
            use_batchnorm=batch_norm,
            dropout_=dropout_vals[2]
        )
        self.upsample_block2 = upsample_block(
            nb_filters*4, nb_filters*2)
        self.c5 = conv2dblock(
            2, nb_filters*4, nb_filters*2,
            use_batchnorm=batch_norm
        )
        self.upsample_block3 = upsample_block(
            nb_filters*2, nb_filters)
        self.c6 = conv2dblock(
            1, nb_filters*2, nb_filters,
            use_batchnorm=batch_norm
        )
        self.px = nn.Conv2d(nb_filters, nb_classes, 1, 1, 0)

    def forward(self, x):
        '''Defines a forward path'''
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
    '''
    Builds  a fully convolutional neural network model
    by utilizing a combination of regular and dilated convolutions

    Args:
        nb_classes: int
            number of classes in the ground truth
        nb_filters: int
            number of filters in 1st convolutional block
            (gets multiplied by 2 in each next block)
        use_dropout: bool
            use / not use dropout in the 3 inner layers
        batch_norm: bool
            use / not use batch normalization after each convolutional layer
    '''

    def __init__(self,
                 nb_classes=1,
                 nb_filters=25,
                 use_dropout=False,
                 batch_norm=True):
        super(dilnet, self).__init__()
        dropout_vals = [.3, .3] if use_dropout else [0, 0]
        self.c1 = conv2dblock(3, 1, nb_filters,
                              use_batchnorm=batch_norm)
        self.at1 = dilated_block(
                    nb_filters, nb_filters*2,
                    dilation_values=[2, 4, 6], padding_values=[2, 4, 6],
                    use_batchnorm=batch_norm, dropout_=dropout_vals[0]
        )
        self.at2 = dilated_block(
                    nb_filters*2, nb_filters*2,
                    dilation_values=[2, 4, 6], padding_values=[2, 4, 6],
                    use_batchnorm=batch_norm, dropout_=dropout_vals[1]
        )
        self.up1 = upsample_block(nb_filters*2, nb_filters)
        self.c2 = conv2dblock(3, nb_filters*2, nb_filters,
                              use_batchnorm=batch_norm)
        self.px = nn.Conv2d(nb_filters, nb_classes, 1, 1, 0)

    def forward(self, x):
        '''Defines a forward path'''
        c1 = self.c1(x)
        d1 = F.max_pool2d(c1, kernel_size=2, stride=2)
        at1 = self.at1(d1)
        at2 = self.at2(at1)
        u1 = self.up1(at2)
        u1 = torch.cat([c1, u1], dim=1)
        u1 = self.c2(u1)
        px = self.px(u1)
        return px


class conv2dblock(nn.Module):
    '''
    Creates block(s) consisting of convolutional
    layer, leaky relu and (optionally) dropout and
    batch normalization
    '''
    def __init__(self, nb_layers, input_channels, output_channels,
                 kernel_size=3, stride=1, padding=1, use_batchnorm=False,
                 lrelu_a=0.01, dropout_=0):
        '''Initializes module parameters'''
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

    def forward(self, x):
        '''Forward path'''
        output = self.block(x)
        return output


class upsample_block(nn.Module):
    '''
    Defines upsampling block performed using
    bilinear interpolation followed by 1-by-1
    convolution (the latter can be used to reduce
    a number of feature channels)
    '''
    def __init__(self, input_channels, output_channels, scale_factor=2):
        '''Initializes module parameters'''
        super(upsample_block, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(
            input_channels, output_channels,
            kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''Defines a forward path'''
        x = F.interpolate(
            x, scale_factor=self.scale_factor,
            mode='bilinear', align_corners=False)
        return self.conv(x)


class dilated_block(nn.Module):
    '''
    Creates a "pyramid" with dilated convolutional
    layers (aka atrous convolutions)
    '''
    def __init__(self, input_channels, output_channels,
                 dilation_values, padding_values,
                 kernel_size=3, stride=1, lrelu_a=0.01,
                 use_batchnorm=False, dropout_=0):
        """Initializes module parameters"""
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

    def forward(self, x):
        '''Forward path'''
        atrous_layers = []
        for conv_layer in self.atrous_module:
            x = conv_layer(x)
            atrous_layers.append(x.unsqueeze(-1))
        return torch.sum(torch.cat(atrous_layers, dim=-1), dim=-1)
