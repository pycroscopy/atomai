"""
models.py
=========

Deep learning models and various custom NN blocks

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 nb_classes=1,
                 nb_filters=16,
                 use_dropout=False,
                 batch_norm=True,
                 upsampling_mode="bilinear",
                 with_dilation=True,
                 **kwargs):
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

    def forward(self, x):
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
                 nb_classes=1,
                 nb_filters=25,
                 use_dropout=False,
                 batch_norm=True,
                 upsampling_mode="bilinear",
                 **kwargs):
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

    def forward(self, x):
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
    def __init__(self, nb_layers, input_channels, output_channels,
                 kernel_size=3, stride=1, padding=1, use_batchnorm=False,
                 lrelu_a=0.01, dropout_=0):
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

    def forward(self, x):
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
                 input_channels,
                 output_channels,
                 scale_factor=2,
                 mode="bilinear"):
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

    def forward(self, x):
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
    def __init__(self, input_channels, output_channels,
                 dilation_values, padding_values,
                 kernel_size=3, stride=1, lrelu_a=0.01,
                 use_batchnorm=False, dropout_=0):
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

    def forward(self, x):
        """
        Defines a forward pass
        """
        atrous_layers = []
        for conv_layer in self.atrous_module:
            x = conv_layer(x)
            atrous_layers.append(x.unsqueeze(-1))
        return torch.sum(torch.cat(atrous_layers, dim=-1), dim=-1)


def load_model(meta_state_dict):
    """
    Loads trained AtomAI models

    Args:
        meta_state_dict (str):
            filepath to dictionary with trained weights and key information
            about model's structure (stored during and after model training
            with atomnet.trainer)

    Returns:
        Model in evaluation state
    """
    torch.manual_seed(0)
    if torch.cuda.device_count() > 0:
        meta_dict = torch.load(meta_state_dict)
    else:
        meta_dict = torch.load(meta_state_dict, map_location='cpu')
    model_type = meta_dict['model_type']
    batchnorm = meta_dict['batchnorm']
    dropout = meta_dict['dropout']
    upsampling = meta_dict['upsampling']
    nb_filters = meta_dict['nb_filters']
    nb_classes = meta_dict['nb_classes']
    checkpoint = meta_dict['weights']
    layers = meta_dict["layers"]
    if "with_dilation" in meta_dict.keys():
        with_dilation = meta_dict["with_dilation"]
    if model_type == 'dilUnet':
        model = dilUnet(
            nb_classes, nb_filters, dropout,
            batchnorm, upsampling, with_dilation,
            layers=layers)
    elif model_type == 'dilnet':
        model = dilnet(
            nb_classes, nb_filters, dropout,
            batchnorm, upsampling, layers=layers)
    else:
        raise NotImplementedError(
            "Select between 'dilUnet' and 'dilnet' neural networks"
        )
    model.load_state_dict(checkpoint)
    return model.eval()

