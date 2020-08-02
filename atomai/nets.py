"""
models.py
=========

Deep learning models and various custom NN blocks

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""
from typing import Dict, List, Tuple, Type

import numpy as np
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
            number of neurons in each fully connnected layer (when mlp=True)
            or number of filters in each convolutional layer (when mlp=False, default)
        mlp (bool):
            using a simple multi-layer perceptron instead of convolutional layers (Default: False)

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
    Spatial decoder network (for rotationally-invariant variational autoencoder)
    (based on https://arxiv.org/abs/1909.11663)

    Args:
        latent_dim (int):
            number of latent dimensions associated with images content
        num_layers (int):
            number of fully connected layers
        hidden_dim (int):
            number of neurons in each fully connected layer
        out_dim (tuple):
            output dimensions: (height, width) or (height, width, channels)
    """
    def __init__(self,
                 latent_dim: int,
                 num_layers: int,
                 hidden_dim: int,
                 out_dim: Tuple[int]) -> None:
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
        self.fc_coord = nn.Linear(2, hidden_dim)
        self.fc_latent = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.activation = nn.Tanh()
        fc_decoder = []
        for i in range(num_layers):
            fc_decoder.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        self.fc_decoder = nn.Sequential(*fc_decoder)
        self.out = nn.Linear(hidden_dim, c)

    def forward(self, x_coord: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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
        h = self.activation(h)
        h = self.fc_decoder(h)
        h = self.out(h)
        out = h.reshape(batch_dim, *self.reshape_)
        if self.apply_softplus:
            return F.softplus(out)
        return out


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
            using a simple multi-layer perceptron instead of convolutional layers (Default: False)
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


def load_model(meta_state_dict: str) -> Type[torch.nn.Module]:
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
    if "with_dilation" in meta_dict.keys():
        (model_type, batchnorm, dropout, upsampling,
         nb_filters, layers, nb_classes, checkpoint,
         with_dilation) = meta_dict.values()
    else:
        (model_type, batchnorm, dropout, upsampling,
         nb_filters, layers, nb_classes, checkpoint) = meta_dict.values()
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


def load_ensemble(meta_state_dict: str) -> Tuple[Type[torch.nn.Module], Dict[int, Dict[str, torch.Tensor]]]:
    """
    Loads trained ensemble models

    Args:
        meta_state_dict (str):
            filepath to dictionary with trained weights and key information
            about model's structure

    Returns:
        Model skeleton (initialized) and dictionary with weights of all the models
    """
    torch.manual_seed(0)
    if torch.cuda.device_count() > 0:
        meta_dict = torch.load(meta_state_dict)
    else:
        meta_dict = torch.load(meta_state_dict, map_location='cpu')
    if "with_dilation" in meta_dict.keys():
        (model_type, batchnorm, dropout, upsampling,
         nb_filters, layers, nb_classes, checkpoint,
         with_dilation) = meta_dict.values()
    else:
        (model_type, batchnorm, dropout, upsampling,
         nb_filters, layers, nb_classes, checkpoint) = meta_dict.values()
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
            "The network must be either 'dilUnet' or 'dilnet'"
        )
    model.load_state_dict(checkpoint[0])
    return model.eval(), checkpoint
