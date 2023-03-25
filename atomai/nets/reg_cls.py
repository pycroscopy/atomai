"""
reg_cls.py
==========

Neural nets for regression and classification tasks

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import torch
import torch.nn as nn
from .blocks import CustomBackbone


class RegressorNet(nn.Module):
    """
    Regressor network using a custom backbone.

    Args:
        input_channels (int): The number of input channels.
        output_size (int): The size of the output tensor.
        backbone_type (str, optional): The type of backbone architecture. Choose from "resnet", "vgg", or "mobilenet". Default is "mobilenet".
    """
    def __init__(self, input_channels: int, output_size: int, backbone_type: str = "mobilenet"):
        super(RegressorNet, self).__init__()

        # Create the backbone with adaptive pooling
        self.backbone = CustomBackbone(input_channels, backbone_type)
        # Create the output layer
        self.output_layer = nn.Linear(self.backbone.in_features, output_size)
        # Flatten layer
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the RegressorNet.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, output_size).
        """
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.output_layer(x)
        return x


class ClassifierNet(nn.Module):
    """
    Classifier network using a custom backbone.

    Args:
        input_channels (int): The number of input channels.
        num_classes (int): The number of output classes.
        backbone_type (str, optional): The type of backbone architecture. Choose from "resnet", "vgg", or "mobilenet". Default is "resnet".
    """
    def __init__(self, input_channels: int, num_classes: int, backbone_type: str = "resnet"):
        super(ClassifierNet, self).__init__()

        # Create the backbone with adaptive pooling
        self.backbone = CustomBackbone(input_channels, backbone_type)
        # Create the output layer
        self.output_layer = nn.Sequential(
                nn.Linear(self.backbone.in_features, num_classes),
                nn.LogSoftmax(dim=1)
            )
        # Flatten layer
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the ClassifierNet.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, num_classes).
        """
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.output_layer(x)
        return x
