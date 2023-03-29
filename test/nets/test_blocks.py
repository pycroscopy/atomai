import sys
import pytest
import torch
from numpy.testing import assert_

sys.path.append("../../../")

from atomai.nets import CustomBackbone


def test_custom_backbone_resnet():
    input_channels = 3
    model = CustomBackbone(input_channels, backbone_type="resnet")
    x = torch.randn(4, input_channels, 224, 224)
    y = model(x)
    assert_(y.shape == (4, model.in_features, 1, 1))


def test_custom_backbone_vgg():
    input_channels = 3
    model = CustomBackbone(input_channels, backbone_type="vgg")
    x = torch.randn(4, input_channels, 224, 224)
    y = model(x)
    assert y.shape == (4, model.backbone_layers(x).size(1), 1, 1)


def test_custom_backbone_mobilenet():
    input_channels = 3
    model = CustomBackbone(input_channels, backbone_type="mobilenet")
    x = torch.randn(4, input_channels, 224, 224)
    y = model(x)
    assert_(y.shape == (4, model.in_features, 1, 1))

def test_custom_backbone_invalid_backbone_type():
    input_channels = 3
    with pytest.raises(ValueError):
        model = CustomBackbone(input_channels, backbone_type="invalid_backbone")
