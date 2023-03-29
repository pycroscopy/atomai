import sys
import pytest
import torch
from numpy.testing import assert_

sys.path.append("../../../")

from atomai.nets import RegressorNet, ClassifierNet


@pytest.mark.parametrize("backbone_type", ["resnet", "vgg", "mobilenet"])
def test_regressor_net(backbone_type):
    input_channels = 3
    output_size = 1
    input_tensor = torch.randn(4, input_channels, 224, 224)

    model = RegressorNet(input_channels, output_size, backbone_type)
    output = model(input_tensor)
    assert_(output.shape == (input_tensor.shape[0], output_size))


@pytest.mark.parametrize("backbone_type", ["resnet", "vgg", "mobilenet"])
def test_classifier_net(backbone_type):
    input_channels = 3
    num_classes = 10
    input_tensor = torch.randn(4, input_channels, 224, 224)

    model = ClassifierNet(input_channels, num_classes, backbone_type)
    output = model(input_tensor)
    assert_(output.shape == (input_tensor.shape[0], num_classes))
    assert_(torch.allclose(torch.exp(output).sum(dim=1), torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32), atol=1e-6))
