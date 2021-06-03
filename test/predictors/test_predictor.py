import sys

import numpy as np
import torch
import pytest
from numpy.testing import assert_equal, assert_, assert_allclose

sys.path.append("../../../")

from atomai.predictors import BasePredictor, SegPredictor
from atomai.nets import ConvBlock


def init_model():
    model = torch.nn.Sequential(ConvBlock(2, 1, 1, 8), ConvBlock(2, 1, 8, 1))
    return model

#@pytest.mark.parametrize("model", [nets.Unet, nets.dilnet, nets.SegResNet, nets.ResHedNet])
def test_basepredictor_preprocess_np():
    x_np = np.random.randn(2, 8, 8)
    model = init_model()
    p = BasePredictor(model)
    x_torch = p.preprocess(x_np)
    assert_(isinstance(x_torch, torch.Tensor))
    assert_(x_torch.dtype == torch.float32)


def test_basepredictor_preprocess_torch():
    x_torch = torch.randn(2, 8, 8)
    model = init_model()
    p = BasePredictor(model)
    x_torch_ = p.preprocess(x_torch)
    assert_(isinstance(x_torch_, torch.Tensor))
    assert_(x_torch_.dtype == torch.float32)


def test_basepredictor_device_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = init_model()
    p = BasePredictor(model)
    p._model2device(device)
    assert_(next(model.parameters()).is_cuda == torch.cuda.is_available())


def test_basepredictor_device_data():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = init_model()
    p = BasePredictor(model)
    x_torch = torch.randn(2, 8, 8)
    x_torch_ = p._data2device(x_torch, device)
    assert_(x_torch_.is_cuda == torch.cuda.is_available())


def test_basepredictor_forward():
    x_torch = torch.randn(2, 1, 8, 8)
    model = init_model()
    p = BasePredictor(model)
    out = p.forward_(x_torch)
    assert_(not torch.equal(x_torch, out))


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_basepredictor_batchpredict(batch_size):
    x_torch = torch.randn(5, 1, 8, 8)
    model = init_model()
    p = BasePredictor(model)
    out = p.batch_predict(x_torch, x_torch.shape, batch_size)
    assert_(not torch.equal(x_torch, out))
    assert_(not out.is_cuda)


def test_basepredictor_predict():
    x_torch = torch.randn(2, 1, 8, 8)
    model = init_model()
    p = BasePredictor(model)
    p.predict(x_torch, x_torch.shape)
