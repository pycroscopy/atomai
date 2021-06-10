import sys

import numpy as np
import torch
import pytest
from numpy.testing import assert_

sys.path.append("../../../")

from atomai.predictors import BasePredictor, SegPredictor, ImSpecPredictor
from atomai.nets import ConvBlock, Unet, ResHedNet, SegResNet, dilnet, SignalED


def init_model():
    model = torch.nn.Sequential(ConvBlock(2, 1, 1, 8), ConvBlock(2, 1, 8, 1))
    return model


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
    out = p.predict(x_torch, x_torch.shape)
    assert_(not torch.equal(x_torch, out))
    assert_(not out.is_cuda)


@pytest.mark.parametrize("model", [Unet, dilnet, SegResNet, ResHedNet])
@pytest.mark.parametrize("shape", [(2, 8, 8), (8, 8)])
def test_SegPredictor_predict(model, shape):
    x_np = np.random.randn(*shape)
    p = SegPredictor(model())
    out = p.predict(x_np)
    assert_(isinstance(out, np.ndarray))
    assert_(not np.array_equal(x_np, out))


@pytest.mark.parametrize("shape", [(2, 8, 8), (8, 8)])
def test_ImSpecPredictor_predict(shape):
    x_np = np.random.randn(*shape)
    model = SignalED((8, 8), (16,), 2)
    p = ImSpecPredictor(model, (16,))
    out = p.predict(x_np)
    assert_(isinstance(out, np.ndarray))
    assert_(out.shape, (16,))


@pytest.mark.parametrize("shape", [(2, 16), (16,)])
def test_SpecImPredictor_predict(shape):
    x_np = np.random.randn(*shape)
    model = SignalED((16,), (8, 8), 2)
    p = ImSpecPredictor(model, (8, 8))
    out = p.predict(x_np)
    assert_(isinstance(out, np.ndarray))
    assert_(out.shape, (8, 8))


@pytest.mark.parametrize("model", [Unet, dilnet, SegResNet, ResHedNet])
@pytest.mark.parametrize("shape", [(2, 8, 8), (8, 8)])
def test_SegPredictor_run_wcoord(model, shape):
    x_np = np.random.randn(*shape)
    p = SegPredictor(model())
    out = p.run(x_np, compute_coords=True)
    assert_(len(out) == 2)


@pytest.mark.parametrize("model", [Unet, dilnet, SegResNet, ResHedNet])
@pytest.mark.parametrize("shape", [(2, 8, 8), (8, 8)])
def test_SegPredictor_run_nocoord(model, shape):
    x_np = np.random.randn(*shape)
    p = SegPredictor(model())
    out = p.run(x_np, compute_coords=False)
    assert_(isinstance(out, np.ndarray))


@pytest.mark.parametrize("shape", [(2, 8, 8), (8, 8)])
def test_ImSpecPredictor_run(shape):
    x_np = np.random.randn(*shape)
    model = SignalED((8, 8), (16,), 2)
    p = ImSpecPredictor(model, (16,))
    out = p.run(x_np)
    assert_(isinstance(out, np.ndarray))
    assert_(out.shape, (16,))
