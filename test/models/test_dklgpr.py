import sys

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_

sys.path.append("../../../")

from atomai.models import dklGPR


def test_model_fit():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPR(indim, precision="single")
    assert_equal(len(t.train_loss), 0)
    t.fit(X, y, 2)
    assert_equal(len(t.train_loss), 2)


@pytest.mark.parametrize("shared_emb", [0, 1])
def test_model_fit_ensemble(shared_emb):
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPR(indim, precision="single", shared_embedding_space=shared_emb)
    assert_equal(len(t.train_loss), 0)
    t.fit_ensemble(X, y, 2, n_models=3)
    assert_equal(len(t.train_loss), 2)

def test_model_predict():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    X_test = np.random.randn(50, indim)
    t = dklGPR(indim, precision="single")
    t.fit(X, y)
    y_pred = t.predict(X_test)
    assert_equal(len(y_pred[0]), len(y_pred[1]))
    assert_equal(y_pred[0].shape[0], X.shape[0])
    assert_(isinstance(y_pred[0], np.ndarray))
    assert_(isinstance(y_pred[1], np.ndarray))


def test_multi_model_predict():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(2, 50)
    X_test = np.random.randn(50, indim)
    t = dklGPR(indim, shared_embedding_space=False, precision="single")
    t.fit(X, y)
    y_pred = t.predict(X_test)
    assert_equal(y_pred[0].shape, y_pred[1].shape)
    assert_equal(y_pred[0].shape[1], X.shape[0])
    assert_(isinstance(y_pred[0], np.ndarray))
    assert_(isinstance(y_pred[1], np.ndarray))


@pytest.mark.parametrize("shared_emb", [0, 1])
@pytest.mark.parametrize("ydim", [(50,), (1, 50)])
def test_ensemble_predict(shared_emb, ydim):
    indim = 32
    n_models = 3
    X = np.random.randn(50, indim)
    y = np.random.randn(*ydim)
    X_test = np.random.randn(50, indim)
    t = dklGPR(indim, shared_embedding_space=shared_emb, precision="single")
    t.fit_ensemble(X, y, 1, n_models=n_models)
    y_pred = t.predict(X_test)
    assert_equal(y_pred[0].shape, y_pred[1].shape)
    assert_equal(y_pred[0].shape[1], X.shape[0])
    assert_equal(y_pred[0].shape[0], n_models)
    assert_(isinstance(y_pred[0], np.ndarray))
    assert_(isinstance(y_pred[1], np.ndarray))


@pytest.mark.parametrize("reg_dim", [1, 2])
def test_sample_from_posterior(reg_dim):
    indim = 32
    num_samples = 100
    X = np.random.randn(50, indim)
    y = np.random.randn(reg_dim, 50)
    X_test = np.random.randn(50, indim)
    t = dklGPR(indim, precision="single")
    t.fit(X, y)
    samples = t.sample_from_posterior(X_test, num_samples)
    assert_equal(samples.shape[0], num_samples)
    assert_equal(samples.shape[1], reg_dim)
    assert_equal(samples.shape[2], len(X))
    assert_(isinstance(samples, np.ndarray))


def test_sample_from_multi_model_posterior():
    indim = 32
    num_samples = 100
    X = np.random.randn(50, indim)
    y = np.random.randn(2, 50)
    X_test = np.random.randn(50, indim)
    t = dklGPR(indim, shared_embedding_space=False, precision="single")
    t.fit(X, y)
    samples = t.sample_from_posterior(X_test, num_samples)
    print(samples.shape)
    assert_equal(samples.shape[0], num_samples)
    assert_equal(samples.shape[1], 2)
    assert_equal(samples.shape[2], len(X_test))
    assert_(isinstance(samples, np.ndarray))


@pytest.mark.parametrize("reg_dim", [1, 2])
def test_thompson(reg_dim):
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(reg_dim, 50)
    X_test = np.random.randn(50, indim)
    m = dklGPR(indim, precision="single")
    m.fit(X, y)
    sample, xnext = m.thompson(X_test)
    assert_(isinstance(xnext, int))
    assert_(isinstance(sample, np.ndarray))
    assert_equal(sample.shape, (50,))


def test_thompson_scalarize():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(3, 50)
    X_test = np.random.randn(50, indim)
    m = dklGPR(indim, precision="single")
    m.fit(X, y)
    sample, xnext = m.thompson(X_test, scalarize_func=lambda x: x.mean(0))
    assert_(isinstance(xnext, int))
    assert_(isinstance(sample, np.ndarray))
    assert_equal(sample.shape, (50,))


@pytest.mark.parametrize("embedim", [1, 2])
def test_model_embed(embedim):
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    X_test = np.random.randn(50, indim)
    t = dklGPR(indim, embedim, precision="single")
    t.fit(X, y)
    y_embedded = t.embed(X_test)
    assert_equal(y_embedded.shape[0], X_test.shape[0])
    assert_equal(y_embedded.shape[1], embedim)
    assert_(isinstance(y_embedded, np.ndarray))


@pytest.mark.parametrize("embedim", [1, 2])
def test_multi_model_embed(embedim):
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(3, 50)
    X_test = np.random.randn(50, indim)
    t = dklGPR(indim, embedim, shared_embedding_space=False, precision="single")
    t.fit(X, y)
    y_embedded = t.embed(X_test)
    print(y_embedded.shape)
    assert_equal(y_embedded.shape[0], 3)
    assert_equal(y_embedded.shape[1], X_test.shape[0])
    assert_equal(y_embedded.shape[2], embedim)
    assert_(isinstance(y_embedded, np.ndarray))


@pytest.mark.parametrize("embedim", [1, 2])
def test_model_decode_scalar(embedim):
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPR(indim, embedim, precision="single")
    t.fit(X, y)
    z = np.random.randn(2, embedim)
    decoded = t.decode(z)
    assert_equal(decoded[0].shape, (2, 1))


@pytest.mark.parametrize("embedim", [1, 2])
def test_model_decode_vector(embedim):
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(3, 50)
    t = dklGPR(indim, embedim, precision="single")
    t.fit(X, y)
    z = np.random.randn(2, embedim)
    decoded = t.decode(z)
    assert_equal(decoded[0].shape, (2, 3))
