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


@pytest.mark.parametrize("embedim", [1, 2])
def test_model_embed(embedim):
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    X_test = np.random.randn(50, indim)
    t = dklGPR(indim, embedim, precision="single")
    t.fit(X, y)
    y_embedded = t.embed(X_test)
    assert_equal(y_embedded.shape[0], X.shape[0])
    assert_equal(y_embedded.shape[1], embedim)
    assert_(isinstance(y_embedded, np.ndarray))
