import sys

import numpy as np
from numpy.testing import assert_equal, assert_

sys.path.append("../../../")

from atomai.models import Reconstructor


def test_model_reconstruct():
    X = np.abs(np.random.randn(10, 10)) + 1
    X[2] = 0
    t = Reconstructor()
    assert_equal(len(t.train_loss), 0)
    y_pred = t.reconstruct(X, 2)
    assert_equal(len(t.train_loss), 2)
    assert_(isinstance(y_pred, np.ndarray))
    assert_(not np.array_equal(X, y_pred))