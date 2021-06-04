import sys

import numpy as np
import pytest
from numpy.testing import assert_

sys.path.append("../../../")

from atomai.transforms.imaug import datatransform


@pytest.mark.parametrize(
    "transforms",
    [{"gauss_noise": True}, {"poisson_noise": True}, {"salt_and_pepper": True},
     {"contrast": True}, {"jitter": True}, {"background": True}])
def test_individual_noise_transforms(transforms):
    X = np.random.randn(5, 8, 8)
    y = np.random.randint(0, 2, size=(5, 8, 8, 1))
    X_t, y_t = datatransform(1, **transforms).run(X, y)
    assert_(np.array_equal(y.squeeze(), y_t.squeeze()))
    assert_(not np.array_equal(X, X_t.squeeze()))


@pytest.mark.parametrize(
    "transforms",
    [{"rotation": True}, {"rotation": True, "zoom": True}])
def test_individual_affine_transforms(transforms):
    X = np.random.randn(2, 32, 32)
    y = np.random.randint(0, 2, size=(2, 32, 32, 1)).astype(np.float64)
    X_t, y_t = datatransform(1, **transforms).run(X, y)
    assert_(not np.array_equal(y.squeeze(), y_t.squeeze()))
    assert_(not np.array_equal(X, X_t.squeeze()))


def test_multiple_transforms():
    transforms = {"gauss_noise": True, "poisson_noise": True, "zoom": True}
    X = np.random.randn(2, 32, 32)
    y = np.random.randint(0, 2, size=(2, 32, 32, 1)).astype(np.float64)
    X_t, y_t = datatransform(1, **transforms).run(X, y)
    assert_(not np.array_equal(X, X_t.squeeze()))
