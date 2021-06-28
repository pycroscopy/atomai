import sys

import numpy as np
import pytest
import torch
from numpy.testing import assert_, assert_equal

sys.path.append("../../../")

from atomai.utils.nn import set_seed_and_precision, channels2indices


def test_set_same_seed():
    seed = 1
    set_seed_and_precision(seed)
    a = torch.randn(2, 1, 8, 8)
    set_seed_and_precision(seed)
    b = torch.randn(2, 1, 8, 8)
    assert_(np.array_equal(a, b))


def test_set_diff_seed():
    seed = 1
    set_seed_and_precision(seed)
    a = torch.randn(2, 1, 8, 8)
    set_seed_and_precision(seed + 1)
    b = torch.randn(2, 1, 8, 8)
    assert_(not np.array_equal(a, b))


@pytest.mark.parametrize(
    "precision, dtype",
    [("single", torch.float32), ("double", torch.float64)])
def test_set_precision(precision, dtype):
    set_seed_and_precision(precision=precision)
    a = torch.randn(2, 1, 8, 8)
    assert_equal(a.dtype, dtype)


def test_channels2indices():
    a1 = np.random.randint(0, 2, size=(5, 8, 8, 1))
    a2 = np.random.randint(0, 2, size=(5, 8, 8, 1))
    a3 = np.random.randint(0, 2, size=(5, 8, 8, 1))
    a = np.concatenate([a1, a2, a3], -1)
    a_sq = channels2indices(a)
    assert_equal(a_sq.shape, (5, 8, 8))
    assert_equal(np.unique(a_sq), np.array([0., 1., 2., 3.]))
