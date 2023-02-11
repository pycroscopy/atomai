import sys
import os
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_
import matplotlib
matplotlib.use('Agg')

sys.path.append("../../../")

from atomai.stat import imlocal


test_coord_m_ = os.path.join(
    os.path.dirname(__file__), 'test_data/test_coord_m.npy')
test_nn_output_ = os.path.join(
    os.path.dirname(__file__), 'test_data/test_output_m.npy')
test_pca_ = os.path.join(
    os.path.dirname(__file__), 'test_data/test_pca.npy')
test_ica_ = os.path.join(
    os.path.dirname(__file__), 'test_data/test_ica.npy')
test_nmf_ = os.path.join(
    os.path.dirname(__file__), 'test_data/test_nmf.npy')


@pytest.fixture
def imstack_():
    test_nn_output = np.load(test_nn_output_)
    test_coord_m = np.load(test_coord_m_)
    test_coord_m = {'0': test_coord_m}
    imstack = imlocal(
        test_nn_output, test_coord_m, window_size=32, coord_class=1)
    return imstack


@pytest.mark.parametrize("n", [3, 4])
def test_pca(imstack_, n):
    components, Xt, coord = imstack_.pca(n)
    assert_equal(components.shape, (n, 32, 32, 3))
    assert_equal(Xt.shape, (2833, n))
    assert_equal(coord.shape, (2833, 3))


@pytest.mark.parametrize("n", [3, 4])
def test_ica(imstack_, n):
    components, Xt, coord = imstack_.ica(n)
    assert_equal(components.shape, (n, 32, 32, 3))
    assert_equal(Xt.shape, (2833, n))
    assert_equal(coord.shape, (2833, 3))


@pytest.mark.parametrize("n", [3, 4])
def test_nmf(imstack_, n):
   components, Xt, coord = imstack_.nmf(n)
   assert_equal(components.shape, (n, 32, 32, 3))
   assert_equal(Xt.shape, (2833, n))
   assert_equal(coord.shape, (2833, 3))

