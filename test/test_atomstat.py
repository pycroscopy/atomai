import os
import numpy as np
from atomai.core import atomstat
import pytest
from numpy.testing import assert_allclose
import matplotlib
matplotlib.use('Agg')


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
    imstack = atomstat.imlocal(
        test_nn_output, test_coord_m, crop_size=32, coord_class=1)
    return imstack


def test_pca(imstack_):
    test_pca = np.load(test_pca_, allow_pickle=True)
    components_desired, Xt_desired = test_pca[0:2]
    components, Xt, _ = imstack_.pca(4)
    assert_allclose(components, components_desired)
    assert_allclose(Xt, Xt_desired)


def test_ica(imstack_):
    test_ica = np.load(test_ica_, allow_pickle=True)
    components_desired, Xt_desired = test_ica[0:2]
    components, Xt, _ = imstack_.ica(4)
    assert_allclose(components, components_desired)
    assert_allclose(Xt, Xt_desired)
  
    
def test_nmf(imstack_):
    test_nmf = np.load(test_nmf_, allow_pickle=True)
    components_desired, Xt_desired = test_nmf[0:2]
    components, Xt, _ = imstack_.nmf(4)
    assert_allclose(components, components_desired)
    assert_allclose(Xt, Xt_desired)