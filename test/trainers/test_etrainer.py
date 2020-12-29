import sys

import numpy as np
import pytest
from numpy.testing import assert_

sys.path.append("../../../")

from atomai.trainers import EnsembleTrainer


def gen_image_data():
    """
    Dummy images with random pixels
    """
    X = np.random.random(size=(5, 1, 8, 8))
    X_ = np.random.random(size=(5, 1, 8, 8))
    return X, X_


def gen_image_labels(binary=False):
    """
    Dummy labels for dummy images
    """
    if binary:
        y = np.random.randint(0, 2, size=(5, 1, 8, 8))
        y_ = np.random.randint(0, 2, size=(5, 1, 8, 8))
    else:
        y = np.random.randint(0, 3, size=(5, 8, 8))
        y_ = np.random.randint(0, 3, size=(5, 8, 8))
    return y, y_


def gen_spectra():
    """
    Dummy 1D signal with random points
    """
    X = np.random.random(size=(5, 1, 16))
    X_ = np.random.random(size=(5, 1, 16))
    return X, X_


def assert_weights_equal(m1, m2):
    eq_w = []
    for p1, p2 in zip(m1.values(), m2.values()):
        eq_w.append(np.array_equal(
            p1.detach().cpu().numpy(),
            p2.detach().cpu().numpy()))
    return all(eq_w)


@pytest.mark.parametrize("full_epoch", [0, 1])
@pytest.mark.parametrize("binary", [1, 0])
@pytest.mark.parametrize("model", ["Unet", "dilnet", "SegResNet", "ResHedNet"])
def test_ensemble_seg(model, binary, full_epoch):
    ncls = 1 if binary else 3
    X, X_test = gen_image_data()
    y, y_test = gen_image_labels(binary=binary)
    etrainer = EnsembleTrainer(model, nb_classes=ncls, upsampling="nearest")
    etrainer.compile_ensemble_trainer(
        training_cycles=4, full_epoch=full_epoch, batch_size=2)
    smodel, ensemble = etrainer.train_ensemble_from_scratch(
        X, y, X_test, y_test, n_models=3)
    m_eq = []
    m_not_eq = []
    for i in ensemble.keys():
        for j in ensemble.keys():
            assrtn = assert_weights_equal(ensemble[i], ensemble[j])
            if i == j:
                m_eq.append(assrtn)
            else:
                m_not_eq.append(assrtn)
    assert_(all(m_eq))
    assert_(not any(m_not_eq))


@pytest.mark.parametrize("full_epoch", [0, 1])
def test_ensemble_imspec(full_epoch):
    X, X_test = gen_image_data()
    y, y_test = gen_spectra()
    etrainer = EnsembleTrainer(
        "imspec", in_dim=(8, 8), out_dim=(16,), latent_dim=2)
    etrainer.compile_ensemble_trainer(
        training_cycles=4, full_epoch=full_epoch, batch_size=2)
    smodel, ensemble = etrainer.train_ensemble_from_scratch(
        X, y, X_test, y_test, n_models=3)
    m_eq = []
    m_not_eq = []
    for i in ensemble.keys():
        for j in ensemble.keys():
            assrtn = assert_weights_equal(ensemble[i], ensemble[j])
            if i == j:
                m_eq.append(assrtn)
            else:
                m_not_eq.append(assrtn)
    assert_(all(m_eq))
    assert_(not any(m_not_eq))
