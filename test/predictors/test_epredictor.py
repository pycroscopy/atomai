import sys

import pytest
import numpy as np
from numpy.testing import assert_equal

sys.path.append("../../../")

from atomai.predictors import EnsemblePredictor
from atomai.trainers import EnsembleTrainer



def gen_image_data():
    """
    Dummy images with random pixels
    """
    X = np.random.random(size=(5, 1, 32, 32))
    X_ = np.random.random(size=(5, 1, 32, 32))
    return X, X_


def gen_image_labels():
    """
    Dummy labels for dummy images
    """
    y = np.random.randint(0, 3, size=(5, 32, 32))
    y_ = np.random.randint(0, 3, size=(5, 32, 32))
    return y, y_


def gen_spectra():
    """
    Dummy 1D signal with random points
    """
    X = np.random.random(size=(5, 1, 16))
    X_ = np.random.random(size=(5, 1, 16))
    return X, X_


@pytest.mark.parametrize("model", ["Unet", "dilnet", "ResHedNet"])
def test_epredictor_seg(model):
    X, X_test = gen_image_data()
    y, y_test = gen_image_labels()
    etrainer = EnsembleTrainer(model, batch_norm=False, nb_classes=3)
    etrainer.compile_ensemble_trainer(
        training_cycles=32, batch_size=2, compute_accuracy=False)
    smodel, ensemble = etrainer.train_swag(X, y, X_test, y_test, n_models=7)
    p = EnsemblePredictor(smodel, ensemble, nb_classes=3)
    nn_out_mean, nn_out_var = p.predict(X_test)
    assert_equal(nn_out_mean.shape, nn_out_var.shape)
    assert_equal(nn_out_mean.shape, (*y.shape, 3))


def test_epredictor_imspec():
    X, X_test = gen_image_data()
    y, y_test = gen_spectra()
    etrainer = EnsembleTrainer(
        "imspec", in_dim=(32, 32), out_dim=(16,), latent_dim=5)
    etrainer.compile_ensemble_trainer(
        training_cycles=32, batch_size=2, loss="mse")
    smodel, ensemble = etrainer.train_swag(X, y, X_test, y_test, n_models=7)
    p = EnsemblePredictor(smodel, ensemble,
                          data_dype="image", output_type="spectra",
                          in_dim=(32, 32), out_dim=(16,))
    nn_out_mean, nn_out_var = p.predict(X_test, norm=False)
    assert_equal(nn_out_mean.shape, nn_out_var.shape)
    assert_equal(nn_out_mean.squeeze().shape, y.squeeze().shape)

