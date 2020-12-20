import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal

sys.path.append("../../../")

from atomai.models import Segmentor, ImSpec, VAE, rVAE, load_model


def gen_image_data():
    """
    Dummy images with random pixels
    """
    X = np.random.random(size=(5, 1, 8, 8))
    X_ = np.random.random(size=(5, 1, 8, 8))
    return X, X_


def gen_image_labels():
    """
    Dummy labels for dummy images
    """
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


@pytest.mark.parametrize("model", ["Unet", "dilnet", "SegResNet", "ResHedNet"])
def test_io_segmentor(model):
    X, X_test = gen_image_data()
    y, y_test = gen_image_labels()
    segmodel = Segmentor(model, nb_classes=3)
    segmodel.fit(X, y, X_test, y_test, training_cycles=4, batch_size=2)
    loaded_model = load_model("model_metadict_final.tar")
    for p1, p2 in zip(loaded_model.net.parameters(), segmodel.net.parameters()):
        assert_array_equal(p1.detach().cpu().numpy(), p2.detach().cpu().numpy())


def test_io_imspec():
    X, X_test = gen_image_data()
    y, y_test = gen_spectra()
    i2s_model = ImSpec((8, 8), (16,))
    i2s_model.fit(X, y, X_test, y_test, training_cycles=4, batch_size=2)
    loaded_model = load_model("model_metadict_final.tar")
    for p1, p2 in zip(loaded_model.net.parameters(), i2s_model.net.parameters()):
        assert_array_equal(p1.detach().cpu().numpy(), p2.detach().cpu().numpy())


@pytest.mark.parametrize("model", [VAE, rVAE])
def test_io_VAE(model):
    X, _ = gen_image_data()
    X = X[:, 0, ...]
    vae_model = model((8, 8))
    vae_model.fit(X, training_cycles=4, batch_size=2, filename="vae_metadict")
    loaded_model = load_model("vae_metadict.tar")
    for p1, p2 in zip(loaded_model.encoder_net.parameters(),
                      vae_model.encoder_net.parameters()):
        assert_array_equal(p1.detach().cpu().numpy(), p2.detach().cpu().numpy())
    for p1, p2 in zip(loaded_model.decoder_net.parameters(),
                      vae_model.decoder_net.parameters()):
        assert_array_equal(p1.detach().cpu().numpy(), p2.detach().cpu().numpy())
