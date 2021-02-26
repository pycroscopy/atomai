import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_

sys.path.append("../../../")

from atomai.models import (Segmentor, ImSpec, VAE, rVAE, jVAE,
                           jrVAE, load_model, load_ensemble)
from atomai.trainers import EnsembleTrainer


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

def compare_optimizers(opt1, opt2):
    for group_param1, group_param2 in zip(opt1.param_groups, opt2.param_groups):
        for param1, param2 in zip(group_param1["params"], group_param1["params"]):
            for p1, p2 in zip(param1, param2):
                assert_array_equal(p1.detach().cpu().numpy(), p2.detach().cpu().numpy())

@pytest.mark.parametrize("model", ["Unet", "dilnet", "SegResNet", "ResHedNet"])
def test_io_segmentor(model):
    X, X_test = gen_image_data()
    y, y_test = gen_image_labels()
    segmodel = Segmentor(model, nb_classes=3)
    segmodel.fit(X, y, X_test, y_test, training_cycles=4, batch_size=2)
    loaded_model = load_model("model_metadict_final.tar")
    for p1, p2 in zip(loaded_model.net.parameters(), segmodel.net.parameters()):
        assert_array_equal(p1.detach().cpu().numpy(), p2.detach().cpu().numpy())


@pytest.mark.parametrize("model", ["Unet", "dilnet", "SegResNet", "ResHedNet"])
def test_saved_optimizer_segmentor(model):
    X, X_test = gen_image_data()
    y, y_test = gen_image_labels()
    segmodel = Segmentor(model, nb_classes=3)
    segmodel.fit(X, y, X_test, y_test, training_cycles=4, batch_size=2,
                 filename="segmodel")
    opt1 = segmodel.optimizer
    loaded_model = load_model("segmodel_metadict_final.tar")
    opt2 = loaded_model.optimizer
    compare_optimizers(opt1, opt2)


def test_io_imspec():
    X, X_test = gen_image_data()
    y, y_test = gen_spectra()
    i2s_model = ImSpec((8, 8), (16,))
    i2s_model.fit(X, y, X_test, y_test, training_cycles=4, batch_size=2)
    loaded_model = load_model("model_metadict_final.tar")
    for p1, p2 in zip(loaded_model.net.parameters(), i2s_model.net.parameters()):
        assert_array_equal(p1.detach().cpu().numpy(), p2.detach().cpu().numpy())


def test_saved_optimizer_imspec():
    X, X_test = gen_image_data()
    y, y_test = gen_spectra()
    i2s_model = ImSpec((8, 8), (16,))
    i2s_model.fit(X, y, X_test, y_test, training_cycles=4, batch_size=2)
    opt1 = i2s_model.optimizer
    loaded_model = load_model("model_metadict_final.tar")
    opt2 = loaded_model.optimizer
    compare_optimizers(opt1, opt2)


@pytest.mark.parametrize("model", [VAE, rVAE, jVAE, jrVAE])
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


@pytest.mark.parametrize("model", [VAE, rVAE, jVAE, jrVAE])
def test_saved_optimizer_VAE(model):
    X, _ = gen_image_data()
    X = X[:, 0, ...]
    vae_model = model((8, 8))
    vae_model.fit(X, training_cycles=4, batch_size=2, filename="vae_metadict")
    opt1 = vae_model.optim
    loaded_model = load_model("vae_metadict.tar")
    opt2 = loaded_model.optim
    compare_optimizers(opt1, opt2)

@pytest.mark.parametrize("model", [jVAE, jrVAE])
def test_saved_iter_jVAE(model):
    X, _ = gen_image_data()
    X = X[:, 0, ...]
    vae_model = model((8, 8))
    vae_model.fit(X, training_cycles=4, batch_size=2, filename="jvae_metadict")
    num_iter = vae_model.kdict_["num_iter"]
    loaded_model = load_model("jvae_metadict.tar")
    assert_equal(num_iter, loaded_model.kdict_["num_iter"])


@pytest.mark.parametrize("model", [VAE, rVAE, jVAE, jrVAE])
def test_resume_training(model):
    X, _ = gen_image_data()
    X = X[:, 0, ...]
    vae_model = model((8, 8))
    vae_model.fit(X, training_cycles=4, batch_size=2, filename="vae_metadict")
    loss0 = abs(vae_model.loss_history["train_loss"][0])
    loaded_model = load_model("vae_metadict.tar")
    loaded_model.fit(X, training_cycles=4, batch_size=2, filename="vae_metadict")
    loss1 = abs(loaded_model.loss_history["train_loss"][0])
    assert_(not np.isnan(loss1))
    assert_(loss1 < loss0)


@pytest.mark.parametrize("model", ["Unet", "dilnet", "SegResNet", "ResHedNet"])
def test_io_ensemble_seg(model):
    X, X_test = gen_image_data()
    y, y_test = gen_image_labels()

    etrainer = EnsembleTrainer(model, nb_classes=3)
    etrainer.compile_ensemble_trainer(training_cycles=4, batch_size=2)
    smodel, ensemble = etrainer.train_ensemble_from_scratch(
        X, y, X_test, y_test, n_models=3)
    smodel_, ensemble_ = load_ensemble("model_ensemble_metadict.tar")
    for i in ensemble.keys():
        m1 = ensemble[i]
        m2 = ensemble_[i]
        for p1, p2 in zip(m1.values(), m2.values()):
            assert_array_equal(
                p1.detach().cpu().numpy(),
                p2.detach().cpu().numpy())


def test_io_ensemble_imspec():
    X, X_test = gen_image_data()
    y, y_test = gen_spectra()

    etrainer = EnsembleTrainer(
        "imspec", in_dim=(8, 8), out_dim=(16,), latent_dim=2)
    etrainer.compile_ensemble_trainer(training_cycles=4, batch_size=2)
    smodel, ensemble = etrainer.train_ensemble_from_scratch(
        X, y, X_test, y_test, n_models=3)
    smodel_, ensemble_ = load_ensemble("model_ensemble_metadict.tar")
    for i in ensemble.keys():
        m1 = ensemble[i]
        m2 = ensemble_[i]
        for p1, p2 in zip(m1.values(), m2.values()):
            assert_array_equal(
                p1.detach().cpu().numpy(),
                p2.detach().cpu().numpy())
