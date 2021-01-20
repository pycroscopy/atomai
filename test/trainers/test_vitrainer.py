import sys

import numpy as np
import torch
import pytest
from numpy.testing import assert_equal, assert_, assert_allclose

sys.path.append("../../../")

from atomai.trainers import viBaseTrainer
from atomai.nets import fcEncoderNet, fcDecoderNet, convEncoderNet, convDecoderNet
from atomai.losses_metrics import vae_loss


def gen_image_data(torch_format=True):
    """
    Dummy image with random pixels
    """
    X = np.random.random(size=(100, 28, 28))
    if torch_format:
        X = torch.from_numpy(X).float()
    return X


class simple_vae(viBaseTrainer):

    def __init__(self, in_dim):
        super(simple_vae, self).__init__()
        self.in_dim = in_dim

    def elbo_fn(self, x, x_reconstr, *args):
        return vae_loss("mse", self.in_dim, x, x_reconstr, *args)

    def forward_compute_elbo(self, x, y=None, mode="train"):
        if mode == "eval":
            with torch.no_grad():
                z_mean, z_logsd = self.encoder_net(x)
        else:
            z_mean, z_logsd = self.encoder_net(x)
        z_sd = torch.exp(z_logsd)
        z = self.reparameterize(z_mean, z_sd)
        if mode == "eval":
            with torch.no_grad():
                x_reconstr = self.decoder_net(z)
        else:
            x_reconstr = self.decoder_net(z)
        return self.elbo_fn(x, x_reconstr, z_mean, z_logsd)


@pytest.mark.parametrize("encoder", [fcEncoderNet, convEncoderNet])
@pytest.mark.parametrize("decoder", [fcDecoderNet, convDecoderNet])
def test_set_nets(encoder, decoder):
    in_dim = (28, 28)
    v = viBaseTrainer()
    encoder_ = encoder(in_dim, 2)
    decoder_ = decoder(in_dim, 2)
    v.set_model(encoder_, decoder_)
    assert_(hasattr(v.encoder_net, "state_dict"))
    assert_(hasattr(v.decoder_net, "state_dict"))


@pytest.mark.parametrize("encoder", [fcEncoderNet, convEncoderNet])
@pytest.mark.parametrize("decoder", [fcDecoderNet, convDecoderNet])
def test_set_nets_separately(encoder, decoder):
    in_dim = (28, 28)
    v = viBaseTrainer()
    v.set_encoder(encoder(in_dim, 2))
    v.set_decoder(decoder(in_dim, 2))
    assert_(hasattr(v.encoder_net, "state_dict"))
    assert_(hasattr(v.decoder_net, "state_dict"))


@pytest.mark.parametrize("torch_format", [True, False])
def test_set_data(torch_format):
    data = gen_image_data(torch_format)
    v = viBaseTrainer()
    v.set_data(data)
    assert_(isinstance(v.train_iterator, torch.utils.data.DataLoader))


def test_reparametrize():
    input_dim = (28, 28)
    data = gen_image_data()
    v = viBaseTrainer()
    v.set_encoder(fcEncoderNet(input_dim, 2))
    v.set_data(data)
    z_mu, z_sd = v.encoder_net(data)
    z = v.reparameterize(z_mu, z_sd)
    assert_(np.any(np.not_equal(z.detach().cpu().numpy(), z_mu.detach().cpu().numpy())))
    assert_equal(z.detach().cpu().numpy().shape, (100, 2))


def test_custom_optimizer():
    in_dim = (28, 28)
    data = gen_image_data()
    custom_optimizer1 = lambda x: torch.optim.Adam(x, lr=1e-2)
    custom_optimizer2 = lambda x: torch.optim.Adam(x, lr=1e-6)
    vae = simple_vae(in_dim)
    vae.set_encoder(fcEncoderNet(in_dim, 2))
    vae.set_decoder(fcDecoderNet(in_dim, 2))
    vae.compile_trainer((data,), optimizer=custom_optimizer1)
    for _ in range(4):
        vae.loss_history["train_loss"].append(vae.train_epoch())
    delta1 = abs(vae.loss_history["train_loss"][-1]
                 - vae.loss_history["train_loss"][0])
    vae._reset_weights()
    vae._reset_training_history()
    vae._delete_optimizer()
    vae.compile_trainer((data,), optimizer=custom_optimizer2)
    for _ in range(4):
        vae.loss_history["train_loss"].append(vae.train_epoch())
    delta2 = abs(vae.loss_history["train_loss"][-1]
                 - vae.loss_history["train_loss"][0])
    assert_(delta1 > 2 * delta2)
