from typing import Dict, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from atomai.core import models
from atomai.utils import transform_coordinates


class EncoderNet(nn.Module):
    """
    Encoder (inference) network

    Args:
        dim: tuple with image height and width
        latent_dim: number of latent dimensions (the first three latent dimensions are angle & translations by default)
        num_layers: number of NN layers
        hidden_dim: number of neurons in each fully connnected layer (when mlp=True)
            or number of filters in each convolutional layer (when mlp=False, default)
        mlp: using a simple multi-layer perceptron instead of convolutional layers (Default: False)

    """
    def __init__(self,
                 dim: Tuple[int, int],
                 latent_dim: int = 5,
                 num_layers: int = 2,
                 hidden_dim: int = 32,
                 mlp: bool = False) -> None:
        """
        Initializes network parameters
        """
        super(EncoderNet, self).__init__()
        n, m = dim
        self.mlp = mlp
        if not self.mlp:
            conv2dblock = models.conv2dblock
            self.econv = conv2dblock(num_layers, 1, hidden_dim, lrelu_a=0.1)
            self.reshape_ = hidden_dim * n * m
        else:
            edense = []
            for i in range(num_layers):
                input_dim = n * m if i == 0 else hidden_dim
                edense.extend([nn.Linear(input_dim, hidden_dim), nn.Tanh()])
            self.edense = nn.Sequential(*edense)
            self.reshape_ = hidden_dim
        self.fc11 = nn.Linear(self.reshape_, latent_dim)
        self.fc12 = nn.Linear(self.reshape_, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        """
        if self.mlp:
            x = x.reshape(-1, x.size(1) * x.size(2))
            x = self.edense(x)
        else:
            x = self.econv(x[:, None, ...])
            x = x.reshape(-1, self.reshape_)
        z_mu = self.fc11(x)
        z_logstd = self.fc12(x)
        return z_mu, z_logstd


class rDecoderNet(nn.Module):
    """
    Spatial decoder network
    (based on https://arxiv.org/abs/1909.11663)

    Args:

    latent_dim: number of latent dimensions associated with images content
    num_layers: number of fully connected layers
    hidden_dim: number of neurons in each fully connected layer
    out_dim: output image dimensions (height and width)
    """
    def __init__(self,
                 latent_dim: int,
                 num_layers: int,
                 hidden_dim: int,
                 out_dim: Tuple) -> None:
        """
        Initializes network parameters
        """
        super(rDecoderNet, self).__init__()
        self.reshape_ = out_dim
        self.fc_coord = nn.Linear(2, hidden_dim)
        self.fc_latent = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.activation = nn.Tanh()
        fc_decoder = []
        for i in range(num_layers):
            fc_decoder.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        self.fc_decoder = nn.Sequential(*fc_decoder)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x_coord: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        batch_dim, n = x_coord.size()[:2]
        x_coord = x_coord.reshape(batch_dim * n, -1)
        h_x = self.fc_coord(x_coord)
        h_x = h_x.reshape(batch_dim, n, -1)
        h_z = self.fc_latent(z)
        h = h_x.add(h_z.unsqueeze(1))
        h = h.reshape(batch_dim * n, -1)
        h = self.activation(h)
        h = self.fc_decoder(h)
        h = self.out(h)
        out = h.reshape(batch_dim, *self.reshape_)
        return F.softplus(out)


class EncoderDecoder:
    """
    General class for Encoder-Decoder type of deep latent variable models
    indepentent of rotations and translations

    Args:
        im_dim: xy planar dimensions (height, width) of input images
        latent_dim: latent dimension in deep latent variable model
        conv_encoder: use convolutional layers in encoder
        seed: seed for torch and numpy (pseudo-)random numbers generators
        **numlayers_encoder: number of layers in encoder (Default: 2)
        **numlayers_decoder: number of layers in decoder (Default: 2)
        **numhidden_encoder: number of hidden units OR conv filters in encoder (Default: 128)
        **numhidden_decoder: number of hidden units OR conv filters in decoder (Default: 128)
    """
    def __init__(self,
                 im_dim: Tuple[int, int],
                 latent_dim: int,
                 conv_encoder: bool = False,
                 seed: int = 0,
                 **kwargs: Dict) -> None:

        if torch.cuda.is_available:
            torch.cuda.empty_cache()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        self.im_dim = im_dim
        self.z_dim = latent_dim + 3
        mlp = not conv_encoder
        numlayers_e = kwargs.get("numlayers_encoder", 2)
        numlayers_d = kwargs.get("numlayers_decoder", 2)
        numhidden_e = kwargs.get("numhidden_encoder", 128)
        numhidden_d = kwargs.get("numhidden_encoder", 128)

        self.decoder_net = rDecoderNet(
            latent_dim, numlayers_d, numhidden_d, self.im_dim)
        self.encoder_net = EncoderNet(
            self.im_dim, self.z_dim, numlayers_e, numhidden_e, mlp)

    def load_weights(self, filepath):
        weights = torch.load(filepath)
        encoder_weights = weights["encoder"]
        decoder_weights = weights["decoder"]
        self.encoder_net.load_state_dict(encoder_weights)
        self.encoder_net.eval()
        self.decoder_net.load_state_dict(decoder_weights)
        self.decoder_net.eval()

    def save_weights(self, *args):
        try:
            savepath = args[0]
        except IndexError:
            savepath = "./rVAE"
        torch.save({"encoder": self.encoder_net.state_dict(),
                    "decoder": self.decoder_net.state_dict()},
                     savepath + ".tar")

    def encode(self,
               x_test: Union[np.ndarray, torch.Tensor],
               **kwargs: Dict) -> Tuple[np.ndarray]:
        """
        Encodes input image data using a trained VAE's encoder
        """
        def inference() -> np.ndarray:
            with torch.no_grad():
                z_mean, z_sd = self.encoder_net(x_i)
            return z_mean.cpu().numpy(), z_sd.cpu().numpy()

        if isinstance(x_test, np.ndarray):
            x_test = torch.from_numpy(x_test).float()
        x_test = x_test.unsqueeze(0) if x_test.ndim == 2 else x_test
        if torch.cuda.is_available():
            x_test = x_test.cuda()
            self.encoder_net.cuda()
        num_batches = kwargs.get("num_batches", 10)
        batch_size = len(x_test) // num_batches
        z_mean_all = np.zeros((x_test.shape[0], self.z_dim))
        z_sd_all = np.zeros((x_test.shape[0], self.z_dim))

        for i in range(num_batches):
            x_i = x_test[i*batch_size:(i+1)*batch_size]
            z_mean_i, z_sd_i = inference()
            z_mean_all[i*batch_size:(i+1)*batch_size] = z_mean_i
            z_sd_all[i*batch_size:(i+1)*batch_size] = z_sd_i
        x_i = x_test[(i+1)*batch_size:]
        if len(x_i) > 0:
            z_mean_i, z_sd_i = inference()
            z_mean_all[(i+1)*batch_size:] = z_mean_i
            z_sd_all[(i+1)*batch_size:] = z_sd_i

        return z_mean_all, z_sd_all

    def decode(self, z_sample: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Takes a point in latent space and and maps it to data space
        via the learned generative model

        Args:
            z_sample: point in latent space
        """

        n, m = self.im_dim
        if isinstance(z_sample, np.ndarray):
            z_sample = torch.from_numpy(z_sample).float()
        if len(z_sample.size()) == 1:
            z_sample = z_sample[None, ...]
        xx = torch.linspace(-1, 1, n)
        yy = torch.linspace(1, -1, m)
        x0, x1 = torch.meshgrid(xx, yy)
        x_coord = torch.stack([x0.T.flatten(), x1.T.flatten()], axis=1)
        x_coord = x_coord.expand(z_sample.size(0), *x_coord.size())
        if torch.cuda.is_available():
            z_sample = z_sample.cuda()
            x_coord = x_coord.cuda()
            self.decoder_net.cuda()
        self.decoder_net.eval()
        with torch.no_grad():
            x_decoded = self.decoder_net(x_coord, z_sample)
        imdec = x_decoded.cpu().numpy()
        return imdec

    def forward_(self,
                 x_new: Union[np.ndarray, torch.Tensor],
                 **kwargs: Dict) -> np.ndarray:
        """
        Forward prediction with uncertainty quantification by sampling from
        the encoded mean and std. Works ony for regular VAE
        (without rotational / translational invariance)
        """
        num_samples = kwargs.get("num_samples", 32)
        if isinstance(x_new, np.ndarray):
            x_new = torch.from_numpy(x_new).float()
        if torch.cuda.is_available():
            x_new = x_new.cuda()
            self.encoder_net.cuda()
        with torch.no_grad():
            z_mean, z_logsd = self.encoder_net(x_new)
        z_sd = torch.exp(z_logsd)
        ndist = torch.distributions.Normal(z_mean, z_sd)
        decoded_all = []
        for i in range(num_samples):
            z_sample = ndist.rsample()
            z_sample = z_sample.view(1, -1)
            decoded_all.append(self.decode_(z_sample))  # Works only for regular VAE
        decoded_all = np.concatenate(decoded_all, axis=0)
        return decoded_all

    def manifold2d(self, **kwargs: Dict) -> None:
        """
        Performs mapping from latent space to data space allowing the learned
        manifold to be visualized. This works only for 2d latent variable
        (not counting angle & translation dimensions)
        """
        n, m = self.im_dim
        d = kwargs.get("d", 9)
        cmap = kwargs.get("cmap", "gnuplot")
        figure = np.zeros((n * d, m * d))
        grid_x = norm.ppf(np.linspace(0.05, 0.95, d))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, d))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                imdec = self.decode(z_sample)
                figure[i * n: (i + 1) * n, j * m: (j + 1) * m] = imdec

        _, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(figure, cmap=cmap)
        major_ticks_x = np.arange(0, d * n, n)
        major_ticks_y = np.arange(0, d * m, m)
        ax.set_xticks(major_ticks_x)
        ax.set_yticks(major_ticks_y)
        ax.grid(which='major', alpha=0.6)
        plt.show()


class rVAE(EncoderDecoder):
    """
    Implements rotationally and translationally invariant
    Variational Autoencoder (VAE), which is based on the work
    by Bepler et al. in arXiv:1909.11663

    Args:
        imstack: 3D stack of training images (n x w x h)
        latent_dim: number of VAE latent dimensions associated with image content
        training_cycles: number of training 'epochs' (Default: 300)
        minibatch_size: size of training batch for each training epoch (Default: 200)
        test_size: proportion of the dataset for model evaluation (Default: 0.15)
        conv_encoder: use convolutional layers in encoder
        seed: seed for torch and numpy (pseudo-)random numbers generators
        **numlayers_encoder: number of layers in encoder (Default: 2)
        **numlayers_decoder: number of layers in decoder (Default: 2)
        **numhidden_encoder: number of hidden units OR conv filters in encoder (Default: 128)
        **numhidden_decoder: number of hidden units OR conv filters in decoder (Default: 128)
        **translation_prior: translation prior
        **rotation_prior: rotational prior
    """
    def __init__(self,
                 imstack: np.ndarray,
                 latent_dim: int = 2,
                 training_cycles: int = 300,
                 minibatch_size: int = 200,
                 test_size: float = 0.15,
                 conv_encoder: bool = False,
                 seed: int = 0,
                 **kwargs: Dict) -> None:
        dim = imstack.shape[1:3]
        super(rVAE, self).__init__(dim, latent_dim, conv_encoder, seed, **kwargs)

        if torch.cuda.is_available:
            torch.cuda.empty_cache()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        imstack_train, imstack_test = train_test_split(
            imstack, test_size=test_size, shuffle=True, random_state=seed)

        X_train = torch.from_numpy(imstack_train).float()
        X_test = torch.from_numpy(imstack_test).float()

        self.im_dim = imstack.shape[1:3]
        xx = torch.linspace(-1, 1, self.im_dim[0])
        yy = torch.linspace(1, -1, self.im_dim[1])
        x0, x1 = torch.meshgrid(xx, yy)
        self.x_coord = torch.stack(
            [x0.T.flatten(), x1.T.flatten()], axis=1)

        if torch.cuda.is_available():
            X_train = X_train.cuda()
            X_test = X_test.cuda()
            self.x_coord = self.x_coord.cuda()

        data_train = torch.utils.data.TensorDataset(X_train)
        data_test = torch.utils.data.TensorDataset(X_test)
        self.train_iterator = torch.utils.data.DataLoader(
            data_train, batch_size=minibatch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(
            data_test, batch_size=minibatch_size)

        if torch.cuda.is_available():
            self.decoder_net.cuda()
            self.encoder_net.cuda()

        params = list(self.decoder_net.parameters()) +\
            list(self.encoder_net.parameters())
        self.optim = torch.optim.Adam(params, lr=1e-4)

        self.dx_prior = kwargs.get("translation_prior", 0.1)
        self.phi_prior = kwargs.get("rotation_prior", 0.1)

        self.training_cycles = training_cycles

    def step(self,
             x: torch.Tensor,
             mode: str = "train") -> torch.Tensor:
        """
        Single training/test step
        """
        batch_dim = x.size(0)
        x_coord_ = self.x_coord.expand(batch_dim, *self.x_coord.size())
        if torch.cuda.is_available():
            x = x.cuda()
        if mode == "eval":
            with torch.no_grad():
                z_mean, z_logsd = self.encoder_net(x)
        else:
            z_mean, z_logsd = self.encoder_net(x)
        z_sd = torch.exp(z_logsd)
        z_dim = z_mean.size(1)
        eps = x_coord_.new(batch_dim, z_dim).normal_()
        z = z_mean + z_sd * eps
        phi = z[:, 0]  # angle
        phi_sd, phi_logsd = z_sd[:, 0], z_logsd[:, 0]
        dx = z[:, 1:3]  # translation
        dx = (dx * self.dx_prior).unsqueeze(1)
        z = z[:, 3:]  # image content
        z_mean, z_sd, z_logsd = z_mean[:, 1:], z_sd[:, 1:], z_logsd[:, 1:]
        x_coord_ = transform_coordinates(x_coord_, phi, dx)
        kl_rot = (-phi_logsd + np.log(self.phi_prior) +
                  phi_sd**2 / (2 * self.phi_prior**2) - 0.5)
        if mode == "eval":
            with torch.no_grad():
                x_reconstr = self.decoder_net(x_coord_, z)
        else:
            x_reconstr = self.decoder_net(x_coord_, z)
        reconstr_error = -0.5 * torch.sum(
            (x_reconstr.view(batch_dim, -1) - x.view(batch_dim, -1))**2, 1).mean()
        kl_z = -z_logsd + 0.5 * z_sd**2 + 0.5 * z_mean**2 - 0.5
        kl_div = (kl_rot + torch.sum(kl_z, 1)).mean()
        return reconstr_error - kl_div

    def train_epoch(self):
        """
        Trains a single epoch
        """
        self.decoder_net.train()
        self.encoder_net.train()
        c = 0
        elbo_epoch = 0
        for x, in self.train_iterator:
            b = x.size(0)
            elbo = self.step(x)
            loss = -elbo
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            elbo = elbo.item()
            c += b
            delta = b * (elbo - elbo_epoch)
            elbo_epoch += delta / c
        return elbo_epoch

    def evaluate_model(self):
        """
        Evaluates model on the test data
        """
        self.decoder_net.eval()
        self.encoder_net.eval()
        c = 0
        elbo_epoch_test = 0
        for x, in self.test_iterator:
            b = x.size(0)
            elbo = self.step(x, mode="eval")
            elbo = elbo.item()
            c += b
            delta = b * (elbo - elbo_epoch_test)
            elbo_epoch_test += delta / c
        return elbo_epoch_test

    def run(self):
        """
        Trains rVAE model
        """
        for e in range(self.training_cycles):
            elbo_epoch = self.train_epoch()
            elbo_epoch_test = self.evaluate_model()
            template = 'Epoch: {}/{}, Training loss: {:.4f}, Test loss: {:.4f}'
            print(template.format(e+1, self.training_cycles,
                  -elbo_epoch, -elbo_epoch_test))
        return
