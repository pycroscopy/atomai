import os
from typing import Dict, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from atomai.core import models
from atomai.utils import (crop_borders, extract_subimages, get_coord_grid,
                          subimg_trajectories, transform_coordinates)
from scipy.stats import norm
from sklearn.model_selection import train_test_split


class EncoderNet(nn.Module):
    """
    Encoder (inference) network

    Args:
        dim: tuple with image dimensions: (height, width) or (height, width, channels)
        latent_dim: number of latent dimensions (the first three latent dimensions are angle & translations by default)
        num_layers: number of NN layers
        hidden_dim: number of neurons in each fully connnected layer (when mlp=True)
            or number of filters in each convolutional layer (when mlp=False, default)
        mlp: using a simple multi-layer perceptron instead of convolutional layers (Default: False)

    """
    def __init__(self,
                 dim: Tuple[int],
                 latent_dim: int = 5,
                 num_layers: int = 2,
                 hidden_dim: int = 32,
                 mlp: bool = False) -> None:
        """
        Initializes network parameters
        """
        super(EncoderNet, self).__init__()
        c = 1 if len(dim) == 2 else dim[-1]
        self.mlp = mlp
        if not self.mlp:
            conv2dblock = models.conv2dblock
            self.econv = conv2dblock(num_layers, c, hidden_dim, lrelu_a=0.1)
            self.reshape_ = hidden_dim * dim[0] * dim[1]
        else:
            edense = []
            for i in range(num_layers):
                input_dim = np.product(dim) if i == 0 else hidden_dim
                edense.extend([nn.Linear(input_dim, hidden_dim), nn.Tanh()])
            self.edense = nn.Sequential(*edense)
            self.reshape_ = hidden_dim
        self.fc11 = nn.Linear(self.reshape_, latent_dim)
        self.fc12 = nn.Linear(self.reshape_, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass
        """
        if self.mlp:
            x = x.reshape(-1, np.product(x.size()[1:]))
            x = self.edense(x)
        else:
            x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, -1, 1, 2)
            x = self.econv(x)
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
        out_dim: tuple with output dimensions: (height, width) or (height, width, channels)
    """
    def __init__(self,
                 latent_dim: int,
                 num_layers: int,
                 hidden_dim: int,
                 out_dim: Tuple[int]) -> None:
        """
        Initializes network parameters
        """
        super(rDecoderNet, self).__init__()
        if len(out_dim) == 2:
            c = 1
            self.reshape_ = (out_dim[0], out_dim[1])
            self.apply_softplus = True
        else:
            c = out_dim[-1]
            self.reshape_ = (out_dim[0], out_dim[1], c)
            self.apply_softplus = False
        self.fc_coord = nn.Linear(2, hidden_dim)
        self.fc_latent = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.activation = nn.Tanh()
        fc_decoder = []
        for i in range(num_layers):
            fc_decoder.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        self.fc_decoder = nn.Sequential(*fc_decoder)
        self.out = nn.Linear(hidden_dim, c)

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
        if self.apply_softplus:
            return F.softplus(out)
        return out


class DecoderNet(nn.Module):
    """
    Decoder network

    Args:
        latent_dim: number of latent dimensions associated with images content
        num_layers: number of fully connected layers
        hidden_dim: number of neurons in each fully connected layer
        out_dim: tuple with image dimensions: (height, width) or (height, width, channels)
        mlp: using a simple multi-layer perceptron instead of convolutional layers (Default: False)
    """
    def __init__(self,
                 latent_dim: int,
                 num_layers: int,
                 hidden_dim: int,
                 out_dim: Tuple[int],
                 mlp: bool = False,) -> None:
        """
        Initializes network parameters
        """
        super(DecoderNet, self).__init__()
        c = 1 if len(out_dim) == 2 else out_dim[-1]
        self.mlp = mlp
        if not self.mlp:
            self.fc_linear = nn.Linear(
                latent_dim, hidden_dim * out_dim[0] * out_dim[1], bias=False)
            self.reshape_ = (hidden_dim, out_dim[0], out_dim[1])
            self.decoder = models.conv2dblock(
                num_layers, hidden_dim, hidden_dim, lrelu_a=0.1)
            self.out = nn.Conv2d(hidden_dim, c, 1, 1, 0)
        else:
            decoder = []
            for i in range(num_layers):
                hidden_dim_ = latent_dim if i == 0 else hidden_dim
                decoder.extend([nn.Linear(hidden_dim_, hidden_dim), nn.Tanh()])
            self.decoder = nn.Sequential(*decoder)
            self.out = nn.Linear(hidden_dim, np.product(out_dim))
        self.out_dim = (c, out_dim[0], out_dim[1])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        if not self.mlp:
            z = self.fc_linear(z)
            z = z.reshape(-1, *self.reshape_)
        h = self.decoder(z)
        h = self.out(h)
        h = h.reshape(-1, *self.out_dim)
        if h.size(1) == 1:
            h = h.squeeze(1)
        else:
            h = h.permute(0, 2, 3, 1)
        return h


class EncoderDecoder:
    """
    General class for Encoder-Decoder type of deep latent variable models

    Args:
        im_dim: xy planar dimensions (height, width) of input images
        latent_dim: latent dimension in deep latent variable model
        seed: seed for torch and numpy (pseudo-)random numbers generators
        **conv_encoder: use convolutional layers in encoder
        **conv_decoder: use convolutional layers in encoder (doesn't apply to spatial decoder)
        **numlayers_encoder: number of layers in encoder (Default: 2)
        **numlayers_decoder: number of layers in decoder (Default: 2)
        **numhidden_encoder: number of hidden units OR conv filters in encoder (Default: 128)
        **numhidden_decoder: number of hidden units OR conv filters in decoder (Default: 128)
    """
    def __init__(self,
                 im_dim: Tuple[int],
                 latent_dim: int,
                 coord: bool = True,
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
        self.z_dim = latent_dim
        if coord:
            self.z_dim = latent_dim + 3
        mlp_e = not kwargs.get("conv_encoder", False)
        if not coord:
            mlp_d = not kwargs.get("conv_decoder", False)
        numlayers_e = kwargs.get("numlayers_encoder", 2)
        numlayers_d = kwargs.get("numlayers_decoder", 2)
        numhidden_e = kwargs.get("numhidden_encoder", 128)
        numhidden_d = kwargs.get("numhidden_decoder", 128)

        if not coord:
            self.decoder_net = DecoderNet(
                latent_dim, numlayers_d, numhidden_d, self.im_dim, mlp_d)
        else:
            self.decoder_net = rDecoderNet(
                latent_dim, numlayers_d, numhidden_d, self.im_dim)
        self.encoder_net = EncoderNet(
            self.im_dim, self.z_dim, numlayers_e, numhidden_e, mlp_e)

        self.coord = coord

    def load_weights(self, filepath: str) -> None:
        """
        Loads saved weights
        """
        weights = torch.load(filepath)
        encoder_weights = weights["encoder"]
        decoder_weights = weights["decoder"]
        self.encoder_net.load_state_dict(encoder_weights)
        self.encoder_net.eval()
        self.decoder_net.load_state_dict(decoder_weights)
        self.decoder_net.eval()

    def save_weights(self, *args: List[str]) -> None:
        """
        Saves trained weights
        """
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

        Args:
            x_test: image array to encode
            **num_batches: number of batches (Default: 10)
        
        Returns:
            Mean and SD of the encoded distribution
        """
        def inference() -> np.ndarray:
            with torch.no_grad():
                z_mean, z_sd = self.encoder_net(x_i)
            return z_mean.cpu().numpy(), torch.exp(z_sd.cpu()).numpy()

        if isinstance(x_test, np.ndarray):
            x_test = torch.from_numpy(x_test).float()
        if (x_test.ndim == len(self.im_dim) == 2 or
           x_test.ndim == len(self.im_dim) == 3):
            x_test = x_test.unsqueeze(0)
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
        
        Returns:
            Generated ("decoded") image(s)
        """

        if isinstance(z_sample, np.ndarray):
            z_sample = torch.from_numpy(z_sample).float()
        if len(z_sample.size()) == 1:
            z_sample = z_sample[None, ...]
        if self.coord:
            xx = torch.linspace(-1, 1, self.im_dim[0])
            yy = torch.linspace(1, -1, self.im_dim[1])
            x0, x1 = torch.meshgrid(xx, yy)
            x_coord = torch.stack([x0.T.flatten(), x1.T.flatten()], axis=1)
            x_coord = x_coord.expand(z_sample.size(0), *x_coord.size())
            if torch.cuda.is_available():
                x_coord = x_coord.cuda()
        if torch.cuda.is_available():
            z_sample = z_sample.cuda()
            self.decoder_net.cuda()
        self.decoder_net.eval()
        with torch.no_grad():
            if self.coord:
                x_decoded = self.decoder_net(x_coord, z_sample)
            else:
                x_decoded = self.decoder_net(z_sample)
        imdec = x_decoded.cpu().numpy()
        return imdec

    def forward_(self,
                 x_new: Union[np.ndarray, torch.Tensor],
                 **kwargs: Dict) -> np.ndarray:
        """
        Forward prediction with uncertainty quantification by sampling from
        the encoded mean and std. Works only for regular VAE (and not for rVAE)

        Args:
            x_new: image array to encode
            **num_samples: number of samples to generate from normal distribution
        
        Returns:
            Ensemble of "decoded" images
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
            decoded_all.append(self.decode_(z_sample))
        decoded_all = np.concatenate(decoded_all, axis=0)
        return decoded_all

    def encode_images(self,
                      imgdata: np.ndarray,
                      **kwargs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encodes every pixel of every image in image stack

        Args:
            imgdata: 3D numpy array. Can also be a single 2D image
            **num_batches: number of batches for for encoding pixels of a single image

        Returns:
            Cropped original image stack and encoded array (cropping is due to finite window size)
        """

        if (imgdata.ndim == len(self.im_dim) == 2 or
           imgdata.ndim == len(self.im_dim) == 3):
            imgdata = np.expand_dims(imgdata, axis=0)
        imgdata_encoded, imgdata_ = [], []
        for i, img in enumerate(imgdata):
            print("\rImage {}/{}".format(i+1, imgdata.shape[0]), end="")
            img_, img_encoded = self.encode_image_(img, **kwargs)
            imgdata_encoded.append(img_encoded)
            imgdata_.append(img_)
        return np.array(imgdata_), np.array(imgdata_encoded)

    def encode_image_(self,
                      img: np.ndarray,
                      **kwargs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crops and encodes a subimage around each pixel in the input image.
        The size of subimage is determined by size of images in VAE training data.

        Args:
            img: 2D numpy array
            **num_batches: number of batches for encoding subimages

        Returns:
            Cropped original image and encoded array (cropping is due to finite window size)
        """

        num_batches = kwargs.get("num_batches", 10)
        inf = np.int(1e5)
        img_to_encode = img.copy()
        coordinates = get_coord_grid(img_to_encode, 1, return_dict=False)
        batch_size = coordinates.shape[0] // num_batches
        encoded_img = -inf * np.ones((*img_to_encode.shape, self.z_dim))
        for i in range(num_batches):
            coord_i = coordinates[i*batch_size:(i+1)*batch_size]
            subimgs_i, com_i, _ = extract_subimages(
                                    img_to_encode, coord_i, self.im_dim[0])
            if len(subimgs_i) > 0:
                z_mean, _ = self.encode(subimgs_i, num_batches=10)
                for k, (l, m) in enumerate(com_i):
                    encoded_img[int(l), int(m)] = z_mean[k]
        coord_i = coordinates[(i+1)*batch_size:]
        if len(coord_i) > 0:
            subimgs_i, com_i, _ = extract_subimages(
                                    img_to_encode, coord_i, self.im_dim[0])
            if len(subimgs_i) > 0:
                z_mean, _ = self.encode(subimgs_i, num_batches=10)
                for k, (l, m) in enumerate(com_i):
                    encoded_img[int(l), int(m)] = z_mean[k]

        img_to_encode[encoded_img[..., 0] == -inf] = 0
        img_to_encode = crop_borders(img_to_encode[..., None], 0)
        encoded_img = crop_borders(encoded_img, -inf)

        return img_to_encode[..., 0], encoded_img

    def encode_trajectories(self,
                            imgdata: np.ndarray,
                            coord_class_dict: Dict[int, np.ndarray],
                            window_size: int,
                            min_length: int,
                            rmax: int,
                            **kwargs: Dict
                            ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Calculates trajectories and latent variable value
        for each point in a trajectory.

        Args:
            imgdata: NN output (preferable) or raw data
            coord_class_dict: atomic/defect/particle coordinates
            window_size: size of subimages to crop
            min_length: minimum length of trajectory to be included
            rmax: maximum allowed distance (projected on xy plane) between defect
                in one frame and the position of its nearest neigbor in the next one
            **num_batches: number of batches for self.encode (Default: 10)
        
        Returns:
            List of encoded trajectories and corresponding movie frame numbers
        """
        t = subimg_trajectories(
                imgdata, coord_class_dict, window_size, min_length, rmax)
        trajectories, frames, subimgs_all = t.get_all_trajectories()
        trajectories_enc_all = []
        for traj, subimgs in zip(trajectories, subimgs_all):
            z_mean, _ = self.encode(
                subimgs, num_batches=kwargs.get("num_batches", 10))
            traj_enc = np.concatenate((traj[:, :2], z_mean), axis=-1)
            trajectories_enc_all.append(traj_enc)
        return trajectories_enc_all, frames

    def manifold2d(self, **kwargs: Dict) -> None:
        """
        Performs mapping from latent space to data space allowing the learned
        manifold to be visualized. This works only for 2d latent variable
        (not counting angle & translation dimensions)

        Args:
            **d: grid size
            **cmap: color map (Default: gnuplot)
            **draw_grid: plot semi-transparent grid
        """
        d = kwargs.get("d", 9)
        cmap = kwargs.get("cmap", "gnuplot")
        if len(self.im_dim) == 2:
            figure = np.zeros((self.im_dim[0] * d, self.im_dim[1] * d))
        elif len(self.im_dim) == 3:
            figure = np.zeros((self.im_dim[0] * d, self.im_dim[1] * d, self.im_dim[-1]))
        grid_x = norm.ppf(np.linspace(0.05, 0.95, d))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, d))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                imdec = self.decode(z_sample)
                figure[i * self.im_dim[0]: (i + 1) * self.im_dim[0],
                       j * self.im_dim[1]: (j + 1) * self.im_dim[1]] = imdec
        if figure.min() < 0:
            figure = (figure - figure.min()) / figure.ptp()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(figure, cmap=cmap, origin="lower")
        draw_grid = kwargs.get("draw_grid")
        if draw_grid:
            major_ticks_x = np.arange(0, d * self.im_dim[0], self.im_dim[0])
            major_ticks_y = np.arange(0, d * self.im_dim[1], self.im_dim[1])
            ax.set_xticks(major_ticks_x)
            ax.set_yticks(major_ticks_y)
            ax.grid(which='major', alpha=0.6)
        if not kwargs.get("savefig"):
            plt.show()
        else:
            savedir = kwargs.get("savedir", './vae_learning/')
            fname = kwargs.get("filename", "manifold_2d")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            fig.savefig(os.path.join(savedir, '{}.png'.format(fname)))
            plt.close(fig)


class rVAE(EncoderDecoder):
    """
    Implements rotationally and translationally invariant
    Variational Autoencoder (VAE), which is based on the work
    by Bepler et al. in arXiv:1909.11663

    Args:
        imstack: 3D or 4D stack of training images ( n x w x h or n x w x h x c )
        latent_dim: number of VAE latent dimensions associated with image content
        training_cycles: number of training 'epochs' (Default: 300)
        minibatch_size: size of training batch for each training epoch (Default: 200)
        test_size: proportion of the dataset for model evaluation (Default: 0.15)
        seed: seed for torch and numpy (pseudo-)random numbers generators
        **conv_encoder: use convolutional layers in encoder
        **numlayers_encoder: number of layers in encoder (Default: 2)
        **numlayers_decoder: number of layers in decoder (Default: 2)
        **numhidden_encoder: number of hidden units OR conv filters in encoder (Default: 128)
        **numhidden_decoder: number of hidden units in decoder (Default: 128)
        **loss: reconstruction loss function, "ce" or "mse" (Default: "mse")
        **translation_prior: translation prior
        **rotation_prior: rotational prior
        **recording: saves a learned 2d manifold at each training step
    """
    def __init__(self,
                 imstack: np.ndarray,
                 latent_dim: int = 2,
                 training_cycles: int = 300,
                 minibatch_size: int = 200,
                 test_size: float = 0.15,
                 seed: int = 0,
                 **kwargs: Dict) -> None:
        dim = imstack.shape[1:]
        coord = True
        super(rVAE, self).__init__(dim, latent_dim, coord, seed, **kwargs)

        if torch.cuda.is_available:
            torch.cuda.empty_cache()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        self.im_dim = imstack.shape[1:]
        imstack_train, imstack_test = train_test_split(
            imstack, test_size=test_size, shuffle=True, random_state=seed)

        X_train = torch.from_numpy(imstack_train).float()
        X_test = torch.from_numpy(imstack_test).float()

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
        self.loss = kwargs.get("loss", "mse")

        self.dx_prior = kwargs.get("translation_prior", 0.1)
        self.phi_prior = kwargs.get("rotation_prior", 0.1)

        self.training_cycles = training_cycles

        self.recording = kwargs.get("recording", False)

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
        if self.loss == "mse":
            reconstr_error = -0.5 * torch.sum(
                (x_reconstr.view(batch_dim, -1) - x.view(batch_dim, -1))**2, 1).mean()
        else:
            px_size = np.product(self.im_dim)
            rs = (self.im_dim[0] * self.im_dim[1], self.im_dim[-1])
            reconstr_error = -F.binary_cross_entropy_with_logits(
                x_reconstr.view(-1, *rs), x.view(-1, *rs)) * px_size
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
        Evaluates model on test data
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
            if self.recording and self.z_dim == 5:
                self.manifold2d(savefig=True, filename=str(e))
        if self.recording and self.z_dim == 5:
            self.visualize_manifold_learning("./vae_learning")
        return

    @classmethod
    def visualize_manifold_learning(cls,
                                    frames_dir: str,
                                    **kwargs: Dict) -> None:
        """
        Creates and stores a video showing evolution of
        learned 2D manifold during rVAE's training

        Args:
            frames_dir: directory with snapshots of manifold as .png files
                        (the files should be named as "1.png", "2.png", etc.)
            **moviename: name of the movie
            **frame_duration: duration of each movie frame
        """
        from atomai.utils import animation_from_png
        movie_name = kwargs.get("moviename", "manifold_learning")
        duration = kwargs.get("frame_duration", 1)
        animation_from_png(frames_dir, movie_name, duration, remove_dir=False)


class VAE(EncoderDecoder):
    """
    Implements a standard Variational Autoencoder (VAE)

    Args:
        imstack: 3D or 4D stack of training images ( n x w x h or n x w x h x c )
        latent_dim: number of VAE latent dimensions associated with image content
        training_cycles: number of training 'epochs' (Default: 300)
        minibatch_size: size of training batch for each training epoch (Default: 200)
        test_size: proportion of the dataset for model evaluation (Default: 0.15)
        seed: seed for torch and numpy (pseudo-)random numbers generators
        **conv_encoder: use convolutional layers in encoder
        **conv_decoder: use convolutional layers in decoder
        **numlayers_encoder: number of layers in encoder (Default: 2)
        **numlayers_decoder: number of layers in decoder (Default: 2)
        **numhidden_encoder: number of hidden units OR conv filters in encoder (Default: 128)
        **numhidden_decoder: number of hidden units OR conv filters in decoder (Default: 128)
    """
    def __init__(self,
                 imstack: np.ndarray,
                 latent_dim: int = 2,
                 training_cycles: int = 300,
                 minibatch_size: int = 200,
                 test_size: float = 0.15,
                 seed: int = 0,
                 **kwargs: Dict) -> None:
        dim = imstack.shape[1:]
        coord = False
        super(VAE, self).__init__(dim, latent_dim, coord, seed, **kwargs)

        if torch.cuda.is_available:
            torch.cuda.empty_cache()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        self.im_dim = imstack.shape[1:]
        imstack_train, imstack_test = train_test_split(
            imstack, test_size=test_size, shuffle=True, random_state=seed)

        X_train = torch.from_numpy(imstack_train).float()
        X_test = torch.from_numpy(imstack_test).float()

        if torch.cuda.is_available():
            X_train = X_train.cuda()
            X_test = X_test.cuda()

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

        self.training_cycles = training_cycles

    def step(self,
             x: torch.Tensor,
             mode: str = "train") -> torch.Tensor:
        """
        Single training/test step
        """
        batch_dim = x.size(0)
        if torch.cuda.is_available():
            x = x.cuda()
        if mode == "eval":
            with torch.no_grad():
                z_mean, z_logsd = self.encoder_net(x)
        else:
            z_mean, z_logsd = self.encoder_net(x)
        z_sd = torch.exp(z_logsd)
        z_dim = z_mean.size(1)
        eps = x.new(batch_dim, z_dim).normal_()
        z = z_mean + z_sd * eps

        if mode == "eval":
            with torch.no_grad():
                x_reconstr = self.decoder_net(z)
        else:
            x_reconstr = self.decoder_net(z)
        if self.loss == "mse":
            reconstr_error = -0.5 * torch.sum(
                (x_reconstr.reshape(batch_dim, -1) - x.reshape(batch_dim, -1))**2, 1).mean()
        else:
            px_size = np.product(self.im_dim)
            rs = (self.im_dim[0] * self.im_dim[1], self.im_dim[-1])
            reconstr_error = -F.binary_cross_entropy_with_logits(
                x_reconstr.reshape(-1, *rs), x.reshape(-1, *rs)) * px_size
        kl_z = -z_logsd + 0.5 * z_sd**2 + 0.5 * z_mean**2 - 0.5
        kl_z = torch.sum(kl_z, 1).mean()
        return reconstr_error - kl_z

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
        Evaluates model on test data
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
        Trains VAE model
        """
        for e in range(self.training_cycles):
            elbo_epoch = self.train_epoch()
            elbo_epoch_test = self.evaluate_model()
            template = 'Epoch: {}/{}, Training loss: {:.4f}, Test loss: {:.4f}'
            print(template.format(e+1, self.training_cycles,
                  -elbo_epoch, -elbo_epoch_test))
        return
