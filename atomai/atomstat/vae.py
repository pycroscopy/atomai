"""
vae.py
===========

Module for analysis of system "building blocks"" with variational autoencoders

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import os
from typing import Dict, List, Tuple, Type, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from atomai.nets import DecoderNet, EncoderNet, rDecoderNet
from atomai.utils import (crop_borders, extract_subimages, get_coord_grid,
                          imcoordgrid, init_vae_dataloaders, set_train_rng,
                          subimg_trajectories, transform_coordinates)
from scipy.stats import norm
from sklearn.model_selection import train_test_split


class BaseVAE:
    """
    General class for encoder-decoder type of deep latent variable models

    Args:
        im_dim (tuple):
            (height, width) or (height, width, channel) of input images
        latent_dim (int):
            latent dimension in deep latent variable model
        seed (int):
            seed for torch and numpy (pseudo-)random numbers generators
        **conv_encoder (bool):
            use convolutional layers in encoder
        **conv_decoder (bool):
            use convolutional layers in decoder (doesn't apply to  rVAE)
        **numlayers_encoder (int):
            number of layers in encoder (Default: 2)
        **numlayers_decoder (int):
            number of layers in decoder (Default: 2)
        **numhidden_encoder (int):
            number of hidden units OR conv filters in encoder (Default: 128)
        **numhidden_decoder (int):
            number of hidden units OR conv filters in decoder (Default: 128)
    """
    def __init__(self,
                 im_dim: Tuple[int],
                 latent_dim: int,
                 coord: bool = True,
                 seed: int = 0,
                 **kwargs: Union[int, bool]) -> None:
        """
        Initializes encoder-decoder object
        """

        if torch.cuda.is_available:
            torch.cuda.empty_cache()
        set_train_rng(seed)
        np.random.seed(seed)

        self.im_dim = im_dim
        self.z_dim = latent_dim
        if coord:
            self.z_dim = latent_dim + coord
        mlp_e = not kwargs.get("conv_encoder", False)
        if not coord:
            mlp_d = not kwargs.get("conv_decoder", False)
        numlayers_e = kwargs.get("numlayers_encoder", 2)
        numlayers_d = kwargs.get("numlayers_decoder", 2)
        numhidden_e = kwargs.get("numhidden_encoder", 128)
        numhidden_d = kwargs.get("numhidden_decoder", 128)
        self.num_classes = kwargs.get("num_classes")
        skip = kwargs.get("skip", False)
        if not coord:
            self.decoder_net = DecoderNet(
                latent_dim, numlayers_d, numhidden_d,
                self.im_dim, mlp_d, self.num_classes)
        else:
            self.decoder_net = rDecoderNet(
                latent_dim, numlayers_d, numhidden_d, self.im_dim,
                skip)
        self.encoder_net = EncoderNet(
            self.im_dim, self.z_dim, numlayers_e, numhidden_e, mlp_e)

        self.train_iterator = None
        self.test_iterator = None
        self.optim = None

        self.coord = coord

        self.metadict = {
            "im_dim": self.im_dim,
            "latent_dim": latent_dim,
            "coord": coord,
            "conv_encoder": not mlp_e,
            "numlayers_encoder": numlayers_e,
            "numlayers_decoder": numlayers_d,
            "numhidden_encoder": numhidden_e,
            "numhidden_decoder": numhidden_d,
            "skip": skip,
            "num_classes": self.num_classes
        }
        if not coord:
            self.metadict["conv_decoder"] = not mlp_d

    def load_weights(self, filepath: str) -> None:
        """
        Loads saved weights
        """
        device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        weights = torch.load(filepath, map_location=device_)
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
            savepath = "./VAE"
        torch.save({"encoder": self.encoder_net.state_dict(),
                   "decoder": self.decoder_net.state_dict()},
                   savepath + ".tar")

    def save_model(self, *args: List[str]) -> None:
        """
        Saves trained weights and the key model parameters
        """
        try:
            savepath = args[0]
        except IndexError:
            savepath = "./VAE_metadict"
        self.metadict["encoder"] = self.encoder_net.state_dict()
        self.metadict["decoder"] = self.decoder_net.state_dict()
        self.metadict["optimizer"] = self.optim
        torch.save(self.metadict, savepath + ".tar")

    def encode(self,
               x_test: Union[np.ndarray, torch.Tensor],
               **kwargs: int) -> Tuple[np.ndarray]:
        """
        Encodes input image data using a trained VAE's encoder

        Args:
            x_test (numpy array or torch tensor):
                image array to encode
            **num_batches (int):
                number of batches (Default: 10)

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

    def decode(self, z_sample: Union[np.ndarray, torch.Tensor],
               y: Optional[Union[int, np.ndarray, torch.Tensor]] = None
               ) -> np.ndarray:
        """
        Takes a point in latent space and maps it to data space
        via the learned generative model

        Args:
            z_sample (numpy array or torch tensor):
                point(s) in latent space

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
        if y is not None:
            if isinstance(y, int):
                y = torch.tensor(y)
            elif isinstance(y, np.ndarray):
                y = torch.from_numpy(y)
            if y.dim() == 0:
                y = y.unsqueeze(0)
            y = y.cuda() if torch.cuda.is_available() else y
            z_sample = z_sample.cuda() if torch.cuda.is_available() else z_sample
            
            targets = to_onehot(y, self.num_classes)
            z_sample = torch.cat((z_sample, targets), dim=-1)
        if torch.cuda.is_available():
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
                 **kwargs: int) -> np.ndarray:
        """
        Forward prediction with uncertainty quantification by sampling from
        the encoded mean and std. Works only for regular VAE (and not for rVAE)

        Args:
            x_new (numpy array or torch tensor):
                image array to encode
            **num_samples (int):
                number of samples to generate from normal distribution

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
                      **kwargs: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encodes every pixel of every image in image stack

        Args:
            imgdata (numpy array):
                3D numpy array. Can also be a single 2D image
            **num_batches (int):
                number of batches for for encoding pixels of a single image

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
                      **kwargs: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crops and encodes a subimage around each pixel in the input image.
        The size of subimage is determined by size of images in VAE training data.

        Args:
            img (numpy array):
                2D numpy array
            **num_batches (int):
                number of batches for encoding subimages

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
                            **kwargs: int
                            ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Calculates trajectories and latent variable value
        for each point in a trajectory.

        Args:
            imgdata (np.array):
                NN output (preferable) or raw data
            coord_class_dict (dict):
                atomic/defect/particle coordinates
            window_size (int):
                size of subimages to crop
            min_length (int):
                minimum length of trajectory to be included
            rmax (int):
                maximum allowed distance (projected on xy plane) between defect
                in one frame and the position of its nearest neigbor in the next one
            **num_batches (int):
                number of batches for self.encode (Default: 10)

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
        return trajectories_enc_all, frames, subimgs_all

    def manifold2d(self, **kwargs: Union[int, List, str, bool]) -> None:
        """
        Performs mapping from latent space to data space allowing the learned
        manifold to be visualized. This works only for 2d latent variable
        (not counting angle & translation dimensions)

        Args:
            **d (int): grid size
            l1 (list): range of 1st latent varianle
            l2 (list): range of 2nd latent variable
            **cmap (str): color map (Default: gnuplot)
            **draw_grid (bool): plot semi-transparent grid
            **origin (str): plot origin (e.g. 'lower')
        """
        y = kwargs.get("label")
        l1, l2 = kwargs.get("l1"), kwargs.get("l2")
        d = kwargs.get("d", 9)
        cmap = kwargs.get("cmap", "gnuplot")
        if len(self.im_dim) == 2:
            figure = np.zeros((self.im_dim[0] * d, self.im_dim[1] * d))
        elif len(self.im_dim) == 3:
            figure = np.zeros((self.im_dim[0] * d, self.im_dim[1] * d, self.im_dim[-1]))
        if l1 and l2:
            grid_x = np.linspace(l1[0], l1[1], d)
            grid_y = np.linspace(l2[0], l2[1], d)
        else:
            grid_x = norm.ppf(np.linspace(0.05, 0.95, d))
            grid_y = norm.ppf(np.linspace(0.05, 0.95, d))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                if y is not None:
                    imdec = self.decode(z_sample, y)
                else:
                    imdec = self.decode(z_sample)
                figure[i * self.im_dim[0]: (i + 1) * self.im_dim[0],
                       j * self.im_dim[1]: (j + 1) * self.im_dim[1]] = imdec
        if figure.min() < 0:
            figure = (figure - figure.min()) / figure.ptp()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(figure, cmap=cmap, origin=kwargs.get("origin", "lower"))
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

    @classmethod
    def visualize_manifold_learning(cls,
                                    frames_dir: str,
                                    **kwargs: Union[str, int]) -> None:
        """
        Creates and stores a video showing evolution of
        learned 2D manifold during rVAE's training

        Args:
            frames_dir (str):
                directory with snapshots of manifold as .png files
                (the files should be named as "1.png", "2.png", etc.)
            **moviename (str): name of the movie
            **frame_duration (int): duration of each movie frame
        """
        from atomai.utils import animation_from_png
        movie_name = kwargs.get("moviename", "manifold_learning")
        duration = kwargs.get("frame_duration", 1)
        animation_from_png(frames_dir, movie_name, duration, remove_dir=False)

    def step(self, x: torch.Tensor, mode: str = "train") -> None:
        pass

    @classmethod
    def reparameterize(cls, z_mean: torch.Tensor,
                       z_sd: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick
        """
        batch_dim = z_mean.size(0)
        z_dim = z_mean.size(1)
        eps = z_mean.new(batch_dim, z_dim).normal_()
        return z_mean + z_sd * eps

    def train_epoch(self):
        """
        Trains a single epoch
        """
        self.decoder_net.train()
        self.encoder_net.train()
        c = 0
        elbo_epoch = 0
        for x in self.train_iterator:
            if len(x) == 1:
                x = x[0]
                y = None
            else:
                x, y = x
            b = x.size(0)
            elbo = self.step(x) if y is None else self.step(x, y)
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
        for x in self.test_iterator:
            if len(x) == 1:
                x = x[0]
                y = None
            else:
                x, y = x
            b = x.size(0)
            if y is None:
                elbo = self.step(x, mode="eval")
            else:
                elbo = self.step(x, y, mode="eval")
            elbo = elbo.item()
            c += b
            delta = b * (elbo - elbo_epoch_test)
            elbo_epoch_test += delta / c
        return elbo_epoch_test


class rVAE(BaseVAE):
    """
    Implements rotationally and translationally invariant
    Variational Autoencoder (VAE), which is based on the work
    by Bepler et al. in arXiv:1909.11663

    Args:
        X_train (np.ndarray):
            3D or 4D stack of training images ( n x w x h or n x w x h x c )
        latent_dim (int):
            number of VAE latent dimensions associated with image content
        training_cycles (int):
            number of training 'epochs' (Default: 300)
        batch_size (int):
            size of training batch for each training epoch (Default: 200)
        test_size (float):
            proportion of the dataset for model evaluation (Default: 0.15)
        translation (bool):
            account for xy shifts of image content (Default: True)
        seed(int):
            seed for torch and numpy (pseudo-)random numbers generators
        **conv_encoder (bool):
            use convolutional layers in encoder
        **numlayers_encoder (int):
            number of layers in encoder (Default: 2)
        **numlayers_decoder (int):
            number of layers in decoder (Default: 2)
        **numhidden_encoder (int):
            number of hidden units OR conv filters in encoder (Default: 128)
        **numhidden_decoder (int):
            number of hidden units in decoder (Default: 128)
        **skip (bool):
            uses generative skip model with residual paths between
            latents and decoder layers (Default: False)
        **loss (str):
            reconstruction loss function, "ce" or "mse" (Default: "mse")
        **translation_prior (float):
            translation prior
        **rotation_prior (float):
            rotational prior
        **savename (str):
            file name/path for saving model at the end of training
        **recording (bool):
            saves a learned 2d manifold at each training step
    """
    def __init__(self,
                 X_train: np.ndarray,
                 latent_dim: int = 2,
                 training_cycles: int = 300,
                 batch_size: int = 200,
                 test_size: float = 0.15,
                 translation: bool = True,
                 seed: int = 0,
                 **kwargs: Union[int, float, bool, str]) -> None:
        """
        Initialize rVAE trainer
        """
        dim = X_train.shape[1:]
        coord = 3 if translation else 1  # xy translations and/or rotation
        super(rVAE, self).__init__(dim, latent_dim, coord, seed, **kwargs)

        if torch.cuda.is_available:
            torch.cuda.empty_cache()
        set_train_rng(seed)
        np.random.seed(seed)

        self.im_dim = X_train.shape[1:]
        self.x_coord = imcoordgrid(X_train.shape[1:])
        self.translation = translation
        if torch.cuda.is_available():
            self.x_coord = self.x_coord.cuda()

        X_train, X_test = train_test_split(
            X_train, test_size=test_size, shuffle=True, random_state=seed)
        iterators = init_vae_dataloaders(X_train, X_test, batch_size)
        self.train_iterator, self.test_iterator = iterators

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
        self.savename = kwargs.get("savename", "./rVAE_metadict")
        self.recording = kwargs.get("recording", False)

    def loss_fn(self, x: torch.Tensor, x_reconstr: torch.Tensor,
                *args: torch.Tensor, **kwargs: str) -> torch.Tensor:
        """
        Calculates ELBO
        """
        z_mean, z_logsd = args
        loss = kwargs.get("loss")
        if loss is None:
            loss = self.loss
        z_sd = torch.exp(z_logsd)
        phi_sd, phi_logsd = z_sd[:, 0], z_logsd[:, 0]
        z_mean, z_sd, z_logsd = z_mean[:, 1:], z_sd[:, 1:], z_logsd[:, 1:]
        batch_dim = x.size(0)
        if self.loss == "mse":
            reconstr_error = -0.5 * torch.sum(
                (x_reconstr.view(batch_dim, -1) - x.view(batch_dim, -1))**2, 1).mean()
        else:
            px_size = np.product(self.im_dim)
            rs = (self.im_dim[0] * self.im_dim[1],)
            if len(self.im_dim) == 3:
                rs = rs + (self.im_dim[-1],)
            reconstr_error = -F.binary_cross_entropy_with_logits(
                x_reconstr.view(-1, *rs), x.view(-1, *rs)) * px_size
        kl_rot = (-phi_logsd + np.log(self.phi_prior) +
                  phi_sd**2 / (2 * self.phi_prior**2) - 0.5)
        kl_z = -z_logsd + 0.5 * z_sd**2 + 0.5 * z_mean**2 - 0.5
        kl_div = (kl_rot + torch.sum(kl_z, 1)).mean()
        return reconstr_error - kl_div

    def step(self,
             x: torch.Tensor,
             mode: str = "train") -> torch.Tensor:
        """
        Single training/test step
        """
        x_coord_ = self.x_coord.expand(x.size(0), *self.x_coord.size())
        if torch.cuda.is_available():
            x = x.cuda()
        if mode == "eval":
            with torch.no_grad():
                z_mean, z_logsd = self.encoder_net(x)
        else:
            z_mean, z_logsd = self.encoder_net(x)
        z_sd = torch.exp(z_logsd)
        z = self.reparameterize(z_mean, z_sd)
        phi = z[:, 0]  # angle
        if self.translation:
            dx = z[:, 1:3]  # translation
            dx = (dx * self.dx_prior).unsqueeze(1)
            z = z[:, 3:]  # image content
        else:
            dx = 0  # no translation
            z = z[:, 1:]  # image content
        x_coord_ = transform_coordinates(x_coord_, phi, dx)
        if mode == "eval":
            with torch.no_grad():
                x_reconstr = self.decoder_net(x_coord_, z)
        else:
            x_reconstr = self.decoder_net(x_coord_, z)
        return self.loss_fn(x, x_reconstr, z_mean, z_logsd)

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
            if self.recording and self.z_dim in [3, 5]:
                self.manifold2d(savefig=True, filename=str(e))
        self.save_model(self.savename)
        if self.recording and self.z_dim in [3, 5]:
            self.visualize_manifold_learning("./vae_learning")
        return


class VAE(BaseVAE):
    """
    Implements a standard Variational Autoencoder (VAE)

    Args:
        X_train (numpy array):
            3D or 4D stack of training images ( n x w x h or n x w x h x c )
        latent_dim (int):
            number of VAE latent dimensions associated with image content
        training_cycles (int):
            number of training 'epochs' (Default: 300)
        batch_size (int):
            size of training batch for each training epoch (Default: 200)
        test_size (float):
            proportion of the dataset for model evaluation (Default: 0.15)
        seed (int):
            seed for torch and numpy (pseudo-)random numbers generators
        **conv_encoder (bool):
            use convolutional layers in encoder
        **conv_decoder (bool):
            use convolutional layers in decoder
        **numlayers_encoder (int):
            number of layers in encoder (Default: 2)
        **numlayers_decoder (int):
            number of layers in decoder (Default: 2)
        **numhidden_encoder (int):
            number of hidden units OR conv filters in encoder (Default: 128)
        **numhidden_decoder (int):
            number of hidden units OR conv filters in decoder (Default: 128)
        **savename (str):
            file name/path for saving model at the end of training
    """
    def __init__(self,
                 X_train: Union[np.ndarray, Tuple[np.ndarray]],
                 latent_dim: int = 2,
                 training_cycles: int = 300,
                 batch_size: int = 200,
                 test_size: float = 0.15,
                 seed: int = 0,
                 **kwargs: Union[int, bool]) -> None:
        dim = X_train[0].shape[1:] if isinstance(X_train, tuple) else X_train.shape[1:]
        kwargs["num_classes"] = len(np.unique(X_train[1])) if isinstance(X_train, tuple) else 0
        coord = 0
        super(VAE, self).__init__(dim, latent_dim, coord, seed, **kwargs)

        if torch.cuda.is_available:
            torch.cuda.empty_cache()
        set_train_rng(seed)
        np.random.seed(seed)

        if isinstance(X_train, tuple):
            self.num_classes = kwargs.get("num_classes")
            X_train, X_test, y_train, y_test = train_test_split(
                X_train[0], X_train[1], test_size=test_size,
                shuffle=True, random_state=seed)
            iterators = init_vae_dataloaders(
                X_train, X_test, y_train, y_test, batch_size)
        else:
            X_train, X_test = train_test_split(
                X_train, test_size=test_size, shuffle=True,
                random_state=seed) 
            iterators = init_vae_dataloaders(X_train, X_test, batch_size)
        self.im_dim = X_train.shape[1:]

        self.train_iterator, self.test_iterator = iterators

        if torch.cuda.is_available():
            self.decoder_net.cuda()
            self.encoder_net.cuda()

        params = list(self.decoder_net.parameters()) +\
            list(self.encoder_net.parameters())
        self.optim = torch.optim.Adam(params, lr=1e-4)
        self.loss = kwargs.get("loss", "mse")
        self.training_cycles = training_cycles
        self.savename = kwargs.get("savename", "./VAE_metadict")

    def loss_fn(self, x: torch.Tensor, x_reconstr: torch.Tensor,
                *args: torch.Tensor, **kwargs: str) -> torch.Tensor:
        """
        Calculates ELBO
        """
        z_mean, z_logsd = args
        loss = kwargs.get("loss")
        if loss is None:
            loss = self.loss
        z_sd = torch.exp(z_logsd)
        batch_dim = x.size(0)
        if loss == "mse":
            reconstr_error = -0.5 * torch.sum(
                (x_reconstr.reshape(batch_dim, -1) - x.reshape(batch_dim, -1))**2, 1).mean()
        else:
            px_size = np.product(self.im_dim)
            rs = (self.im_dim[0] * self.im_dim[1],)
            if len(self.im_dim) == 3:
                rs = rs + (self.im_dim[-1],)
            reconstr_error = -F.binary_cross_entropy_with_logits(
                x_reconstr.reshape(-1, *rs), x.reshape(-1, *rs)) * px_size
        kl_z = -z_logsd + 0.5 * z_sd**2 + 0.5 * z_mean**2 - 0.5
        kl_z = torch.sum(kl_z, 1).mean()
        return reconstr_error - kl_z

    def step(self,
             x: torch.Tensor,
             y: Optional[torch.Tensor] = None,
             mode: str = "train") -> torch.Tensor:
        """
        Single training/test step
        """
        if y is not None and not hasattr(self, "num_classes"):
            raise AssertionError(
                "Please provide total number of classes as 'num_classes'")
        if torch.cuda.is_available():
            x = x.cuda()
        if mode == "eval":
            with torch.no_grad():
                z_mean, z_logsd = self.encoder_net(x)
        else:
            z_mean, z_logsd = self.encoder_net(x)
        z_sd = torch.exp(z_logsd)
        z = self.reparameterize(z_mean, z_sd)
        if y is not None:
            targets = to_onehot(y, self.num_classes)
            z = torch.cat((z, targets), -1)

        if mode == "eval":
            with torch.no_grad():
                x_reconstr = self.decoder_net(z)
        else:
            x_reconstr = self.decoder_net(z)

        return self.loss_fn(x, x_reconstr, z_mean, z_logsd)

    def run(self) -> None:
        """
        Trains VAE model
        """
        for e in range(self.training_cycles):
            elbo_epoch = self.train_epoch()
            elbo_epoch_test = self.evaluate_model()
            template = 'Epoch: {}/{}, Training loss: {:.4f}, Test loss: {:.4f}'
            print(template.format(e+1, self.training_cycles,
                  -elbo_epoch, -elbo_epoch_test))
        self.save_model(self.savename)
        return


def to_onehot(idx: torch.Tensor, n: int) -> torch.Tensor: # move to utils!
    """
    One-hot encoding of label
    """
    if torch.max(idx).item() >= n:
        raise AssertionError(
            "Labelling must start from 0 and "
            "maximum label value must be less than total number of classes")
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    onehot = torch.zeros(idx.size(0), n, device=device_)
    onehot.scatter_(1, idx, 1)
    return onehot


def load_vae_model(meta_dict: str) -> Type[BaseVAE]:
    """
    Loads trained AtomAI's VAE model

    Args:
        meta_state_dict (str):
            filepath to dictionary with trained weights and key information
            about model's structure

    Returns:
        VAE module
    """
    torch.manual_seed(0)
    if torch.cuda.device_count() > 0:
        meta_dict = torch.load(meta_dict)
    else:
        meta_dict = torch.load(meta_dict, map_location='cpu')
    im_dim = meta_dict.pop("im_dim")
    latent_dim = meta_dict.pop("latent_dim")
    coord = meta_dict.pop("coord")
    encoder_weights = meta_dict.pop("encoder")
    decoder_weights = meta_dict.pop("decoder")
    m = BaseVAE(im_dim, latent_dim, coord, **meta_dict)
    m.encoder_net.load_state_dict(encoder_weights)
    m.encoder_net.eval()
    m.decoder_net.load_state_dict(decoder_weights)
    m.decoder_net.eval()
    return m


def rvae(imstack: np.ndarray,
         latent_dim: int = 2,
         training_cycles: int = 300,
         minibatch_size: int = 200,
         test_size: float = 0.15,
         seed: int = 0,
         **kwargs: Union[int, bool]) -> Type[BaseVAE]:
    """
    "Wrapper function" for initializing rotationally invariant
    variational autoencoder (rVAE)

    Args:
        imstack (np.ndarray):
            3D or 4D stack of training images ( n x w x h or n x w x h x c )
        latent_dim (int):
            number of VAE latent dimensions associated with image content
        training_cycles (int):
            number of training 'epochs' (Default: 300)
        minibatch_size (int):
            size of training batch for each training epoch (Default: 200)
        test_size (float):
            proportion of the dataset for model evaluation (Default: 0.15)
        seed(int):
            seed for torch and numpy (pseudo-)random numbers generators
        **conv_encoder (bool):
            use convolutional layers in encoder
        **numlayers_encoder (int):
            number of layers in encoder (Default: 2)
        **numlayers_decoder (int):
            number of layers in decoder (Default: 2)
        **numhidden_encoder (int):
            number of hidden units OR conv filters in encoder (Default: 128)
        **numhidden_decoder (int):
            number of hidden units in decoder (Default: 128)
        **loss (str):
            reconstruction loss function, "ce" or "mse" (Default: "mse")
        **translation_prior (float):
            translation prior
        **rotation_prior (float):
            rotational prior
        **recording (bool):
            saves a learned 2d manifold at each training step
    """

    rvae_ = rVAE(
        imstack, latent_dim, training_cycles,
        minibatch_size, test_size, seed, **kwargs)
    return rvae_


def vae(imstack: np.ndarray,
        latent_dim: int = 2,
        training_cycles: int = 300,
        minibatch_size: int = 200,
        test_size: float = 0.15,
        seed: int = 0,
        **kwargs: Union[int, bool]) -> Type[BaseVAE]:
    """
    "Wrapper function" for initializing standard Variational Autoencoder (VAE)

    Args:
        imstack (numpy array):
            3D or 4D stack of training images ( n x w x h or n x w x h x c )
        latent_dim (int):
            number of VAE latent dimensions associated with image content
        training_cycles (int):
            number of training 'epochs' (Default: 300)
        minibatch_size (int):
            size of training batch for each training epoch (Default: 200)
        test_size (float):
            proportion of the dataset for model evaluation (Default: 0.15)
        seed (int):
            seed for torch and numpy (pseudo-)random numbers generators
        **conv_encoder (bool):
            use convolutional layers in encoder
        **conv_decoder (bool):
            use convolutional layers in decoder
        **numlayers_encoder (int):
            number of layers in encoder (Default: 2)
        **numlayers_decoder (int):
            number of layers in decoder (Default: 2)
        **numhidden_encoder (int):
            number of hidden units OR conv filters in encoder (Default: 128)
        **numhidden_decoder (int):
            number of hidden units OR conv filters in decoder (Default: 128)
    """
    vae_ = VAE(
        imstack, latent_dim, training_cycles,
        minibatch_size, test_size, seed, **kwargs)
    return vae_
