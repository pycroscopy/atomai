"""
vae.py
===========

Module for analysis of system "building blocks"" with variational autoencoders

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import os
from typing import Dict, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from atomai.nets import EncoderNet, DecoderNet, rDecoderNet
from atomai.utils import (crop_borders, extract_subimages, get_coord_grid,
                          subimg_trajectories, transform_coordinates)
from scipy.stats import norm
from sklearn.model_selection import train_test_split


class EncoderDecoder:
    """
    General class for Encoder-Decoder type of deep latent variable models

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
        skip = kwargs.get("skip", False)
        if not coord:
            self.decoder_net = DecoderNet(
                latent_dim, numlayers_d, numhidden_d, self.im_dim, mlp_d)
        else:
            self.decoder_net = rDecoderNet(
                latent_dim, numlayers_d, numhidden_d, self.im_dim,
                skip)
        self.encoder_net = EncoderNet(
            self.im_dim, self.z_dim, numlayers_e, numhidden_e, mlp_e)

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
            "skip": skip
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

    def decode(self, z_sample: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Takes a point in latent space and and maps it to data space
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
        """
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
                 imstack: np.ndarray,
                 latent_dim: int = 2,
                 training_cycles: int = 300,
                 minibatch_size: int = 200,
                 test_size: float = 0.15,
                 seed: int = 0,
                 **kwargs: Union[int, float, bool, str]) -> None:
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

        self.savename = kwargs.get("savename", "./rVAE_metadict")
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
        self.save_model(self.savename)
        if self.recording and self.z_dim == 5:
            self.visualize_manifold_learning("./vae_learning")
        return

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


class VAE(EncoderDecoder):
    """
    Implements a standard Variational Autoencoder (VAE)

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
        **savename (str):
            file name/path for saving model at the end of training
    """
    def __init__(self,
                 imstack: np.ndarray,
                 latent_dim: int = 2,
                 training_cycles: int = 300,
                 minibatch_size: int = 200,
                 test_size: float = 0.15,
                 seed: int = 0,
                 **kwargs: Union[int, bool]) -> None:
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
        self.loss = kwargs.get("loss", "mse")

        self.training_cycles = training_cycles

        self.savename = kwargs.get("savename", "./VAE_metadict")

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

    def train_epoch(self) -> None:
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

    def evaluate_model(self) -> None:
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


def load_vae_model(meta_dict: str) -> Type[EncoderDecoder]:
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
    m = EncoderDecoder(im_dim, latent_dim, coord, **meta_dict)
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
         **kwargs: Union[int, bool]) -> Type[EncoderDecoder]:
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
        **kwargs: Union[int, bool]) -> Type[EncoderDecoder]:
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
