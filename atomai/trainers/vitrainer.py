"""
vitrainer.py
===========

Module for training VAE/VED models

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""


from typing import Callable, Optional, Tuple, Type, Union

import numpy as np
import torch

from ..utils import get_array_memsize, reset_bnorm, set_train_rng, weights_init


class viBaseTrainer:
    """
    Initializes base trainer for VAE and VED models
    """
    def __init__(self):
        set_train_rng(1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.in_dim = None
        self.out_dim = None
        self.z_dim = 1
        self.encoder_net = None
        self.decoder_net = None
        self.train_iterator = None
        self.test_iterator = None
        self.aux_model_params = []
        self.optim = None
        self.current_epoch = 0
        self.metadict = {}
        self.loss_history = {"train_loss": [], "test_loss": []}
        self.filename = "model"
        self.training_cycles = 1
        self.batch_size = 1

    def set_model(self,
                  encoder_net: Type[torch.nn.Module],
                  decoder_net: Type[torch.nn.Module]
                  ) -> None:
        """
        Sets encoder and decoder models
        """
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.encoder_net.to(self.device)
        self.decoder_net.to(self.device)

    def set_encoder(self,
                    encoder_net: Type[torch.nn.Module]
                    ) -> None:
        """
        Sets an encoder network only
        """
        self.encoder_net = encoder_net
        self.encoder_net.to(self.device)

    def set_decoder(self,
                    decoder_net: Type[torch.nn.Module]
                    ) -> None:
        """
        Sets a decoder network only
        """
        self.decoder_net = decoder_net
        self.decoder_net.to(self.device)

    def set_data(self,
                 X_train: Union[torch.Tensor, np.ndarray],
                 y_train: Union[torch.Tensor, np.ndarray] = None,
                 X_test: Union[torch.Tensor, np.ndarray] = None,
                 y_test: Union[torch.Tensor, np.ndarray] = None,
                 memory_alloc: float = 4) -> None:
        """
        Initializes train and (optionally) test data loaders
        """
        all_data = [X_train, y_train, X_test, y_test]
        arrsize = sum([get_array_memsize(x) for x in all_data])
        store_on_cpu = (arrsize / 1e9) > memory_alloc
        self.train_iterator = self._set_data(X_train, y_train, store_on_cpu)
        if X_test is not None:
            self.test_iterator = self._set_data(X_test, y_test, store_on_cpu)

    def _2torch(self,
                X: Union[np.ndarray, torch.Tensor],
                y: Union[np.ndarray, torch.Tensor] = None
                ) -> Tuple[torch.Tensor]:
        """
        ndarray to torch tensor conversion
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        return X, y

    def _set_data(self,
                  X: Union[np.ndarray, torch.Tensor],
                  y: Union[np.ndarray, torch.Tensor] = None,
                  store_on_cpu: bool = False
                  ) -> Type[torch.utils.data.DataLoader]:
        """
        Initializes PyTorch dataloader given a pair of ndarrays/tensors
        """
        if X is None:
            raise AssertionError(
                "You must provide input train/test data")
        device_ = 'cpu' if store_on_cpu else self.device
        X, y = self._2torch(X, y)
        X = X.to(device_)
        y = y.to(device_) if y is not None else y
        if y is not None:  # VED or cVAE
            data_train = torch.utils.data.TensorDataset(X, y)
        else:  # VAE
            data_train = torch.utils.data.TensorDataset(X,)
        data_loader = torch.utils.data.DataLoader(
            data_train, batch_size=self.batch_size,
            shuffle=True, drop_last=True)

        return data_loader

    def elbo_fn(self):
        """
        Computes ELBO
        """
        raise NotImplementedError

    def forward_compute_elbo(self):
        """
        Computes ELBO in "train" and "eval" modes.
        Specifically, it passes input data x through encoder,
        "compresses" it to latent variables z_mean and z_sd/z_logsd,
        performs reparametrization trick, passes the reparameterized
        latent vector through decoder to obtain y/x_reconstructed,
        and then computes the "loss" via self.elbo_fn, which usually takes
        as parameters x, y/x_reconstructed, z_mean, and z_sd/z_logsd.
        """
        raise NotImplementedError

    def _reset_rng(self, seed: int) -> None:
        """
        (re)sets seeds for pytorch and numpy random number generators
        """
        set_train_rng(seed)

    def _reset_weights(self) -> None:
        """
        Resets weights of convolutional and linear NN layers
        using Xavier initialization
        """
        self.encoder_net.apply(weights_init)
        self.encoder_net.apply(reset_bnorm)
        self.decoder_net.apply(weights_init)
        self.decoder_net.apply(reset_bnorm)

    def _reset_training_history(self) -> None:
        """
        Empties training/test losses and accuracies
        (can be useful for ensemble training)
        """
        self.loss_history = {"train_loss": [], "test_loss": []}

    def _delete_optimizer(self) -> None:
        """
        Sets optimizer to None.
        """
        self.optim = None

    def compile_trainer(self,
                        train_data: Tuple[Union[torch.Tensor, np.ndarray]],
                        test_data: Tuple[Union[torch.Tensor, np.ndarray]] = None,
                        optimizer: Optional[Type[torch.optim.Optimizer]] = None,
                        elbo_fn: Callable = None,
                        training_cycles: int = 100,
                        batch_size: int = 32,
                        **kwargs: Union[str, float]) -> None:
        """
        Compiles model's trainer

        Args:
            train_data:
                Train data and (optionally) corresponding targets or labels
            train_data:
                Test data and (optionally) corresponding targets or labels
            optimizer:
                Weights optimizer. Defaults to Adam with learning rate 1e-4
            elbo_fn:
                function that calculates elbo loss
            training_cycles:
                Number of training iterations (aka "epochs")
            batch_size:
                Size of mini-batch for training
            **kwargs:
                Additional keyword arguments are 'filename' (for saving model)
                and 'memory alloc' (threshold for keeping data on GPU)
        """
        self.training_cycles = training_cycles
        self.batch_size = batch_size
        if elbo_fn is not None:
            self.elbo_fn = elbo_fn
        alloc = kwargs.get("memory_alloc", 4)
        if test_data is not None:
            self.set_data(
                *train_data, *test_data, memory_alloc=alloc)
        else:
            self.set_data(*train_data, memory_alloc=alloc)

        params = list(self.decoder_net.parameters()) +\
            list(self.encoder_net.parameters())
        for aux_param in self.aux_model_params:
            params.extend(list(aux_param))
        if self.optim is None:
            if optimizer is None:
                self.optim = torch.optim.Adam(params, lr=1e-4)
            else:
                self.optim = optimizer(params)
        self.filename = kwargs.get("filename", "./model")

    @classmethod
    def reparameterize(cls,
                       z_mean: torch.Tensor,
                       z_sd: torch.Tensor
                       ) -> torch.Tensor:
        """
        Reparameterization trick for continuous distributions
        """
        batch_dim = z_mean.size(0)
        z_dim = z_mean.size(1)
        eps = z_mean.new(batch_dim, z_dim).normal_()
        return z_mean + z_sd * eps

    @classmethod
    def reparameterize_discrete(cls,
                                alpha: torch.Tensor,
                                tau: float):
        """
        Reparameterization trick for discrete gumbel-softmax distributions
        """
        eps = 1e-12
        su = alpha.new(alpha.size()).uniform_()
        gumbel = -torch.log(-torch.log(su + eps) + eps)
        log_alpha = torch.log(alpha + eps)
        logit = (log_alpha + gumbel) / tau
        return torch.nn.functional.softmax(logit, dim=1)

    def kld_normal(self,
                   z: torch.Tensor,
                   q_param: Tuple[torch.Tensor],
                   p_param: Optional[Tuple[torch.Tensor]] = None
                   ) -> torch.Tensor:
        """
        Calculates KL divergence term between two normal distributions
        or (if p_param = None) between normal and standard normal distributions

        Args:
            z: latent vector (reparametrized)
            q_param: tuple with mean and SD of the 1st distribution
            p_param: tuple with mean and SD of the 2nd distribution (optional)
        """
        qz = self.log_normal(z, *q_param)
        if p_param is None:
            pz = self.log_unit_normal(z)
        else:
            pz = self.log_normal(z, *p_param)
        return qz - pz

    @classmethod
    def log_normal(cls,
                   x: torch.Tensor,
                   mu: torch.Tensor,
                   log_sd: torch.Tensor
                   ) -> torch.Tensor:
        """
        Computes log-pdf for a normal distribution
        """
        log_pdf = (-0.5 * np.log(2 * np.pi) - log_sd -
                   (x - mu)**2 / (2 * torch.exp(log_sd)**2))
        return torch.sum(log_pdf, dim=-1)

    @classmethod
    def log_unit_normal(cls, x: torch.Tensor) -> torch.Tensor:
        """
        Computes log-pdf of a unit normal distribution
        """
        log_pdf = -0.5 * (np.log(2 * np.pi) + x ** 2)
        return torch.sum(log_pdf, dim=-1)

    def train_epoch(self):
        """
        Trains a single epoch
        """
        step = self.forward_compute_elbo
        self.decoder_net.train()
        self.encoder_net.train()
        c = 0
        elbo_epoch = 0
        for x in self.train_iterator:
            if len(x) == 1:  # VAE mode
                x = x[0].to(self.device)
                y = None
            else:  # VED or cVAE mode
                x, y = x
                x, y = x.to(self.device), y.to(self.device)
            b = x.size(0)
            elbo = step(x) if y is None else step(x, y)
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
        step = self.forward_compute_elbo
        self.decoder_net.eval()
        self.encoder_net.eval()
        c = 0
        elbo_epoch_test = 0
        for x in self.test_iterator:
            if len(x) == 1:
                x = x[0].to(self.device)
                y = None
            else:
                x, y = x
                x, y = x.to(self.device), y.to(self.device)
            b = x.size(0)
            if y is None:  # VAE mode
                elbo = step(x, mode="eval")
            else:  # VED or cVAE mode
                elbo = step(x, y, mode="eval")
            elbo = elbo.item()
            c += b
            delta = b * (elbo - elbo_epoch_test)
            elbo_epoch_test += delta / c
        return elbo_epoch_test

    def print_statistics(self, e):
        """
        Prints training and (optionally) test loss after each training cycle
        """
        if self.test_iterator is not None:
            template = 'Epoch: {}/{}, Training loss: {:.4f}, Test loss: {:.4f}'
            print(template.format(e+1, self.training_cycles,
                  -self.loss_history["train_loss"][-1],
                  -self.loss_history["test_loss"][-1]))
        else:
            template = 'Epoch: {}/{}, Training loss: {:.4f}'
            print(template.format(e+1, self.training_cycles,
                  -self.loss_history["train_loss"][-1]))

    def save_model(self, *args: str) -> None:
        """
        Saves trained weights and the key model parameters
        """
        try:
            savepath = args[0]
        except IndexError:
            savepath = self.filename
        self.metadict["encoder"] = self.encoder_net.state_dict()
        self.metadict["decoder"] = self.decoder_net.state_dict()
        self.metadict["optimizer"] = self.optim
        torch.save(self.metadict, savepath + ".tar")

    def save_weights(self, *args: str) -> None:
        """
        Saves trained weights
        """
        try:
            savepath = args[0]
        except IndexError:
            savepath = self.filename + "weights"
        torch.save({"encoder": self.encoder_net.state_dict(),
                   "decoder": self.decoder_net.state_dict()},
                   savepath + ".tar")

    def load_weights(self, filepath: str) -> None:
        """
        Loads saved weights
        """
        weights = torch.load(filepath, map_location=self.device)
        encoder_weights = weights["encoder"]
        decoder_weights = weights["decoder"]
        self.encoder_net.load_state_dict(encoder_weights)
        self.encoder_net.eval()
        self.decoder_net.load_state_dict(decoder_weights)
        self.decoder_net.eval()
