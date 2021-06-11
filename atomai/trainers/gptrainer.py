"""
gptrainer.py
============

Module for training deep kernel learning based Gaussian process regression.

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Optional, Tuple, Type, Union

import gpytorch
import numpy as np
import torch

from ..nets import GPRegressionModel, fcFeatureExtractor
from ..utils import set_seed_and_precision


class dklGPTrainer:
    """
    Deep kernel learning (DKL)-based Gaussian process regression (GPR)

    Args:
        indim: input feature dimension
        embedim: embedding dimension (determines dimensionality of kernel space)

    Keyword Args:
        device:
            Sets device to which model and data will be moved.
            Defaults to 'cuda:0' if a GPU is available and to CPU otherwise.
        precision:
            Sets tensor types for 'single' (torch.float32)
            or 'double' (torch.float64) precision
        seed:
            Seed for enforcing reproducibility
    """
    def __init__(self,
                 indim: int,
                 embedim: int = 2,
                 **kwargs: Union[str, int]) -> None:
        """
        Initializes DKL-GPR.
        """
        set_seed_and_precision(**kwargs)
        self.dimdict = {"input_dim": indim, "embedim": embedim}
        self.device = kwargs.get(
            "device", 'cuda:0' if torch.cuda.is_available() else 'cpu')
        precision = kwargs.get("precision", "double")
        self.dtype = torch.float32 if precision == "single" else torch.float64
        self.gp_model = None
        self.likelihood = None
        self.compiled = False
        self.train_loss = []

    def _set_data(self, x: Union[torch.Tensor, np.ndarray],
                  device: str = None) -> torch.tensor:
        """Data preprocessing."""
        device_ = device if device else self.device
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.dtype).to(device_)
        elif isinstance(x, torch.Tensor):
            x = x.to(self.dtype).to(device_)
        else:
            raise TypeError("Pass data as ndarray or torch tensor object")
        return x

    def set_data(self, x: Union[torch.Tensor, np.ndarray],
                 y: Optional[Union[torch.Tensor, np.ndarray]] = None,
                 device: str = None) -> Tuple[torch.tensor]:
        """Data preprocessing. Casts data array to a selected tensor type
        and moves it to a selected devive."""
        x = self._set_data(x, device)
        if y is not None:
            y = y[None] if y.ndim == 1 else y
            y = self._set_data(y, device)
        return x, y

    def compile_trainer(self, X: Union[torch.Tensor, np.ndarray],
                        y: Union[torch.Tensor, np.ndarray],
                        training_cycles: int = 1,
                        **kwargs: Union[Type[torch.nn.Module], int, bool, float]
                        ) -> None:
        """
        Initializes deep kernel (feature extractor NN + base kernel),
        sets optimizer and "loss" function.

        Args:
            X: Input training data (aka features) of N x input_dim dimensions
            y: Output targets of batch_size x N or N (if batch_size=1) dimensions
            training_cycles: Number of training epochs

        Keyword Args:
            feature_extractor:
                (Optional) Custom neural network for feature extractor
            grid_size:
                Grid size for structured kernel interpolation (Default: 50)
            freeze_weights:
                Freezes weights of feature extractor, that is, they are not
                passed to the optimizer. Used for a transfer learning.
            lr: learning rate (Default: 0.01)
        """
        X, y = self.set_data(X, y)
        input_dim, embedim = self.dimdict["input_dim"], self.dimdict["embedim"]
        feature_extractor = kwargs.get("feature_extractor")
        if feature_extractor is None:
            feature_extractor = fcFeatureExtractor(input_dim, embedim)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp_model = GPRegressionModel(
            X, y, likelihood, feature_extractor, embedim,
            kwargs.get("grid_size", 50))
        self.likelihood = likelihood
        self.gp_model.to(self.device)
        self.likelihood.to(self.device)
        self.gp_model.train()
        self.likelihood.train()
        list_of_params = [
            {'params': self.gp_model.covar_module.parameters()},
            {'params': self.gp_model.mean_module.parameters()},
            {'params': self.gp_model.likelihood.parameters()}]
        if not kwargs.get("freeze_weights", False):
            list_of_params.append(
                {'params': self.gp_model.feature_extractor.parameters()})
        self.optimizer = torch.optim.Adam(list_of_params, lr=kwargs.get("lr", 0.01))
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        self.training_cycles = training_cycles
        self.compiled = True

    def train_step(self, X: torch. Tensor, y: torch.Tensor) -> None:
        """
        Single training step with backpropagation
        to computegradients and optimizes weights.
        """
        self.optimizer.zero_grad()
        output = self.gp_model(X)
        loss = -self.mll(output, y).sum()
        loss.backward()
        self.optimizer.step()
        self.train_loss.append(loss.item())

    def run(self, X: Union[torch.Tensor, np.ndarray],
            y: Union[torch.Tensor, np.ndarray],
            training_cycles: int = 1,
            **kwargs: Union[Type[torch.nn.Module], int, bool, float]
            ) -> None:
        """
        Initializes and trains a deep kernel GP model

        Args:
            X: Input training data (aka features) of N x input_dim dimensions
            y: Output targets of batch_size x N or N (if batch_size=1) dimensions
            training_cycles: Number of training epochs

        Keyword Args:
            feature_extractor:
                (Optional) Custom neural network for feature extractor
            freeze_weights:
                Freezes weights of feature extractor, that is, they are not
                passed to the optimizer. Used for a transfer learning.
            grid_size:
                Grid size for structured kernel interpolation (Default: 50)
            lr: learning rate (Default: 0.01)
            print_loss: print loss at every n-th training cycle (epoch)
        """
        X, y = self.set_data(X, y)
        if not self.compiled:
            self.compile_trainer(X, y, training_cycles, **kwargs)
        for e in range(self.training_cycles):
            self.train_step(X, y)
            if any([e == 0, (e + 1) % kwargs.get("print_loss", 10) == 0,
                    e == self.training_cycles - 1]):
                self.print_statistics(e)
        return self.gp_model

    def print_statistics(self, e):
        print('Epoch {}/{} ...'.format(e+1, self.training_cycles),
              'Training loss: {}'.format(np.around(self.train_loss[-1], 4)))

    def save_weights(self, filename: str) -> None:
        """Saves weights of the feature extractor."""
        torch.save(self.gp_model.feature_extractor.state_dict(), filename)
