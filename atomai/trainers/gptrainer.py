"""
gptrainer.py
============

Module for training deep kernel learning based Gaussian process regression.

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""
from copy import deepcopy as dc
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
                 shared_embedding_space: bool = True,
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
        self.correlated_output = shared_embedding_space
        self.gp_model = None
        self.likelihood = None
        self.ensemble = False
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

    def compile_multi_model_trainer(self,
                                    X: Union[torch.Tensor, np.ndarray],
                                    y: Union[torch.Tensor, np.ndarray],
                                    training_cycles: int = 1,
                                    **kwargs: Union[Type[torch.nn.Module], int, bool, float]
                                    ) -> None:

        """
        Initializes deep kernel (feature extractor NNs + base kernels),
        sets optimizer and "loss" function. For vector-valued functions
        (multiple outputs), it assumes one latent space per output, that is,
        the number of neural networks is equal to the number of Gaussian
        processes. For example, if the outputs are spectra of length 128,
        one will have 128 neural networks and 128 GPs trained in parallel.
        It can be also used for training an ensembles of models for the same
        scalar output.
        """
        if self.correlated_output:
            raise NotImplementedError(
                "To compile a DKL-GP trainer for correlated outputs " +
                "use compile_trainer(*args, **kwargs)")
        X, y = self.set_data(X, y)
        if y.shape[0] < 2:
            raise ValueError("The training targets must be vector-valued (d >1)")
        input_dim, embedim = self.dimdict["input_dim"], self.dimdict["embedim"]
        feature_net = kwargs.get("feature_extractor", fcFeatureExtractor)
        freeze_weights = kwargs.get("freeze_weights", False)
        if not self.ensemble:
            feature_extractor = feature_net(input_dim, embedim)
            if freeze_weights:
                for p in feature_extractor.parameters():
                    p.requires_grad = False
        list_of_models = []
        list_of_likelihoods = []
        for i in range(y.shape[0]):
            if self.ensemble:  # different initilization for each model
                feature_extractor = feature_net(input_dim, embedim)
                if freeze_weights:
                    for p in feature_extractor.parameters():
                        p.requires_grad = False
            model_i = GPRegressionModel(
                X, y[i:i+1],
                gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([1])),
                feature_extractor, embedim, kwargs.get("grid_size", 50))
            list_of_models.append(dc(model_i))
            list_of_likelihoods.append(dc(model_i.likelihood))
        self.gp_model = gpytorch.models.IndependentModelList(*list_of_models)
        self.likelihood = gpytorch.likelihoods.LikelihoodList(*list_of_likelihoods)
        self.gp_model.to(self.device)
        self.likelihood.to(self.device)

        list_of_parameters = []
        for m in self.gp_model.models:
            list_of_parameters += list(m.covar_module.parameters())
            list_of_parameters += list(m.mean_module.parameters())
            list_of_parameters += list(m.likelihood.parameters())
            if not freeze_weights:
                list_of_parameters += list(m.feature_extractor.parameters())

        self.optimizer = torch.optim.Adam(list_of_parameters, lr=0.01)
        self.mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.gp_model)

        self.training_cycles = training_cycles
        self.compiled = True

    def compile_trainer(self, X: Union[torch.Tensor, np.ndarray],
                        y: Union[torch.Tensor, np.ndarray],
                        training_cycles: int = 1,
                        **kwargs: Union[Type[torch.nn.Module], int, bool, float]
                        ) -> None:
        """
        Initializes deep kernel (feature extractor NN + base kernel),
        sets optimizer and "loss" function. For vector-valued functions
        (multiple outputs), it assumes a shared latent space, that is,
        a single neural network is connected to multiple Gaussian processes.

        Args:
            X: Input training data (aka features) of N x input_dim dimensions
            y: Output targets of batch_size x N or N (if batch_size=1) dimensions
            training_cycles: Number of training epochs

        Keyword Args:
            feature_extractor:
                (Optional) Custom neural network for feature extractor.
                Must take input/feature dims and embedding dims as its arguments.
            grid_size:
                Grid size for structured kernel interpolation (Default: 50)
            freeze_weights:
                Freezes weights of feature extractor, that is, they are not
                passed to the optimizer. Used for a transfer learning.
            lr: learning rate (Default: 0.01)
        """
        if not self.correlated_output:
            raise NotImplementedError(
                "To compile a DKL-GP trainer for independent outputs " +
                "use compile_multi_model_trainer(*args, **kwargs)")
        X, y = self.set_data(X, y)
        input_dim, embedim = self.dimdict["input_dim"], self.dimdict["embedim"]
        feature_net = kwargs.get("feature_extractor", fcFeatureExtractor)
        feature_extractor = feature_net(input_dim, embedim)
        freeze_weights = kwargs.get("freeze_weights", False)
        if freeze_weights:
            for p in feature_extractor.parameters():
                p.requires_grad = False
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            batch_shape=torch.Size([y.shape[0]]))
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
        if not freeze_weights:
            list_of_params.append(
                {'params': self.gp_model.feature_extractor.parameters()})
        self.optimizer = torch.optim.Adam(list_of_params, lr=kwargs.get("lr", 0.01))
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        self.training_cycles = training_cycles
        self.compiled = True

    def train_step(self) -> None:
        """
        Single training step with backpropagation
        to computegradients and optimizes weights.
        """
        self.optimizer.zero_grad()
        X, y = self.gp_model.train_inputs, self.gp_model.train_targets
        output = self.gp_model(*X)
        loss = -self.mll(output, y).sum()
        loss.backward()
        self.optimizer.step()
        self.train_loss.append(loss.item())

    def run(self, X: Union[torch.Tensor, np.ndarray] = None,
            y: Union[torch.Tensor, np.ndarray] = None,
            training_cycles: int = 1,
            **kwargs: Union[Type[torch.nn.Module], int, bool, float]
            ) -> Type[gpytorch.models.ExactGP]:
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
        if not self.compiled:
            if self.correlated_output:
                self.compile_trainer(X, y, training_cycles, **kwargs)
            else:
                self.compile_multi_model_trainer(X, y, training_cycles, **kwargs)
        for e in range(self.training_cycles):
            self.train_step()
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
