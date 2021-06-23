"""
dklgpr.py
=========

Deep kernel learning (DKL)-based gaussian process regression (GPR)

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import warnings
from typing import Tuple, Type, Union

import gpytorch
import numpy as np
import torch

from ...trainers import dklGPTrainer
from ...utils import init_dataloader


class dklGPR(dklGPTrainer):
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
        Initializes DKL-GPR model
        """
        super(dklGPR, self).__init__(indim, embedim, **kwargs)

    def fit(self, X: Union[torch.Tensor, np.ndarray],
            y: Union[torch.Tensor, np.ndarray],
            training_cycles: int = 1,
            **kwargs: Union[Type[torch.nn.Module], bool, float]
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
            lr: learning rate (Default: 0.01)
            print_loss: print loss at every n-th training cycle (epoch)
        """
        _ = self.run(X, y, training_cycles, **kwargs)

    def _compute_posterior(self, X: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Computes the posterior over model outputs at the provided points (X).
        """
        self.gp_model.eval()
        self.likelihood.eval()
        wrn = gpytorch.models.exact_gp.GPInputWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=wrn)
            with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
                posterior = self.gp_model(X.to(self.device))
        return posterior

    def sample_from_posterior(self, X, num_samples: int = 1000) -> np.ndarray:
        """
        Computes the posterior over model outputs at the provided points (X)
        and samples from it
        """
        X, _ = self.set_data(X)
        posterior = self._compute_posterior(X)
        samples = posterior.rsample(torch.Size([num_samples, ]))
        return samples.cpu().numpy()

    def _predict(self, x_new: torch.Tensor) -> Tuple[torch.Tensor]:
        y_pred = self._compute_posterior(x_new)
        return y_pred.mean.cpu(), y_pred.variance.cpu()

    def predict(self, x_new: Union[torch.Tensor, np.ndarray],
                **kwargs) -> Tuple[np.ndarray]:
        """
        Prediction of mean and variance using the trained model
        """
        gp_batch_dim = self.gp_model.train_targets.size(0)
        x_new, _ = self.set_data(x_new, device='cpu')
        data_loader = init_dataloader(x_new, shuffle=False, **kwargs)
        predicted_mean, predicted_var = [], []
        for (x,) in data_loader:
            x = x.expand(gp_batch_dim, *x.shape)
            mean, var = self._predict(x)
            predicted_mean.append(mean)
            predicted_var.append(var)
        return (torch.cat(predicted_mean, 1).numpy().squeeze(),
                torch.cat(predicted_var, 1).numpy().squeeze())

    def _embed(self, x_new: torch.Tensor):
        self.gp_model.feature_extractor.eval()
        with torch.no_grad():
            embeded = self.gp_model.feature_extractor(x_new)
        return embeded.cpu()

    def embed(self, x_new: Union[torch.Tensor, np.ndarray],
              **kwargs: int) -> torch.Tensor:
        """
        Embeds the input data to a "latent" space using a trained feature extractor NN.
        """
        x_new, _ = self.set_data(x_new, device='cpu')
        data_loader = init_dataloader(x_new, shuffle=False, **kwargs)
        embeded = torch.cat([self._embed(x.to(self.device)) for (x,) in data_loader], 0)
        return embeded.numpy()
