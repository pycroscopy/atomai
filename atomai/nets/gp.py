"""
gp.py
=====

Modules for Gaussian process regression with deep kernel learning
"""

from typing import Type

import torch
import gpytorch


class fcFeatureExtractor(torch.nn.Sequential):
    """MLP feature extractor"""
    def __init__(self, feat_dim, embedim, **kwargs):
        """Initializes a feature extractor module"""
        super(fcFeatureExtractor, self).__init__()
        hidden_dim = kwargs.get("hidden_dim")
        if hidden_dim is None:
            hidden_dim = [1000, 500, 50]
        hidden_dim.append(embedim)
        self.add_module("linear1", torch.nn.Linear(feat_dim, hidden_dim[0]))
        for i, h in enumerate(hidden_dim[1:]):
            self.add_module('relu{}'.format(i+1), torch.nn.ReLU())
            self.add_module('linear{}'.format(i+2), torch.nn.Linear(hidden_dim[i], h))


class GPRegressionModel(gpytorch.models.ExactGP):
    """DKL GPR module"""
    def __init__(self, X: torch.Tensor, y: torch.Tensor,
                 likelihood: Type[gpytorch.likelihoods.Likelihood],
                 feature_extractor: Type[torch.nn.Module], embedim: int,
                 grid_size: int = 50) -> None:
        """
        Initializes DKL GP module
        """
        super(GPRegressionModel, self).__init__(X, y, likelihood)
        batch_dim = y.size(0)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch_dim]))
        base_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=embedim, batch_shape=torch.Size([batch_dim])),
                batch_shape=torch.Size([batch_dim]))
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            base_kernel, num_dims=embedim, grid_size=grid_size)
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward pass
        """
        # Pass data through a neural network
        embedded_x = self.feature_extractor(x)
        embedded_x = self.scale_to_bounds(embedded_x)
        # Standard GP part
        mean_x = self.mean_module(embedded_x)
        covar_x = self.covar_module(embedded_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
