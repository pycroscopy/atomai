"""
gp.py
=====

Modules for Gaussian process regression with deep kenrel learning
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
                 feature_extractor: Type[torch.nn.Module], embedim: int) -> None:
        """
        Initializes DKL GP module
        """
        super(GPRegressionModel, self).__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        batch_dim = y.size(0)
        base_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=embedim, batch_shape=torch.Size([batch_dim])),
                batch_shape=torch.Size([batch_dim]))
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            base_kernel, num_dims=embedim, grid_size=50)
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
