"""
gp.py
=====

Modules for Gaussian process regression with deep kernel learning
"""

from typing import Type, Optional, Union, Tuple, List

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


class CustomGPModel(gpytorch.models.ExactGP):
    def __init__(self,
                 train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood,
                 kernel_type: str = 'kissgp',
                 base_kernel: Union[str, gpytorch.kernels.Kernel] = 'rbf',
                 inducing_points: Optional[torch.Tensor] = None, grid_points_ratio: int = 1.0,
                 lengthscale_constraints: Optional[Tuple[List[float]]] = None, **kwargs):
        """
        Custom GP Model that allows the user to choose different base kernels, kernel types, and lengthscales.

        Args:
            train_x: Input training data.
            train_y: Output training data.
            likelihood: Gaussian likelihood object.
            kernel_type: Type of kernel to use, either 'sparse' or 'kissgp'. Defaults to 'sparse'.
            base_kernel: Name of the base kernel as a string, either 'rbf' or 'matern', or a custom base kernel object. Defaults to 'rbf'.
            inducing_points: Inducing points for the sparse kernel. Defaults to None.
            grid_points_ratio: Determines a grid size for the KISS-GP kernel. Defaults to 1.0
            lengthscale_contraints: Optional lengthscale constraints for the base kernel. Defaults to None.
        """
        super(CustomGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        if isinstance(base_kernel, str):
            
            if lengthscale_constraints:
                lengthscale_constraints = gpytorch.constraints.Interval(
                    torch.tensor(lengthscale_constraints[0]),
                    torch.tensor(lengthscale_constraints[1]))
            
            if base_kernel == 'rbf':
                base_kernel = gpytorch.kernels.RBFKernel(
                    ard_num_dims=train_x.shape[-1],
                    lengthscale_constraint=lengthscale_constraints)
            elif base_kernel == 'matern':
                base_kernel = gpytorch.kernels.MaternKernel(
                    ard_num_dims=train_x.shape[-1],
                    lengthscale_constraint=lengthscale_constraints)
            else:
                raise ValueError("base_kernel must be either 'rbf', 'matern', or a custom gpytorch.kernels.Kernel object")

        self.base_covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

        if kernel_type == 'sparse':
            self.covar_module = gpytorch.kernels.InducingPointKernel(
                self.base_covar_module, inducing_points=inducing_points, likelihood=likelihood)
        elif kernel_type == 'kissgp':
            grid_size = gpytorch.utils.grid.choose_grid_size(train_x, grid_points_ratio)
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                self.base_covar_module, grid_size=grid_size, num_dims=train_x.shape[-1])
        else:
            raise ValueError(
                f"Invalid kernel_type: {kernel_type}. Supported values are 'sparse' and 'kissgp'.")

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward pass for the GP model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            Multivariate normal distribution representing the predicted output.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
