import sys
import pytest
import torch
import gpytorch

sys.path.append("../../../")

from atomai.nets import CustomGPModel


def test_init_custom_gp_model_with_default_parameters():
    train_x = torch.randn(10, 2)
    train_y = torch.randn(10)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = CustomGPModel(train_x, train_y, likelihood)

    assert isinstance(model.mean_module, gpytorch.means.ConstantMean)
    assert isinstance(model.base_covar_module, gpytorch.kernels.ScaleKernel)
    assert isinstance(model.base_covar_module.base_kernel, gpytorch.kernels.RBFKernel)
    assert isinstance(model.covar_module, gpytorch.kernels.GridInterpolationKernel)


def test_init_custom_gp_model_with_matern_kernel():
    train_x = torch.randn(10, 2)
    train_y = torch.randn(10)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = CustomGPModel(
        train_x, train_y, likelihood, base_kernel='matern')

    assert isinstance(model.mean_module, gpytorch.means.ConstantMean)
    assert isinstance(model.base_covar_module, gpytorch.kernels.ScaleKernel)
    assert isinstance(model.base_covar_module.base_kernel, gpytorch.kernels.MaternKernel)
    assert isinstance(model.covar_module, gpytorch.kernels.GridInterpolationKernel)


def test_init_custom_gp_model_with_invalid_base_kernel():
    train_x = torch.randn(10, 2)
    train_y = torch.randn(10)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    with pytest.raises(ValueError):
        CustomGPModel(
            train_x, train_y, likelihood, base_kernel='invalid_kernel')


def test_init_custom_gp_model_with_invalid_kernel_type():
    train_x = torch.randn(10, 2)
    train_y = torch.randn(10)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    with pytest.raises(ValueError):
        CustomGPModel(
            train_x, train_y, likelihood, kernel_type='invalid_kernel_type')


def test_custom_gp_model_forward():
    h = w = 10
    pixel_indices = torch.tensor([(i, j) for i in range(h) for j in range(w)])

    train_x = pixel_indices[::2]
    train_y = torch.randn(len(train_x))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = CustomGPModel(train_x, train_y, likelihood)

    test_x = pixel_indices

    model.eval()
    output = model(test_x)

    assert isinstance(output, gpytorch.distributions.MultivariateNormal)
    assert output.mean.shape == (100,)
    assert output.covariance_matrix.shape == (100, 100)


def test_init_custom_gp_model_with_sparse_kernel():
    train_x = torch.randn(10, 2)
    train_y = torch.randn(10)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    inducing_points = torch.randn(5, 2)

    model = CustomGPModel(
        train_x, train_y, likelihood,
        inducing_points=inducing_points, kernel_type='sparse')

    assert isinstance(model.mean_module, gpytorch.means.ConstantMean)
    assert isinstance(model.base_covar_module, gpytorch.kernels.ScaleKernel)
    assert isinstance(model.base_covar_module.base_kernel, gpytorch.kernels.RBFKernel)
    assert isinstance(model.covar_module, gpytorch.kernels.InducingPointKernel)
    assert torch.all(torch.eq(model.covar_module.inducing_points, inducing_points))


def test_init_custom_gp_model_with_custom_kernel():
    train_x = torch.randn(10, 2)
    train_y = torch.randn(10)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    custom_kernel = gpytorch.kernels.PolynomialKernel(power=2)

    inducing_points = torch.randn(5, 2)

    model = CustomGPModel(
        train_x, train_y, likelihood,
        inducing_points=inducing_points,
        kernel_type='sparse', base_kernel=custom_kernel)

    assert isinstance(model.mean_module, gpytorch.means.ConstantMean)
    assert isinstance(model.base_covar_module, gpytorch.kernels.ScaleKernel)
    assert isinstance(model.base_covar_module.base_kernel, gpytorch.kernels.PolynomialKernel)
    assert isinstance(model.covar_module, gpytorch.kernels.InducingPointKernel)
