"""
vae_losses.py
=========

Custom loss functions for Variational Autoencoders (VAEs)
"""
from typing import Tuple, List, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F


def reconstruction_loss(loss_type: str,
                        in_dim: Tuple[int],
                        x: torch.Tensor,
                        x_reconstr: torch.Tensor,
                        logits: bool = True,
                        ) -> torch.Tensor:
    """
    Computes reconstruction loss (mse or cross-entropy)
    without mean reduction (used in VAE objectives)
    """
    batch_dim = x.size(0)
    if loss_type == "mse":
        reconstr_loss = 0.5 * torch.sum(
            (x_reconstr.reshape(batch_dim, -1) - x.reshape(batch_dim, -1))**2, 1)
    elif loss_type == "ce":
        rs = (np.product(in_dim[:2]),)
        if len(in_dim) == 3:
            rs = rs + (in_dim[-1],)
        xe = (F.binary_cross_entropy_with_logits if
              logits else F.binary_cross_entropy)
        reconstr_loss = xe(x_reconstr.reshape(-1, *rs), x.reshape(-1, *rs),
                           reduction='none').sum(-1)
    else:
        raise NotImplementedError("Reconstruction loss must be 'mse' or 'ce'")
    return reconstr_loss


def kld_normal(q_param: Tuple[torch.Tensor],
               p_param: Optional[Tuple[torch.Tensor]] = None
               ) -> torch.Tensor:
    """
    Kullback–Leibler (KL) divergence between two normal distributions
    """
    mu_1, log_sd_1 = q_param
    sd_1 = torch.exp(log_sd_1)
    if p_param is None:
        # KL divergence b/w normal and standard normal distributions
        kl = -log_sd_1 + 0.5 * sd_1**2 + 0.5 * mu_1**2 - 0.5
    else:
        mu_2, log_sd_2 = p_param
        sd_2 = torch.exp(log_sd_2)
        # KL divergence b/w two normal distributions
        kl = (log_sd_2 - log_sd_1 +
              0.5 * (sd_1**2 + (mu_1 - mu_2)**2) / sd_2**2 - 0.5)
    return torch.sum(kl, -1)


def kld_discrete(alpha: torch.Tensor):
    """
    Calculates the KL divergence between a Gumbel-Softmax distribution
    and a uniform categorical distribution.

    Args:
        alpha:
            Parameters of the Gumbel-Softmax distribution.
    """
    eps = 1e-12
    cat_dim = alpha.size(-1)
    h1 = torch.log(alpha + eps)
    h2 = np.log(1. / cat_dim + eps)
    kld_loss = torch.mean(torch.sum(alpha * (h1 - h2), dim=1), dim=0)
    return kld_loss.view(1)


def kld_rot(phi_prior: torch.Tensor, phi_logsd: torch.Tensor) -> torch.Tensor:
    """
    Kullback–Leibler (KL) divergence for rotation latent variable
    """
    phi_sd = torch.exp(phi_logsd)
    kl_rot = (-phi_logsd + np.log(phi_prior) +
              phi_sd**2 / (2 * phi_prior**2) - 0.5)
    return kl_rot


def vae_loss(recon_loss: str,
             in_dim: Tuple[int],
             x: torch.Tensor,
             x_reconstr: torch.Tensor,
             *args: torch.Tensor,
             ) -> torch.Tensor:
    """
    Calculates ELBO
    """
    if len(args) == 2:
        q_param = args
    else:
        raise ValueError(
            "Pass mean and SD values of encoded distribution as args")
    likelihood = -reconstruction_loss(recon_loss, in_dim, x, x_reconstr).mean()
    kl_z = kld_normal(q_param).mean()
    return likelihood - kl_z


def rvae_loss(recon_loss: str,
              in_dim: Tuple[int],
              x: torch.Tensor,
              x_reconstr: torch.Tensor,
              *args: torch.Tensor,
              **kwargs: float) -> torch.Tensor:
    """
    Calculates ELBO
    """
    if len(args) == 2:
        z_mean, z_logsd = args
    else:
        raise ValueError(
            "Pass mean and SD values of encoded distribution as args")
    phi_prior = kwargs.get("phi_prior", 0.1)
    b1, b2 = kwargs.get("b1", 1), kwargs.get("b2", 1)
    phi_logsd = z_logsd[:, 0]
    z_mean, z_logsd = z_mean[:, 1:], z_logsd[:, 1:]
    likelihood = -reconstruction_loss(recon_loss, in_dim, x, x_reconstr).mean()
    kl_rot = kld_rot(phi_prior, phi_logsd).mean()
    kl_z = kld_normal([z_mean, z_logsd]).mean()
    kl_div = (b1*kl_z + b2 * kl_rot)
    return likelihood - kl_div


def joint_vae_loss(recon_loss: str,
                   in_dim: Tuple[int],
                   x: torch.Tensor,
                   x_reconstr: torch.Tensor,
                   *args: torch.Tensor,
                   **kwargs: Union[List, int],
                   ) -> torch.Tensor:
    """
    Calculates joint ELBO for continuous and discrete variables
    """
    if len(args) == 3:
        z_mean, z_logsd, alphas = args
    else:
        raise ValueError(
            "Pass continuous (mean, SD) and discrete (alphas) values" +
            "of encoded distributions as args")

    cont_capacity = kwargs.get("cont_capacity", [0.0, 5.0, 25000, 30])
    disc_capacity = kwargs.get("disc_capacity", [0.0, 5.0, 25000, 30])
    num_iter = kwargs.get("num_iter", 0)
    disc_dims = [a.size(1) for a in alphas]

    # Calculate reconstruction loss term
    likelihood = -reconstruction_loss(recon_loss, in_dim, x, x_reconstr).mean()

    # Calculate KL term for continuous latent variables
    kl_cont_loss = kld_normal([z_mean, z_logsd]).mean()
    # Calculate KL term for discrete latent variables
    kl_disc = [kld_discrete(alpha) for alpha in alphas]
    kl_disc_loss = torch.sum(torch.cat(kl_disc))

    # Apply information capacity terms to contninuous and discrete channels
    cargs = [kl_cont_loss, kl_disc_loss, cont_capacity,
             disc_capacity, disc_dims, num_iter]
    cont_capacity_loss, disc_capacity_loss = infocapacity(*cargs)

    return likelihood - cont_capacity_loss - disc_capacity_loss


def joint_rvae_loss(recon_loss: str,
                    in_dim: Tuple[int],
                    x: torch.Tensor,
                    x_reconstr: torch.Tensor,
                    *args: torch.Tensor,
                    **kwargs: float) -> torch.Tensor:
    """
    Calculates joint ELBO for continuous and discrete variables
    """
    if len(args) == 3:
        z_mean, z_logsd, alphas = args
    else:
        raise ValueError(
            "Pass continuous (mean, SD) and discrete (alphas) values" +
            "of encoded distributions as args")

    phi_prior = kwargs.get("phi_prior", 0.1)
    klrot_cap = kwargs.get("klrot_cap", True)
    cont_capacity = kwargs.get("cont_capacity", [0.0, 5.0, 25000, 30])
    disc_capacity = kwargs.get("disc_capacity", [0.0, 5.0, 25000, 30])
    num_iter = kwargs.get("num_iter", 0)

    # Calculate reconstruction loss term
    likelihood = -reconstruction_loss(recon_loss, in_dim, x, x_reconstr).mean()

    # Calculate KL term for continuous latent variables
    phi_logsd = z_logsd[:, 0]  # rotation
    z_mean, z_logsd = z_mean[:, 1:], z_logsd[:, 1:]  # image content
    kl_rot = kld_rot(phi_prior, phi_logsd).mean()
    kl_z = kld_normal([z_mean, z_logsd]).mean()
    if klrot_cap:
        kl_cont_loss = kl_z + kl_rot
    else:  # no capacity limit on KL term associated with rotations
        kl_cont_loss = kl_z

    # Calculate KL term for discrete latent variables
    disc_dims = [a.size(1) for a in alphas]
    kl_disc = [kld_discrete(alpha) for alpha in alphas]
    kl_disc_loss = torch.sum(torch.cat(kl_disc))

    # Apply information capacity terms to contninuous and discrete channels
    cargs = [kl_cont_loss, kl_disc_loss, cont_capacity,
             disc_capacity, disc_dims, num_iter]
    cont_capacity_loss, disc_capacity_loss = infocapacity(*cargs)
    if not klrot_cap:
        cont_capacity_loss = cont_capacity_loss + kl_rot

    return likelihood - cont_capacity_loss - disc_capacity_loss


def infocapacity(kl_cont_loss: torch.Tensor,
                 kl_disc_loss: torch.Tensor,
                 cont_capacity: List[float],
                 disc_capacity: List[float],
                 disc_dims: List[int],
                 num_iter: int) -> torch.Tensor:
    """
    Controls information capacity of the continuous and discrete loss
    (based on https://arxiv.org/pdf/1804.00104.pdf &
    https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py)
    """
    # Linearly increase capacity of continuous channels
    cont_min, cont_max, cont_num_iters, cont_gamma = cont_capacity
    # Increase continuous capacity without exceeding cont_max
    cont_cap_current = (cont_max - cont_min) * num_iter
    cont_cap_current = cont_cap_current / float(cont_num_iters) + cont_min
    cont_cap_current = min(cont_cap_current, cont_max)
    # Calculate continuous capacity loss
    cont_capacity_loss = cont_gamma*torch.abs(cont_cap_current - kl_cont_loss)

    # Linearly increase capacity of discrete channels
    disc_min, disc_max, disc_num_iters, disc_gamma = disc_capacity
    # Increase discrete capacity without exceeding disc_max or theoretical
    # maximum (i.e. sum of log of dimension of each discrete variable)
    disc_cap_current = (disc_max - disc_min) * num_iter
    disc_cap_current = disc_cap_current / float(disc_num_iters) + disc_min
    disc_cap_current = min(disc_cap_current, disc_max)
    # Require float conversion here to not end up with numpy float
    disc_theory_max = sum([float(np.log(d)) for d in disc_dims])
    disc_cap_current = min(disc_cap_current, disc_theory_max)
    # Calculate discrete capacity loss
    disc_capacity_loss = disc_gamma*torch.abs(disc_cap_current - kl_disc_loss)

    return cont_capacity_loss, disc_capacity_loss

