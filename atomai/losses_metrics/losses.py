"""
losses.py
=========

Custom Pytorch loss functions
"""
from typing import Tuple, List, Union
import numpy as np
import torch
import torch.nn.functional as F


class focal_loss(torch.nn.Module):
    """
    Loss function for classification tasks  with
    large data imbalance. Focal loss (FL) is define as:
    FL(p_t) = -alpha*((1-p_t)^gamma))*log(p_t),
    where p_t is a cross-entropy loss for binary classification.
    For more details, see https://arxiv.org/abs/1708.02002.

    Args:
        alpha (float):
            "balance" coefficient,
        gamma (float):
            "focusing" parameter (>=0),
        with_logits (bool):
            indicates if the sigmoid operation was applied
            at the end of a neural network's forward path.
    """
    def __init__(self, alpha: int = 0.5,
                 gamma: int = 2, with_logits: bool = True) -> None:
        """
        Parameter initialization
        """
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = with_logits

    def forward(self, prediction: torch.Tensor, labels: torch.Tensor):
        """
        Calculates loss
        """
        if self.logits:
            CE_loss = F.binary_cross_entropy_with_logits(prediction, labels)
        else:
            CE_loss = F.binary_cross_entropy(prediction, labels)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        return F_loss


class dice_loss(torch.nn.Module):
    """
    Computes the Sørensen–Dice loss.
    Adapted with changes from https://github.com/kevinzakka/pytorch-goodies
    """
    def __init__(self, eps: float = 1e-7):
        """
        Parameter initialization
        """
        super(dice_loss, self).__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[labels.squeeze(1).long()]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).contiguous().float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[labels.long()]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).contiguous().float()
            probas = F.softmax(logits, dim=1)

        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, labels.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)


def vae_loss(reconstr_loss: str,
             in_dim: Tuple[int],
             x: torch.Tensor,
             x_reconstr: torch.Tensor,
             *args: torch.Tensor
             ) -> torch.Tensor:
    """
    Calculates ELBO
    """
    batch_dim = x.size(0)
    if len(args) == 2:
        z_mean, z_logsd = args
    else:
        z_mean = z_logsd = torch.zeros((batch_dim, 1))
    z_sd = torch.exp(z_logsd)
    if reconstr_loss == "mse":
        reconstr_error = -0.5 * torch.sum(
            (x_reconstr.reshape(batch_dim, -1) - x.reshape(batch_dim, -1))**2, 1).mean()
    elif reconstr_loss == "ce":
        px_size = np.product(in_dim)
        rs = (np.product(in_dim[:2]),)
        if len(in_dim) == 3:
            rs = rs + (in_dim[-1],)
        reconstr_error = -F.binary_cross_entropy_with_logits(
            x_reconstr.reshape(-1, *rs), x.reshape(-1, *rs)) * px_size
    else:
        raise NotImplementedError("Reconstruction loss must be 'mse' or 'ce'")
    kl_z = -z_logsd + 0.5 * z_sd**2 + 0.5 * z_mean**2 - 0.5
    kl_z = torch.sum(kl_z, 1).mean()
    return reconstr_error - kl_z


def rvae_loss(reconstr_loss: str,
              in_dim: Tuple[int],
              x: torch.Tensor,
              x_reconstr: torch.Tensor,
              *args: torch.Tensor,
              **kwargs: float) -> torch.Tensor:
    """
    Calculates ELBO
    """
    batch_dim = x.size(0)
    if len(args) == 2:
        z_mean, z_logsd = args
    else:
        z_mean = z_logsd = torch.zeros((batch_dim, 1))
    phi_prior = kwargs.get("phi_prior", 0.1)
    b1, b2 = kwargs.get("b1", 1), kwargs.get("b2", 1)
    z_sd = torch.exp(z_logsd)
    phi_sd, phi_logsd = z_sd[:, 0], z_logsd[:, 0]
    z_mean, z_sd, z_logsd = z_mean[:, 1:], z_sd[:, 1:], z_logsd[:, 1:]
    batch_dim = x.size(0)
    if reconstr_loss == "mse":
        reconstr_error = -0.5 * torch.sum(
            (x_reconstr.view(batch_dim, -1) - x.view(batch_dim, -1))**2, 1).mean()
    elif reconstr_loss == "ce":
        px_size = np.product(in_dim)
        rs = (np.product(in_dim[:2]),)
        if len(in_dim) == 3:
            rs = rs + (in_dim[-1],)
        reconstr_error = -F.binary_cross_entropy_with_logits(
            x_reconstr.view(-1, *rs), x.view(-1, *rs)) * px_size
    else:
        raise NotImplementedError("Reconstruction loss must be 'mse' or 'ce'")
    kl_rot = (-phi_logsd + np.log(phi_prior) +
              phi_sd**2 / (2 * phi_prior**2) - 0.5)
    kl_z = -z_logsd + 0.5 * z_sd**2 + 0.5 * z_mean**2 - 0.5
    kl_div = (b1 * torch.sum(kl_z, 1) + b2 * kl_rot).mean()
    return reconstr_error - kl_div


def joint_vae_loss(reconstr_loss: str,
                   in_dim: Tuple[int],
                   x: torch.Tensor,
                   x_reconstr: torch.Tensor,
                   *args: torch.Tensor,
                   **kwargs: Union[List, int],
                   ) -> torch.Tensor:
    """
    Calculates joint ELBO for continuous and discrete variables
    """
    batch_dim = x.size(0)
    if len(args) == 3:
        z_mean, z_logsd, alphas = args
    else:
        z_mean = z_logsd = torch.zeros((batch_dim, 1))
        alphas = [torch.zeros((batch_dim, 1))]

    cont_capacity = kwargs.get("cont_capacity", [0.0, 5.0, 25000, 30])
    disc_capacity = kwargs.get("disc_capacity", [0.0, 5.0, 25000, 30])
    num_iter = kwargs.get("num_iter", 0)
    disc_dims = [a.size(1) for a in alphas]

    # Calculate reconstruction loss
    if reconstr_loss == "mse":
        reconstr_error = -0.5 * torch.sum(
            (x_reconstr.reshape(batch_dim, -1) - x.reshape(batch_dim, -1))**2, 1).mean()
    elif reconstr_loss == "ce":
        px_size = np.product(in_dim)
        rs = (np.product(in_dim[:2]),)
        if len(in_dim) == 3:
            rs = rs + (in_dim[-1],)
        reconstr_error = -F.binary_cross_entropy_with_logits(
            x_reconstr.reshape(-1, *rs), x.reshape(-1, *rs)) * px_size
    else:
        raise NotImplementedError("Reconstruction loss must be 'mse' or 'ce'")

    # Calculate KL term for continuous latent variables
    z_sd = torch.exp(z_logsd)
    kl_z = -z_logsd + 0.5 * z_sd**2 + 0.5 * z_mean**2 - 0.5
    kl_cont_loss = torch.sum(kl_z, 1).mean()
    # Calculate KL term for discrete latent variables
    kl_disc = [kl_discrete_loss(alpha) for alpha in alphas]
    kl_disc_loss = torch.sum(torch.cat(kl_disc))

    # Add information capacity terms
    # (based on https://arxiv.org/pdf/1804.00104.pdf &
    # https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py)
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

    return reconstr_error - cont_capacity_loss - disc_capacity_loss


def joint_rvae_loss(reconstr_loss: str,
                    in_dim: Tuple[int],
                    x: torch.Tensor,
                    x_reconstr: torch.Tensor,
                    *args: torch.Tensor,
                    **kwargs: float) -> torch.Tensor:
    """
    Calculates joint ELBO for continuous and discrete variables
    """
    batch_dim = x.size(0)
    if len(args) == 3:
        z_mean, z_logsd, alphas = args
    else:
        z_mean = z_logsd = torch.zeros((batch_dim, 1))
        alphas = [torch.zeros((batch_dim, 1))]

    phi_prior = kwargs.get("phi_prior", 0.1)
    b1, b2 = kwargs.get("b1", 1), kwargs.get("b2", 1)
    cont_capacity = kwargs.get("cont_capacity", [0.0, 5.0, 25000, 30])
    disc_capacity = kwargs.get("disc_capacity", [0.0, 5.0, 25000, 30])
    num_iter = kwargs.get("num_iter", 0)

    if reconstr_loss == "mse":
        reconstr_error = -0.5 * torch.sum(
            (x_reconstr.view(batch_dim, -1) - x.view(batch_dim, -1))**2, 1).mean()
    elif reconstr_loss == "ce":
        px_size = np.product(in_dim)
        rs = (np.product(in_dim[:2]),)
        if len(in_dim) == 3:
            rs = rs + (in_dim[-1],)
        reconstr_error = -F.binary_cross_entropy_with_logits(
            x_reconstr.view(-1, *rs), x.view(-1, *rs)) * px_size
    else:
        raise NotImplementedError("Reconstruction loss must be 'mse' or 'ce'")

    # Calculate KL term for continuous latent variables
    z_sd = torch.exp(z_logsd)
    phi_sd, phi_logsd = z_sd[:, 0], z_logsd[:, 0]
    z_mean, z_sd, z_logsd = z_mean[:, 1:], z_sd[:, 1:], z_logsd[:, 1:]
    kl_rot = (-phi_logsd + np.log(phi_prior) +
              phi_sd**2 / (2 * phi_prior**2) - 0.5)
    kl_z = -z_logsd + 0.5 * z_sd**2 + 0.5 * z_mean**2 - 0.5
    kl_cont_loss = (b1 * torch.sum(kl_z, 1) + b2 * kl_rot).mean()

    # Calculate KL term for discrete latent variables
    disc_dims = [a.size(1) for a in alphas]
    kl_disc = [kl_discrete_loss(alpha) for alpha in alphas]
    kl_disc_loss = torch.sum(torch.cat(kl_disc))

    # Add information capacity terms
    # (based on https://arxiv.org/pdf/1804.00104.pdf &
    # https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py)
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

    return reconstr_error - cont_capacity_loss - disc_capacity_loss


def kl_discrete_loss(alpha: torch.Tensor):
    """
    Calculates the KL divergence between a categorical distribution and a
    uniform categorical distribution.
    (based on https://arxiv.org/pdf/1611.01144, https://arxiv.org/pdf/1804.00104 &
    https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py)

    Args:
        alpha:
            Parameters of the categorical or Gumbel-Softmax distribution.
    """
    eps = 1e-12
    disc_dim = int(alpha.size()[-1])
    log_dim = torch.Tensor([np.log(disc_dim)])
    if torch.cuda.is_available():
        log_dim = log_dim.cuda()
    # Calculate negative entropy of each row
    neg_entropy = torch.sum(alpha * torch.log(alpha + eps), dim=1)
    # Take mean of negative entropy across batch
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)
    # KL loss of alpha with uniform categorical variable
    return log_dim + mean_neg_entropy


def select_loss(loss: str, nb_classes: int = None):
    """
    Selects loss for a semantic segmentation model training
    """
    if loss == 'ce' and nb_classes is None:
        raise ValueError("For cross-entropy loss function, you must" +
                         " specify the number of classes")
    if loss == 'dice':
        criterion = dice_loss()
    elif loss == 'focal':
        criterion = focal_loss()
    elif loss == 'ce' and nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif loss == 'ce' and nb_classes > 2:
        criterion = torch.nn.CrossEntropyLoss()
    elif loss == 'mse':
        criterion = torch.nn.MSELoss()
    elif hasattr(loss, "__call__"):
        criterion = loss
    else:
        raise NotImplementedError(
            "Select Dice loss ('dice'), focal loss ('focal') "
            " cross-entropy loss ('ce') or means-squared error ('mse')"
            " or pass your custom loss function"
        )
    return criterion
