"""
losses.py
=========

Custom Pytorch loss functions
"""
from typing import List, Type, Optional

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


class MultiTaskLoss(torch.nn.Module):
    """
    Multi-task loss for handling loss computation in multi-task learning.

    Args:
        num_tasks (int): The number of tasks.
        loss_fn (Type[nn.Module], optional): The loss function class to use for each task. Default is nn.NLLLoss.
        weights (List[float], optional): The weights for each task's loss. Default is None, which sets equal weights for all tasks.
    """
    def __init__(self,
                 num_tasks: int,
                 loss_fn: Type[torch.nn.Module] = torch.nn.NLLLoss,
                 weights: Optional[List[float]] = None):
        super(MultiTaskLoss, self).__init__()

        # Create a list of loss functions for each task
        self.loss_functions = [loss_fn() for _ in range(num_tasks)]

        # Set the weights for each task
        if weights is not None:
            assert len(weights) == num_tasks, "The length of weights must match num_tasks"
            self.weights = weights
        else:
            self.weights = [1.0] * num_tasks

    def forward(self, outputs: List[torch.Tensor], labels: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the combined loss for multiple tasks.

        Args:
            outputs (List[torch.Tensor]): A list of output tensors from the multi-task model, one for each task.
            labels (List[torch.Tensor]): A list of ground truth label tensors, one for each task.

        Returns:
            torch.Tensor: The total combined loss.
        """
        # Calculate the individual losses for each task
        individual_losses = [
            weight * loss_fn(output, label)
            for weight, loss_fn, output, label in zip(self.weights, self.loss_functions, outputs, labels)
        ]

        # Combine the individual losses to get the total loss
        total_loss = sum(individual_losses)
        return total_loss


def select_loss(loss: str, nb_classes: int = None, **kwargs):
    """
    Selects loss for DCNN model training
    """
    if loss in ['ce', 'multitask'] and nb_classes is None:
        raise ValueError("For cross-entropy loss function, you must" +
                         " specify the number of classes")
    if loss == 'multitask' and not isinstance(nb_classes, list):
        raise ValueError("Provide number of classes for each task as a list")
    if loss == 'dice':
        criterion = dice_loss()
    elif loss == 'focal':
        criterion = focal_loss()
    elif loss == 'ce' and nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif loss == 'ce' and nb_classes > 2:
        criterion = torch.nn.CrossEntropyLoss()
    elif loss == 'nll':
        criterion = torch.nn.NLLLoss()
    elif loss == 'multitask_nll':
        criterion == MultiTaskLoss(len(nb_classes), **kwargs)
    elif loss == 'multitask_ce':
        criterion == MultiTaskLoss(
            len(nb_classes), torch.nn.CrossEntropyLoss, **kwargs)
    elif loss == 'mse':
        criterion = torch.nn.MSELoss()
    elif hasattr(loss, "__call__"):
        criterion = loss
    else:
        raise NotImplementedError(
            "Select Dice loss ('dice'), focal loss ('focal') "
            " cross-entropy loss ('ce'), means-squared error ('mse'),"
            " multitask loss (multitask_nll and multitask_ce)"
            " or pass your custom loss function"
        )
    return criterion
