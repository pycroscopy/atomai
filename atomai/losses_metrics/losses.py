"""
losses.py
=========

Custom Pytorch loss functions
"""
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


def select_loss(loss: str, nb_classes: int = None):
    """
    Selects loss for DCNN model training
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
