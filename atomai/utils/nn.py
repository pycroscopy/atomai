"""
nn.py
=====

Utility functions for working with NN weights and classes.
Includes utility function for monitoring GPU usage during NN training.

Created by: Maxim Ziatdinov (maxim.ziatdinov@ai4microscopy.com)
"""

import copy
import subprocess
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import (BatchNorm1d, BatchNorm2d, Conv1d, Conv2d,
                      ConvTranspose1d, ConvTranspose2d, Linear)

dc = copy.deepcopy


def load_weights(model: Type[torch.nn.Module],
                 weights_path: str) -> Type[torch.nn.Module]:
    """
    Loads weights saved as pytorch state dictionary into a model skeleton

    Args:
        model (pytorch object):
            Initialized pytorch model
        weights_path (str):
            Filepath to trained weights (pytorch state dict)

    Returns:
        Model with trained weights loaded in evaluation state

    Example:

        >>> from atomai.utils import load_weights
        >>> # Path to file with trained weights
        >>> weights_path = '/content/simple_model_weights.pt'
        >>> # Initialize model (by default all trained models are 'dilUnet')
        >>> # You can also use nb_classes=utils.nb_filters_classes(weights_path)[1]
        >>> model = models.dilUnet(nb_classes=3)
        >>> # Load the weights into the model skeleton
        >>> model = load_weights(model, weights_path)
    """
    torch.manual_seed(0)
    if torch.cuda.device_count() > 0:
        checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model.eval()


def average_weights(ensemble: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Averages weights of all models in the ensemble

    Args:
        ensemble (dict):
            Dictionary with trained weights (model's state_dict)
            of models with exact same architecture.

    Returns:
        Averaged weights (as model's state_dict)
    """
    ensemble_state_dict = dc(ensemble[0])
    names = [name for name in ensemble_state_dict.keys() if
             name.split('_')[-1] not in ["mean", "var", "tracked"]]
    for name in names:
        w_aver = []
        for model in ensemble.values():
            for n, p in model.items():
                if n == name:
                    w_aver.append(dc(p))
        ensemble_state_dict[name].copy_(sum(w_aver) / float(len(w_aver)))
    return ensemble_state_dict


def sample_weights(ensemble: Dict[int, Dict[str, torch.Tensor]],
                   n_samples: int = 30) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Calculate the mean and standard deviation for each trainable parameter
    and use them to draw samples for each parameter independently according to
    :math:`\\theta_i \\sim N(\\mu_i, \\sigma_i)`

    Args:
        ensemble (dict):
            Dictionary with trained weights of models
            with the exact same architecture

    Returns:
        Updated dictionary with sampled weights
    """
    ensemble_ = {i: dc(ensemble[0])
                 for i in range(n_samples)}
    ensemble_state_dict = ensemble[0]
    names = [name for name in ensemble_state_dict.keys() if
             name.split('_')[-1] not in ["mean", "var", "tracked"]]
    for name in names:
        w_all = []
        for model in ensemble.values():
            for n, p in model.items():
                if n == name:
                    w_all.append(dc(p)[None, ...])
        w_all = torch.cat(w_all, dim=0)
        if w_all.dtype == torch.float32:
            w_all_mu = torch.mean(w_all, axis=0)
            w_all_std = torch.std(w_all, axis=0)
            ndist = torch.distributions.Normal(w_all_mu, w_all_std)
            for i in range(n_samples):
                ensemble_[i][name].copy_(ndist.sample())
    return ensemble_


def gpu_usage_map(cuda_device: int) -> int:
    """
    Get the current GPU memory usage
    Adapted with changes from
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--id=' + str(cuda_device),
            '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_usage = [int(y) for y in result.split(',')]
    return gpu_usage[0:2]


def set_train_rng(seed: int = 1):
    """
    For reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Hook():
    """
    Returns the input and output of a
    layer during forward/backward pass
    see https://www.kaggle.com/sironghuang/
        understanding-pytorch-hooks/notebook

    Args:
        module (torch module): single layer or sequential block
        backward (bool): replace forward_hook with backward_hook
    """
    def __init__(self, module: Type[torch.nn.Module], backward: bool = False) -> None:
        if backward is False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module: Type[torch.nn.Module],
                input_: Tuple[torch.Tensor], output_: torch.Tensor) -> None:
        self.input = input_
        self.output = output_

    def close(self) -> None:
        self.hook.remove()


def mock_forward(model: Type[torch.nn.Module],
                 dims: Tuple[int] = (1, 64, 64)) -> torch.Tensor:
    """
    Passes a dummy variable throuh a network
    """

    x = torch.randn(1, *dims)
    if next(model.parameters()).is_cuda:
        x = x.cuda()
    out = model(x)
    return out


def get_nb_classes(model: Type[torch.nn.Module]) -> int:
    """
    Gets number of classes used in a fully convolutional NN
    """
    hookF = [Hook(layer[1]) for layer in list(model._modules.items())]
    mock_forward(model)
    nb_classes = [hook.output.shape for hook in hookF][-1][1]
    return nb_classes


def get_downsample_factor(model: Type[torch.nn.Module]) -> int:
    """
    Gets a downsample factor for UNet-like architectures
    """
    hookF = [Hook(layer[1]) for layer in list(model._modules.items())]
    mock_forward(model)
    imsize = [hook.output.shape[-1] for hook in hookF]
    return max(imsize) / min(imsize)


def dummy_optimizer() -> Type[torch.optim.Optimizer]:
    """
    Returns initialized "dummy" optimizer
    """
    return torch.optim.Optimizer([torch.zeros(1)], dict())


def weights_init(module):
    imodules = (Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d, Linear)
    if isinstance(module, imodules):
        torch.nn.init.xavier_uniform_(module.weight.data)
        torch.nn.init.zeros_(module.bias)


def reset_bnorm(module):
    imodules = (BatchNorm1d, BatchNorm2d)
    if isinstance(module, imodules):
        module.reset_running_stats()
        module.reset_parameters()


def nb_filters_classes(weights_path: str) -> Tuple[int]:
    """
    Inferes the number of filters and the number of classes
    used in trained AtomAI models from the loaded weights.

    Args:
        weight_path (str):
            Path to file with saved weights (.pt extension)

    """
    checkpoint = torch.load(weights_path, map_location='cpu')
    tensor_shapes = [v.shape for v in checkpoint.values() if len(v.shape) > 1]
    nb_classes = tensor_shapes[-1][0]
    nb_filters = tensor_shapes[0][0]
    return nb_filters, nb_classes


def combine_classes(coord_class_dict: Dict[int, np.ndarray],
                    classes_to_combine: List[int],
                    renumerate: bool = True) -> Dict[int, np.ndarray]:
    """
    Combines classes in a dictionary from atomnet.locator or atomnet.predictor outputs
    """
    coord_class_dict_ = dc(coord_class_dict)
    for i in range(len(coord_class_dict_)):
        coord_class_dict_[i][:, -1] = combine_classes_(
            coord_class_dict_[i][:, -1], classes_to_combine)
    if renumerate:
        coord_class_dict_ = renumerate_classes(coord_class_dict_)
    return coord_class_dict_


def combine_classes_(classes_all: np.ndarray,
                     classes_to_combine: List[int]) -> np.ndarray:
    """
    Given a list of classes to combine substitutes listed classes
    with a minimum value from the list
    """
    for comb in classes_to_combine:
        cls_min = min(comb)
        for c in comb:
            classes_all[classes_all == c] = cls_min
    return classes_all


def renumerate_classes_(classes: np.ndarray,
                        start_from_1: bool = True) -> np.ndarray:
    """
    Renumerate classes such that they are ordered starting from 1 or 0
    with an increment of 1
    """
    diff = np.unique(classes) - np.arange(len(np.unique(classes)))
    diff_d = {cl: d for d, cl in zip(diff, np.unique(classes))}
    classes_renum = [cl - diff_d[cl] for cl in classes]
    classes_renum = np.array(classes_renum, dtype=np.float)
    if start_from_1:
        classes_renum = classes_renum + 1
    return classes_renum


def renumerate_classes(coord_class_dict: Dict[int, np.ndarray],
                       start_from_1: bool = True) -> Dict[int, np.ndarray]:
    """
    Renumerate classes in a dictionary from atomnet.locator or atomnet.predictor output
    such that they are ordered starting from 1 or 0 with an increment of 1
    """
    coord_class_dict_ = dc(coord_class_dict)
    for i in range(len(coord_class_dict)):
        coord_class_dict_[i][:, -1] = renumerate_classes_(
            coord_class_dict_[i][:, -1], start_from_1=True)
    return coord_class_dict_
