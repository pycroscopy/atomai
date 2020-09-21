"""
loaders.py
==========

Helpfer functions for loading pre-trained AtomAI's models

Created by Maxim Ziatdinov (maxim.ziatdinov@ai4microscopy.com)

"""
from typing import Type, Tuple, Dict

import torch
from .fcnn import dilnet, dilUnet
from ..utils import average_weights


def load_model(meta_state_dict: str) -> Type[torch.nn.Module]:
    """
    Loads trained AtomAI models

    Args:
        meta_state_dict (str):
            filepath to dictionary with trained weights and key information
            about model's structure (stored during and after model training
            with atomnet.trainer)

    Returns:
        Model in evaluation state
    """
    torch.manual_seed(0)
    if torch.cuda.device_count() > 0:
        meta_dict = torch.load(meta_state_dict)
    else:
        meta_dict = torch.load(meta_state_dict, map_location='cpu')
    if "with_dilation" in meta_dict.keys():
        (model_type, batchnorm, dropout, upsampling,
         nb_filters, layers, nb_classes, checkpoint,
         with_dilation) = meta_dict.values()
    else:
        (model_type, batchnorm, dropout, upsampling,
         nb_filters, layers, nb_classes, checkpoint) = meta_dict.values()
    if model_type == 'dilUnet':
        model = dilUnet(
            nb_classes, nb_filters, dropout,
            batchnorm, upsampling, with_dilation,
            layers=layers)
    elif model_type == 'dilnet':
        model = dilnet(
            nb_classes, nb_filters, dropout,
            batchnorm, upsampling, layers=layers)
    else:
        raise NotImplementedError(
            "Select between 'dilUnet' and 'dilnet' neural networks"
        )
    model.load_state_dict(checkpoint)
    return model.eval()


def load_ensemble(meta_state_dict: str) -> Tuple[Type[torch.nn.Module], Dict[int, Dict[str, torch.Tensor]]]:
    """
    Loads trained ensemble models
    
    Args:
        meta_state_dict (str):
            filepath to dictionary with trained weights and key information
            about model's structure

    Returns:
        Single model with averaged weights and dictionary with weights of all models
    """
    torch.manual_seed(0)
    if torch.cuda.device_count() > 0:
        meta_dict = torch.load(meta_state_dict)
    else:
        meta_dict = torch.load(meta_state_dict, map_location='cpu')
    if "with_dilation" in meta_dict.keys():
        (model_type, batchnorm, dropout, upsampling,
         nb_filters, layers, nb_classes, checkpoint,
         with_dilation) = meta_dict.values()
    else:
        (model_type, batchnorm, dropout, upsampling,
         nb_filters, layers, nb_classes, checkpoint) = meta_dict.values()
    if model_type == 'dilUnet':
        model = dilUnet(
            nb_classes, nb_filters, dropout,
            batchnorm, upsampling, with_dilation,
            layers=layers)
    elif model_type == 'dilnet':
        model = dilnet(
            nb_classes, nb_filters, dropout,
            batchnorm, upsampling, layers=layers)
    else:
        raise NotImplementedError(
            "The network must be either 'dilUnet' or 'dilnet'"
        )
    model.load_state_dict(average_weights(checkpoint))
    return model.eval(), checkpoint
