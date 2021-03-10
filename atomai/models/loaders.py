"""
loaders.py
==========

Helpfer functions for loading pre-trained AtomAI's models

Created by Maxim Ziatdinov (maxim.ziatdinov@ai4microscopy.com)

"""
import warnings
from copy import deepcopy as dc
from typing import Type, Tuple, Dict, Union

import torch
from .segmentor import Segmentor
from .imspec import ImSpec
from .dgm import BaseVAE, VAE, rVAE, jrVAE, jVAE
from ..utils import average_weights


def load_model(filepath: str) -> Union[Segmentor, Union[VAE, rVAE, jrVAE, jVAE], ImSpec]:
    """
    Loads trained AtomAI models

    Args:
        meta_state_dict (str):
            filepath to meta-state dictionary with trained weights
            and information about model's structure

    Returns:
        Model in evaluation state
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaded_dict = torch.load(filepath, map_location=device)
    if 'model_type' in loaded_dict.keys():
        model_type = loaded_dict.pop("model_type")
        if model_type == "seg":
            model = load_seg_model(loaded_dict)
        elif model_type == "imspec":
            model = load_imspec_model(loaded_dict)
        elif model_type == "vae":
            model = load_vae_model(loaded_dict)
        else:
            raise ValueError(
                "The model type {} cannot be loaded".format(model_type))
    else:
        model = loaded_dict["weights"]
        warnings.warn("Returning model's state dictionary." +
                      "You will need to load it into your model's" +
                      " skeleton by yourself",
                      UserWarning)
    return model


def load_seg_model(meta_dict: Dict[str, torch.Tensor]) -> Type[Segmentor]:
    """
    Loads trained AtomAI semantic segmentation models

    Args:
        meta_dict (str):
            dictionary with trained weights and key information
            about model's structure

    Returns:
        Segmentor object with NN in evaluation state
    """
    model_name = meta_dict.pop("model")
    nb_classes = meta_dict.pop("nb_classes")
    weights = meta_dict.pop("weights")
    model = Segmentor(model_name, nb_classes, **meta_dict)
    model.net.load_state_dict(weights)
    if "optimizer" in meta_dict.keys():
        optimizer = meta_dict.pop("optimizer")
        model.optimizer = optimizer
    model.net.eval()
    return model


def load_imspec_model(meta_dict: Dict[str, torch.Tensor]) -> Type[ImSpec]:
    """
    Loads trained AtomAI ImSpec models

    Args:
        meta_dict (str):
            dictionary with trained weights and key information
            about model's structure

    Returns:
        ImSpec object with NN in evaluation state
    """
    in_dim = meta_dict.pop("in_dim")
    out_dim = meta_dict.pop("out_dim")
    z_dim = meta_dict.pop("latent_dim")
    weights = meta_dict.pop("weights")
    optimizer = meta_dict.pop("optimizer")
    model = ImSpec(in_dim, out_dim, z_dim, **meta_dict)
    model.net.load_state_dict(weights)
    model.optimizer = optimizer
    model.net.eval()
    return model


def load_vae_model(meta_dict: Dict[str, torch.Tensor]) -> Type[BaseVAE]:
    """
    Loads trained AtomAI ImSpec models

    Args:
        meta_dict (str):
            dictionary with trained weights and key information
            about model's structure

    Returns:
        BaseVAE object with encoder and ecoder nets in evaluation state
    """
    in_dim = meta_dict.pop("in_dim")
    latent_dim = meta_dict.pop("latent_dim")
    encoder_weights = meta_dict.pop("encoder")
    decoder_weights = meta_dict.pop("decoder")
    coord = meta_dict.pop("coord")
    optimizer = meta_dict.pop("optimizer")
    if coord:
        translate = True if coord == 3 else False
        model = jrVAE if meta_dict["discrete_dim"] else rVAE
        m = model(in_dim, latent_dim, translation=translate, **meta_dict)
    else:
        model = jVAE if meta_dict["discrete_dim"] else VAE
        m = model(in_dim, latent_dim, **meta_dict)
    if meta_dict["discrete_dim"]:
        m.kdict_["num_iter"] = meta_dict.get("num_iter", 0)
    m.encoder_net.load_state_dict(encoder_weights)
    m.encoder_net.eval()
    m.decoder_net.load_state_dict(decoder_weights)
    m.decoder_net.eval()
    m.optim = optimizer
    return m


def load_ensemble(filepath: str) -> Tuple[Type[torch.nn.Module], Dict[int, Dict[str, torch.Tensor]]]:
    """
    Loads trained ensemble models

    Args:
        meta_state_dict (str):
            filepath to dictionary with trained weights and key information
            about model's structure
    Returns:
        Single model with averaged weights and dictionary with weights of all models
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaded_dict = torch.load(filepath, map_location=device)
    if 'model_type' in loaded_dict.keys():
        model_type = loaded_dict.pop("model_type")
        ensemble_weights = dc(loaded_dict["weights"])
        loaded_dict["weights"] = average_weights(loaded_dict["weights"])
        if model_type == "seg":
            smodel = load_seg_model(loaded_dict)
        elif model_type == "imspec":
            smodel = load_imspec_model(loaded_dict)
        elif model_type == "vae":
            smodel = load_vae_model(loaded_dict)
        else:
            raise ValueError(
                "The model type {} cannot be loaded".format(model_type))
    else:
        warnings.warn("Returning dictionary with ensemble weights" +
                      "You will need to load them into your model's" +
                      "skeleton by yourself")
        return None, ensemble_weights
    return smodel.net, ensemble_weights
