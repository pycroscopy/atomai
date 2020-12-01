from copy import deepcopy as dc
from typing import Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ..losses_metrics import IoU
from ..nets import init_fcnn_model, init_imspec_model
from ..utils import (average_weights, check_image_dims, check_signal_dims,
                     init_fcnn_dataloaders, init_imspec_dataloaders,
                     num_classes_from_labels, preprocess_training_image_data,
                     preprocess_training_imspec_data, sample_weights)
from .trainer import BaseTrainer

augfn_type = Callable[[torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]


class BaseEnsembleTrainer(BaseTrainer):

    def __init__(self,
                 model: Type[torch.nn.Module] = None,
                 nb_classes=None
                 ) -> None:
        super(BaseEnsembleTrainer, self).__init__()

        if model is not None:
            self.set_model(model, nb_classes)
        self.ensemble_state_dict = {}

    def compile_ensemble_trainer(self,
                                 strategy: str = "from_scratch",
                                 **kwargs):

        self.strategy = strategy
        if self.strategy not in ["from_baseline", "from_scratch", "swag"]:
            raise NotImplementedError(
                "Select 'from_baseline' 'from_scratch', or 'swag'  strategy")
        self.kdict = kwargs

    def train_baseline(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_test: Optional[np.ndarray] = None,
                       y_test: Optional[np.ndarray] = None,
                       seed: int = 1,
                       augment_fn: augfn_type = None):

        if self.net is None:
            raise AssertionError("You need to set a model first")
        self._reset_rng(seed)
        self._reset_weights()
        self._reset_training_history()
        self.compile_trainer(
            (X_train, y_train, X_test, y_test), **self.kdict)
        self.data_augmentation(augment_fn)
        self.fit()
        return self.net

    def train_ensemble_from_scratch(self,
                                    X_train: np.ndarray,
                                    y_train: np.ndarray,
                                    X_test: Optional[np.ndarray] = None,
                                    y_test: Optional[np.ndarray] = None,
                                    n_models: int = 10,
                                    augment_fn: augfn_type = None,
                                    **kwargs):
        batch_seed = kwargs.get("batch_seed")
        print("Training ensemble models (trategy = 'from_scratch'")
        for i in range(n_models):
            print("Ensemble model {}".format(i + 1))
            if batch_seed is not None:
                self.kdict["batch_seed"] = i
            model_i = self.train_baseline(
                X_train, y_train, X_test, y_test, i, augment_fn)
            self.ensemble_state_dict[i] = dc(model_i.state_dict())
            self.save_ensemble_metadict()
        return self.net, self.ensemble_state_dict

    def train_ensemble_from_baseline(self,
                                     X_train: np.ndarray,
                                     y_train: np.ndarray,
                                     X_test: Optional[np.ndarray] = None,
                                     y_test: Optional[np.ndarray] = None,
                                     basemodel: Type[torch.nn.Module] = None,
                                     n_models: int = 10,
                                     training_cycles_base: int = 1000,
                                     training_cycles_ensemble: int = 100,
                                     augment_fn: augfn_type = None,
                                     **kwargs):

        if len(kwargs) != 0:
            for k, v in kwargs.items():
                self.kdict[k] = v

        if basemodel is None:
            self.kdict["training_cycles"] = training_cycles_base
            basemodel = self.train_baseline(
                X_train, y_train, X_test, y_test, 1, augment_fn)

        self.set_model(basemodel)
        basemodel_state_dict = dc(self.net.state_dict())

        self.kdict["training_cycles"] = training_cycles_ensemble
        if "print_loss" not in self.kdict.keys():
            self.kdict["print_loss"] = 10

        print("Training ensemble models (trategy = 'from_baseline'")
        for i in range(n_models):
            print("Ensemble model {}".format(i + 1))
            if i > 0:
                self.net.load_state_dict(basemodel_state_dict)
                self._reset_rng(i+2)
                self.compile_trainer(  # Note that here we reinitialize optimizer
                    (X_train, y_train, X_test, y_test), **self.kdict)
                model_i = self.fit()
                self.ensemble_state_dict[i] = dc(model_i.state_dict())
                self.save_ensemble_metadict()
            averaged_weights = average_weights(self.ensemble_state_dict)
            model_i.load_state_dict(averaged_weights)
        return model_i, self.ensemble_state_dict

    def train_swag(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test: Optional[np.ndarray] = None,
                   y_test: Optional[np.ndarray] = None,
                   n_models: int = 10,
                   augment_fn: augfn_type = None
                   ):
        """
        Performs SWAG-like weights sampling at the end of single model training
        """
        self.kdict["swa"] = True
        basemodel = self.train_baseline(
                X_train, y_train, X_test, y_test, 1, augment_fn)
        self.ensemble_state_dict = sample_weights(
            self.running_weights, n_models)
        self.save_ensemble_metadict()

        return basemodel, self.ensemble_state_dict

    def save_ensemble_metadict(self) -> None:
        """
        Saves meta dictionary with ensemble weights and key information about
        model's structure (needed to load it back) to disk
        """
        ensemble_metadict = dc(self.meta_state_dict)
        ensemble_metadict["weights"] = self.ensemble_state_dict
        torch.save(ensemble_metadict, self.filename + "_ensemble.tar")


class EnsembleTrainer(BaseEnsembleTrainer):

    def __init__(self,
                 model: Union[str, Type[torch.nn.Module]] = None,
                 nb_classes: int = 1,
                 **kwargs) -> None:
        super(EnsembleTrainer, self).__init__()

        self.nb_classes = nb_classes
        if isinstance(model, str):
            if model in ["dilUnet", "dilnet"]:
                self.net, self.meta_state_dict = init_fcnn_model(
                    model, self.nb_classes, **kwargs)
                self.accuracy_fn = accuracy_fn_seg(nb_classes)
            elif model == "imspec":
                self.in_dim = kwargs.get("in_dim")
                self.out_dim = kwargs.get("out_dim")
                latent_dim = kwargs.get("latent_dim")
                if None in (self.in_dim, self.out_dim, latent_dim):
                    raise AssertionError(
                        "Specify input/output and latent dimensions " +
                        "(in_dim, out_dim, latent_dim)")
                self.net, self.meta_state_dict = init_imspec_model(
                    self.in_dim, self.out_dim, latent_dim, **kwargs)
            self.net.to(self.device)
        else:
            self.set_model(model, nb_classes)

        self.meta_state_dict["weights"] = self.net.state_dict()
        self.meta_state_dict["optimizer"] = self.optimizer

    def compile_ensemble_trainer(self,
                                 strategy: str = "from_scratch",
                                 **kwargs):

        self.strategy = strategy
        if self.strategy not in ["from_baseline", "from_scratch", "swag"]:
            raise NotImplementedError(
                "Select 'from_baseline' 'from_scratch', or 'swag'  strategy")
        self.kdict = kwargs
        self.full_epoch = self.kdict.get("full_epoch", False)
        self.batch_size = self.kdict.get("batch_size", 32)
        self.kdict["overwrite_train_data"] = False

    def train_baseline(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_test: Optional[np.ndarray] = None,
                       y_test: Optional[np.ndarray] = None,
                       seed: int = 1,
                       augment_fn: augfn_type = None):

        if self.net is None:
            raise AssertionError("You need to set a model first")

        if self.meta_state_dict.get("model_type") == "seg":
            train_data = set_data_seg(
                X_train, y_train, X_test, y_test,
                self.nb_classes)
        elif self.meta_state_dict.get("model_type") == "imspec":
            train_data = set_data_imspec(
                X_train, y_train, X_test, y_test,
                (self.in_dim, self.out_dim))
        self.set_data(*train_data)

        self._reset_rng(seed)
        self._reset_weights()
        self._reset_training_history()

        self.compile_trainer(
            (X_train, y_train, X_test, y_test), **self.kdict)
        self.data_augmentation(augment_fn)
        self.fit()

        return self.net


def set_data_seg(X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_test: Optional[np.ndarray] = None,
                 y_test: Optional[np.ndarray] = None,
                 nb_classes_set: int = 1,
                 **kwargs: Union[float, int]
                 ) -> Tuple[np.ndarray]:
    """
    Sets training and test data for semantic segmentation
    """
    nb_classes = num_classes_from_labels(y_train)
    if nb_classes != nb_classes_set:
        raise AssertionError("Number of specified classes" +
                             " is different from the number of classes" +
                             " contained in training data")

    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=kwargs.get("test_size", .15),
            shuffle=True, random_state=kwargs.get("seed", 1))

    X_train, y_train, X_test, y_test = check_image_dims(
        X_train, y_train, X_test, y_test, nb_classes)

    f32, i64 = lambda x: x.astype(np.float32), lambda x: x.astype(np.int64)
    X_train, X_test = f32(X_train), f32(X_test)
    if nb_classes > 1:
        y_train, y_test = i64(y_train), i64(y_test)
    else:
        y_train, y_test = f32(y_train), f32(y_test)

    return X_train, y_train, X_test, y_test


def set_data_imspec(X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_test: Optional[np.ndarray] = None,
                    y_test: Optional[np.ndarray] = None,
                    dims: Tuple[int] = None,
                    **kwargs: Union[float, int]
                    ) -> Tuple[np.ndarray]:
    """
    Sets training and test data for im2spec and spec2im models
    """

    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=kwargs.get("test_size", .15),
            shuffle=True, random_state=kwargs.get("seed", 1))

    X_train, y_train, X_test, y_test = check_signal_dims(
        X_train, y_train, X_test, y_test)

    in_dim, out_dim = X_train.shape[2:], y_train.shape[2:]
    if dims[0] != in_dim or dims[1] != out_dim:
        raise AssertionError(
            "The input/output dimensions of the model must match" +
            " the height, width and length (for spectra) of training")

    f32 = lambda x: x.astype(np.float32)
    X_train, X_test = f32(X_train), f32(X_test)
    y_train, y_test = f32(y_train), f32(y_test)

    return X_train, y_train, X_test, y_test


def accuracy_fn_seg(nb_classes: int
                    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Returns function that computes IoU score
    """
    def accuracy(y, y_prob, *args):
        iou_score = IoU(
                y, y_prob, nb_classes).evaluate()
        return iou_score
    return accuracy
