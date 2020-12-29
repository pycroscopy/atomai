"""
etrainer.py
===========

Module for deeep ensemble training of neural networks

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from copy import deepcopy as dc
from typing import Callable, Dict, Optional, Tuple, Type, Union
import warnings

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ..losses_metrics import IoU
from ..nets import init_fcnn_model, init_imspec_model
from ..utils import (average_weights, check_image_dims, check_signal_dims,
                     num_classes_from_labels, sample_weights)
from .trainer import BaseTrainer

augfn_type = Callable[[torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
compile_kwargs_type = Union[Type[torch.optim.Optimizer], str, int, bool]
ensemble_type = Dict[int, Dict[str, torch.Tensor]]


class BaseEnsembleTrainer(BaseTrainer):
    """
    Base class for deep ensemble training
    """
    def __init__(self,
                 model: Type[torch.nn.Module] = None,
                 nb_classes=None
                 ) -> None:
        """
        Initialize base ensemble trainer
        """
        super(BaseEnsembleTrainer, self).__init__()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if model is not None:
            self.set_model(model, nb_classes)
        self.ensemble_state_dict = {}

    def compile_ensemble_trainer(self,
                                 **kwargs: compile_kwargs_type
                                 ) -> None:
        """
        Compile ensemble trainer.

        Args:
            kwargs:
                Keyword arguments to be passed to BaseTrainer.compile_trainer
                (loss, optimizer, compute_accuracy, full_epoch, swa,
                perturb_weights, batch_size, training_cycles, accuracy_metrics,
                filename, print_loss, plot_training_history)
        """
        self.kdict = kwargs

    def train_baseline(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_test: Optional[np.ndarray] = None,
                       y_test: Optional[np.ndarray] = None,
                       seed: int = 1,
                       augment_fn: augfn_type = None
                       ) -> Type[torch.nn.Module]:
        """
        Trains baseline weights

        Args:
            X_train:
                Training features
            y_train:
                Training targets
            X_test:
                Test features
            y_test:
                Test targets
            seed:
                seed to be used for pytorch and numpy random numbers generator
            augment_fn:
                function that takes two torch tensors (features and targets),
                peforms some transforms, and returns the transformed tensors.
                The dimensions of the transformed tensors must be the same as
                the dimensions of the original ones.

        Returns:
            Trained baseline model
        """
        if self.net is None:
            raise AssertionError("You need to set a model first")
        self._reset_rng(seed)
        self._reset_weights()
        self._reset_training_history()
        (X_train, y_train,
         X_test, y_test) = self.preprocess_train_data(
            X_train, y_train, X_test, y_test)

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
                                    **kwargs
                                    ) -> Tuple[Type[torch.nn.Module], ensemble_type]:
        """
        Trains ensemble of models starting every time from scratch with
        different initialization

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            n_models: number of models to be trained
            augment_fn:
                function that takes two torch tensors (features and targets),
                peforms some transforms, and returns the transformed tensors.
                The dimensions of the transformed tensors must be the same as
                the dimensions of the original ones.
            **kwargs:
                Updates kwargs from initial compilation, which can be useful
                for iterative training.

        Returns:
            The last trained model and dictionary with ensemble weights
        """

        self.update_training_parameters(kwargs)

        print("Training ensemble models (strategy = 'from_scratch')")
        for i in range(n_models):
            print("\nEnsemble model {}".format(i + 1))
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
                                     **kwargs
                                     ) -> Tuple[Type[torch.nn.Module], ensemble_type]:
        """
        Trains ensemble of models starting each time from baseline model.
        Each ensemble model is trained each with different random shuffling
        of batches (and different seed for data augmentation if any).
        If a baseline model is not provided, the baseline weights are trained
        for *N* epochs and then used as a baseline to train multiple ensemble
        models for *n* epochs (*n* << *N*),

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            basemodel: Provide a baseline model (Optional)
            n_models: number of models in ensemble
            training_cycles_base:
                Number of training iterations for baseline model
            training_cycles_ensemble:
                Number of training iterations for every ensemble model
            augment_fn:
                function that takes two torch tensors (features and targets),
                peforms some transforms, and returns the transformed tensors.
                The dimensions of the transformed tensors must be the same as
                the dimensions of the original ones.
            **kwargs: Updates kwargs from initial compilation
                (can be useful for iterative training)

        Returns:
            Model with averaged weights and dictionary with ensemble weights
        """

        self.update_training_parameters(kwargs)

        if basemodel is None:
            self.kdict["training_cycles"] = training_cycles_base
            print("Training baseline model...")
            basemodel = self.train_baseline(
                X_train, y_train, X_test, y_test, 1, augment_fn)
        else:  # this is the only time when we do not use train_from_baseline
            (X_train, y_train,
             X_test, y_test) = self.preprocess_train_data(
                X_train, y_train, X_test, y_test)

        self.set_model(basemodel)
        basemodel_state_dict = dc(self.net.state_dict())

        self.kdict["training_cycles"] = training_cycles_ensemble
        if not self.full_epoch:
            if "print_loss" not in self.kdict.keys():
                self.kdict["print_loss"] = 10

        print("\nTraining ensemble models (strategy = 'from_baseline')")
        for i in range(n_models):
            print("\nEnsemble model {}".format(i + 1))
            if i > 0:
                self.net.load_state_dict(basemodel_state_dict)
            self._reset_rng(i+2)
            self._reset_training_history()
            self.compile_trainer(  # Note that here we reinitialize optimizer
                (X_train, y_train, X_test, y_test),
                batch_seed=i+2, **self.kdict)
            model_i = self.run()
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
                   augment_fn: augfn_type = None,
                   **kwargs: compile_kwargs_type
                   ) -> Tuple[Type[torch.nn.Module], ensemble_type]:
        """
        Performs SWAG-like weights sampling at the end of single model training

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            n_models: number fo samples to be drawn
            augment_fn:
                function that takes two torch tensors (features and targets),
                peforms some transforms, and returns the transformed tensors.
                The dimensions of the transformed tensors must be the same as
                the dimensions of the original ones.
            **kwargs: Updates kwargs from initial compilation
                (can be useful for iterative training)

        Returns:
            Baseline model and dictionary with sampled weights
        """
        self.update_training_parameters(kwargs)
        self.kdict["swa"] = True
        basemodel = self.train_baseline(
                X_train, y_train, X_test, y_test, 1, augment_fn)
        self.ensemble_state_dict = sample_weights(
            self.running_weights, n_models)
        self.save_ensemble_metadict()

        return basemodel, self.ensemble_state_dict

    def update_training_parameters(self, kwargs):
        warn_msg = "Overwriting the initial value '{}' of parameter '{}' with new value '{}'"
        if len(kwargs) != 0:
            for k, v in kwargs.items():
                if k in self.kdict.keys():
                    warnings.warn(
                        warn_msg.format(self.kdict[k], k, kwargs[k]),
                        UserWarning)
                self.kdict[k] = v

    def preprocess_train_data(self,
                              train_data: Tuple[np.ndarray]
                              ) -> Tuple[np.ndarray]:
        X, y, X_, y_ = train_data
        tor = lambda x: torch.from_numpy(x)
        return tor(X), tor(y), tor(X_), tor(y_)

    def save_ensemble_metadict(self) -> None:
        """
        Saves meta dictionary with ensemble weights and key information about
        model's structure (needed to load it back) to disk
        """
        ensemble_metadict = dc(self.meta_state_dict)
        ensemble_metadict["weights"] = self.ensemble_state_dict
        torch.save(ensemble_metadict, self.filename + "_ensemble_metadict.tar")


class EnsembleTrainer(BaseEnsembleTrainer):
    """
    Deep ensemble trainer

    Args:
        model:
            Built-in AtomAI model (passed as string)
            or initialized custom PyTorch model
        nb_classes:
            Number of classes (if any) in the model's output
        **kwargs:
            Number of input, output, and latent dimensions
            for imspec models (in_dim, out_dim, latent_dim)
    
    Example:

        >>> # Train an ensemble of Unet-s
        >>> etrainer = aoi.trainers.EnsembleTrainer(
        >>>    "Unet", batch_norm=True, nb_classes=3, with_dilation=False)
        >>> etrainer.compile_ensemble_trainer(training_cycles=500)
        >>> # Train 10 different models from scratch
        >>> smodel, ensemble = etrainer.train_ensemble_from_scratch(
        >>>    images, labels, images_test, labels_test, n_models=10)
    """
    def __init__(self,
                 model: Union[str, Type[torch.nn.Module]] = None,
                 nb_classes: int = 1,
                 **kwargs) -> None:
        super(EnsembleTrainer, self).__init__()
        """
        Initializes ensemble trainer
        """
        self.nb_classes = nb_classes
        if isinstance(model, str):
            if model in ["Unet", "dilnet", "SegResNet", "ResHedNet"]:
                self.net, self.meta_state_dict = init_fcnn_model(
                    model, self.nb_classes, **kwargs)
                self.accuracy_fn = accuracy_fn_seg(nb_classes)
            elif model == "imspec":
                keys_check = []
                for k in ["in_dim", "out_dim", "latent_dim"]:
                    if k not in kwargs.keys():
                        keys_check.append(k)
                if len(keys_check) > 0:
                    raise AssertionError(
                        "Specify input, output, and latent dimensions " +
                        "(Missing dimensions: {})".format(str(keys_check)[1:-1]))
                self.in_dim = kwargs.pop("in_dim")
                self.out_dim = kwargs.pop("out_dim")
                latent_dim = kwargs.pop("latent_dim")
                self.net, self.meta_state_dict = init_imspec_model(
                    self.in_dim, self.out_dim, latent_dim, **kwargs)
            self.net.to(self.device)
        else:
            self.set_model(model, nb_classes)

        self.meta_state_dict["weights"] = self.net.state_dict()
        self.meta_state_dict["optimizer"] = self.optimizer

    def compile_ensemble_trainer(self,
                                 **kwargs: compile_kwargs_type
                                 ) -> None:
        """
        Compile ensemble trainer.

        Args:
            kwargs:
                Keyword arguments to be passed to BaseTrainer.compile_trainer
                (loss, optimizer, compute_accuracy, full_epoch, swa,
                perturb_weights, batch_size, training_cycles, accuracy_metrics,
                filename, print_loss, plot_training_history)
        """
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
                       augment_fn: augfn_type = None
                       ) -> Type[torch.nn.Module]:
        """
        Trains baseline weights

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            seed:
                seed to be used for pytorch and numpy random numbers generator
            augment_fn:
                function that takes two torch tensors (features and targets),
                peforms some transforms, and returns the transformed tensors.
                The dimensions of the transformed tensors must be the same as
                the dimensions of the original ones.

        Returns:
            Trained baseline weights
        """
        if self.net is None:
            raise AssertionError("You need to set a model first")

        train_data = self.preprocess_train_data(
            X_train, y_train, X_test, y_test)
        self.set_data(*train_data, **self.kdict)

        self._reset_rng(seed)
        self._reset_weights()
        self._reset_training_history()

        self.compile_trainer(
            (X_train, y_train, X_test, y_test), **self.kdict)
        self.data_augmentation(augment_fn)
        self.fit()

        return self.net

    def preprocess_train_data(self,
                              *args: np.ndarray
                              ) -> Tuple[torch.Tensor]:
        """
        Training and test data preprocessing
        """
        if self.meta_state_dict.get("model_type") == "seg":
            train_data = set_data_seg(*args, self.nb_classes)
        elif self.meta_state_dict.get("model_type") == "imspec":
            train_data = set_data_imspec(*args, (self.in_dim, self.out_dim))
        return train_data


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
