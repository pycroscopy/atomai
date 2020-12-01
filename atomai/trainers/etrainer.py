from copy import deepcopy as dc
from typing import Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ..losses_metrics import IoU
from ..nets import init_fcnn_model, init_imspec_model
from ..utils import (average_weights, init_fcnn_dataloaders,
                     init_imspec_dataloaders, preprocess_training_image_data,
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
            self.set_model(model, nb_classes) # This part will be modified in EnsembleTrainer for Unet and ImSpec
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
                                    augment_fn: augfn_type = None):
        print("Training ensemble models (trategy = 'from_scratch'")
        for i in range(n_models):
            print("Ensemble model {}".format(i + 1))
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
                 model: Union[str, Type[torch.nn.Module]],
                 nb_classes: int = 1,
                 **kwargs) -> None:

        self.nb_classes = nb_classes
        seg = True
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
                seg = False
            self.net.to(self.device)
        else:
            self.set_model(model, nb_classes)        
        if seg:
            self.set_data = self.set_data_seg
        else:
            self.set_data = self.set_data_imspec

        self.kdict = kwargs    

    def set_data_imspec(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test: Optional[np.ndarray] = None,
                        y_test: Optional[np.ndarray] = None,
                        **kwargs: Union[float, int]) -> None:
        """
        Sets training and test data.

        Args:

        X_train (numpy array):
            4D numpy array with image data (n_samples x 1 x height x width)
            or 3D numpy array with spectral data (n_samples x 1 x signal_length).
            It is also possible to pass 3D and 2D arrays by ignoring the channel dim,
            which will be added automatically.
        y_train (numpy array):
            3D numpy array with spectral data (n_samples x 1 x signal_length)
            or 4D numpy array with image data (n_samples x 1 x height x width).
            It is also possible to pass 2D and 3D arrays by ignoring the channel dim,
            which will be added automatically. Note that if your X_train data are images,
            then your y_train must be spectra and vice versa.
        X_test (numpy array):
            4D numpy array with image data (n_samples x 1 x height x width)
            or 3D numpy array with spectral data (n_samples x 1 x signal_length).
            It is also possible to pass 3D and 2D arrays by ignoring the channel dim,
            which will be added automatically.
        y_test (numpy array):
            3D numpy array with spectral data (n_samples x 1 x signal_length)
            or 4D numpy array with image data (n_samples x 1 x height x width).
            It is also possible to pass 2D and 3D arrays by ignoring the channel dim,
            which will be added automatically. Note that if your X_train data are images,
            then your y_train must be spectra and vice versa.
        kwargs:
            Parameters for train_test_split ('test_size' and 'seed') when
            separate test set is not provided
        """

        if X_test is None or y_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=kwargs.get("test_size", .15),
                shuffle=True, random_state=kwargs.get("seed", 1))

        if self.full_epoch:
            self.train_loader, self.test_loader, dims = init_imspec_dataloaders(
                X_train, y_train, X_test, y_test, self.batch_size)
        else:
            (self.X_train, self.y_train,
             self.X_test, self.y_test, dims) = preprocess_training_imspec_data(
                X_train, y_train, X_test, y_test, self.batch_size)

        if dims[0] != self.in_dim or dims[1] != self.out_dim:
            raise AssertionError(
                "The input/output dimensions of the imspec model must match" +
                " the height, width and length (for spectra) of training")

    def set_data_seg(self,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_test: Optional[np.ndarray] = None,
                     y_test: Optional[np.ndarray] = None,
                     **kwargs: Union[float, int]) -> None:
        """
        Sets training and test data.

        Args:

        X_train (numpy array):
            4D numpy array (3D image tensors stacked along the first dim)
            representing training images
        y_train (numpy array):
            4D (binary) / 3D (multiclass) numpy array
            where 3D / 2D images stacked along the first array dimension
            represent training labels (aka masks aka ground truth).
            The reason why in the multiclass case the images are 4-dimensional
            tensors and the labels are 3-dimensional tensors is because of how
            the cross-entropy loss is calculated in PyTorch
            (see https://pytorch.org/docs/stable/nn.html#nllloss).
        X_test (numpy array):
            4D numpy array (3D image tensors stacked along the first dim)
            representing test images
        y_test (numpy array):
            4D (binary) / 3D (multiclass) numpy array
            where 3D / 2D images stacked along the first array dimension
            represent test labels (aka masks aka ground truth)
        kwargs:
            Parameters for train_test_split ('test_size' and 'seed') when
            separate test set is not provided
        """

        if X_test is None or y_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=kwargs.get("test_size", .15),
                shuffle=True, random_state=kwargs.get("seed", 1))

        if self.full_epoch:
            loaders = init_fcnn_dataloaders(
                X_train, y_train, X_test, y_test, self.batch_size)
            self.train_loader, self.test_loader, nb_classes = loaders
        else:
            (self.X_train, self.y_train,
             self.X_test, self.y_test,
             nb_classes) = preprocess_training_image_data(
                                    X_train, y_train, X_test, y_test,
                                    self.batch_size)

        if self.nb_classes != nb_classes:
            raise AssertionError("Number of specified classes" +
                                 " is different from the number of classes" +
                                 " contained in training data")


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
