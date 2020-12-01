"""
etrainer.py
========

Module for training ensmebles of fully convolutional neural networs (NNs)
for atom/defect/particle finding and ensembles of encoder-decoder NNs
for prediction of spectra/images from images/spectra.

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)

"""


import copy
import warnings
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from atomai import losses_metrics
from atomai.nets import dilnet, dilUnet, signal_ed
from atomai.transforms import datatransform, unsqueeze_channels
from atomai.utils import (average_weights, check_signal_dims, dummy_optimizer,
                          gpu_usage_map, init_fcnn_dataloaders,
                          init_imspec_dataloaders, ndarray2list, plot_losses,
                          preprocess_training_data, sample_weights,
                          set_train_rng)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", module="torch.nn.functional")

training_data_types = Union[np.ndarray, List[np.ndarray], Dict[int, np.ndarray]]
ensemble_out = Tuple[Dict[int, Dict[str, torch.Tensor]], Type[torch.nn.Module]]


class ensemble_trainer:
    """
    Trains multiple deep learning models, each with its own unique trajectory

    Args:
        X_train (numpy array): Training images
        y_train (numpy array): Training labels (aka ground truth aka masks)
        X_test (numpy array): Test images
        y_test (numpy array): Test labels
        n_models (int): number of models in ensemble
        model(str): 'dilUnet' or 'dilnet'. See atomai.models for details
        strategy (str): Select between 'from_scratch', 'from_baseline' and 'swag'.
            If 'from_scratch' is selected, the *n* models are trained independently
            starting each time with a different random initialization. If
            'from_baseline' is selected, a basemodel is trained for *N* epochs
            and then its weights are used as a baseline to train multiple ensemble models
            for n epochs (*n* << *N*), each with different random shuffling of batches
            (and different seed for data augmentation if any). If 'swag' is
            selected, a SWAG-like sampling of weights is performed at the end of
            a single model training.
        swa (bool):
            Stochastic weights averaging  at the end of each training trajectory
        training_cycles_base (int): Number of training iterations for baseline model
        training_cycles_ensemble (int): Number of training iterations for every ensemble model
        filename (str): Filepath for saving weights
        **kwargs:
            One can also pass kwargs to atomai.atomnet.trainer class for adjusting
            network parameters (e.g. batchnorm=True, nb_filters=25, etc.)
            and to atomai.utils.datatransform class to perform the augmentation
            "on-the-fly" (e.g. rotation=True, gauss=[20, 60], etc.)
    """
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray = None, y_test: np.ndarray = None,
                 trainer="seg", n_models=30, model: str = "dilUnet",
                 strategy: str = "from_baseline", swa=False,
                 training_cycles_base: int = 1000,
                 training_cycles_ensemble: int = 50,
                 filename: str = "./model", **kwargs: Dict) -> None:
        """
        Initializes parameters of ensemble trainer
        """
        if X_test is None or y_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=kwargs.get("test_size", 0.15),
                shuffle=True, random_state=0)
        set_train_rng(seed=1)
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        if trainer not in ["seg", "imspec"]:
            raise NotImplementedError(
                "Choose between 'seg' and 'imspec' trainers")
        self.trainer_type = trainer

        self.model_type, self.n_models = model, n_models
        self.strategy = strategy
        if self.strategy not in ["from_baseline", "from_scratch", "swag"]:
            raise NotImplementedError(
                "Select 'from_baseline' 'from_scratch', or 'swag'  strategy")
        self.iter_base = training_cycles_base
        if self.strategy == "from_baseline":
            self.iter_ensemble = training_cycles_ensemble
        self.filename, self.kdict = filename, kwargs
        if swa or self.strategy == 'swag':
            self.kdict["swa"] = True
            #self.kdict["use_batchnorm"] = False  # there were some issues when using batchnorm together with swa in pytorch 1.4
        self.ensemble_state_dict = {}

    def train_baseline(self,
                       seed: int = 1,
                       batch_seed: int = 1) -> Type[base_trainer]:
        """
        Trains a single "baseline" model
        """
        if self.strategy == "from_baseline":
            print('Training baseline model:')
        trainer = seg_trainer if self.trainer_type == "seg" else imspec_trainer
        trainer_base = trainer(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            self.iter_base, self.model_type,
            seed=seed, batch_seed=batch_seed,
            plot_training_history=True,
            savename=self.filename + "_base",
            **self.kdict)
        _ = trainer_base.run()

        return trainer_base

    def train_from_baseline(self,
                            basemodel: Union[OrderedDict, Type[torch.nn.Module]],
                            **kwargs: Dict) -> ensemble_out:
        """
        Trains ensemble of models starting each time from baseline weights

        Args:
            basemodel (pytorch object): Baseline model or baseline weights
            **kwargs: Updates kwargs from the ensemble class initialization
                (can be useful for iterative training)
        """
        if len(kwargs) != 0:
            for k, v in kwargs.items():
                self.kdict[k] = v
        if isinstance(basemodel, OrderedDict):
            initial_model_state_dict = copy.deepcopy(basemodel)
        else:
            initial_model_state_dict = copy.deepcopy(basemodel.state_dict())
        n_models = kwargs.get("n_models")
        if n_models is not None:
            self.n_models = n_models
        if "print_loss" not in self.kdict.keys():
            self.kdict["print_loss"] = 10
        filename = kwargs.get("filename")
        training_cycles_ensemble = kwargs.get("training_cycles_ensemble")
        if training_cycles_ensemble is not None:
            self.iter_ensemble = training_cycles_ensemble
        if filename is not None:
            self.filename = filename
        trainer = seg_trainer if self.trainer_type == "seg" else imspec_trainer
        print('Training ensemble models:')
        for i in range(self.n_models):
            print('Ensemble model', i+1)
            trainer_i = trainer(
                self.X_train, self.y_train, self.X_test, self.y_test,
                self.iter_ensemble, self.model_type, batch_seed=i+1,
                plot_training_history=False, **self.kdict)
            self.update_weights(trainer_i.net.state_dict().values(),
                                initial_model_state_dict.values())
            trained_model_i = trainer_i.run()
            self.ensemble_state_dict[i] = trained_model_i.state_dict()
            self.save_ensemble_metadict(trainer_i.meta_state_dict)
        averaged_weights = average_weights(self.ensemble_state_dict)
        trainer_i.net.load_state_dict(averaged_weights)
        return self.ensemble_state_dict, trainer_i.net

    def train_ensemble_from_baseline(self) -> ensemble_out:
        """
        Trains a baseline model and ensemble of model starting each time
        from the baseline model weights
        """
        baseline = self.train_baseline()
        ensemble, smodel = self.train_from_baseline(baseline.net)
        return ensemble, smodel

    def train_ensemble_from_scratch(self) -> ensemble_out:
        """
        Trains ensemble of models starting every time from scratch with
        different initialization (for both weights and batches shuffling)
        """
        print("Training ensemble models:")
        for i in range(self.n_models):
            print("Ensemble model {}".format(i + 1))
            trainer_i = self.train_baseline(seed=i+1, batch_seed=i+1)
            self.ensemble_state_dict[i] = trainer_i.net.state_dict()
            self.save_ensemble_metadict(trainer_i.meta_state_dict)
        averaged_weights = average_weights(self.ensemble_state_dict)
        trainer_i.net.load_state_dict(averaged_weights)
        return self.ensemble_state_dict, trainer_i.net

    def train_swag(self) -> ensemble_out:
        """
        Performs SWAG-like weights sampling at the end of single model training
        """
        trainer_i = self.train_baseline()
        sampled_weights = sample_weights(
            trainer_i.recent_weights, self.n_models)
        self.ensemble_state_dict = sampled_weights
        self.save_ensemble_metadict(trainer_i.meta_state_dict)
        return self.ensemble_state_dict, trainer_i.net

    def save_ensemble_metadict(self, meta_state_dict: Dict) -> None:
        """
        Saves meta dictionary with ensemble weights and key information about
        model's structure (needed to load it back) to disk'
        """
        ensemble_metadict = copy.deepcopy(meta_state_dict)
        ensemble_metadict["weights"] = self.ensemble_state_dict
        torch.save(ensemble_metadict, self.filename + "_ensemble.tar")

    @classmethod
    def update_weights(cls,
                       statedict1: Dict[str, torch.Tensor],
                       statedict2: Dict[str, torch.Tensor]) -> None:
        """
        Updates (in place) state dictionary of pytorch model
        with weights from another model with the same structure;
        skips layers that have different dimensions
        (e.g. if one model is for single class classification
        and the other one is for multiclass classification,
        then the last layer wights are not updated)
        """
        for p1, p2 in zip(statedict1, statedict2):
            if p1.shape == p2.shape:
                p1.copy_(p2)

    def set_data(self,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray = None, y_test: np.ndarray = None) -> None:
        """
        Sets data for ensemble training (useful for iterative training)
        """
        if X_test is None or y_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=self.kdict.get("test_size", 0.15),
                shuffle=True, random_state=0)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def run(self) -> ensemble_out:
        """
        Trains a baseline model and ensemble of models
        """
        if self.strategy == 'from_baseline':
            ensemble, smodel = self.train_ensemble_from_baseline()
        elif self.strategy == 'from_scratch':
            ensemble, smodel = self.train_ensemble_from_scratch()
        elif self.strategy == 'swag':
            ensemble, smodel = self.train_swag()
        else:
            raise NotImplementedError(
                "The strategy must be 'from_baseline', 'from_scratch', 'swag' or 'from_scratch_swa'")
        return ensemble, smodel
