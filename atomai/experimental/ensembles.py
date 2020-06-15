import copy
from typing import Dict, Tuple, Type

import numpy as np
import torch
from atomai import atomnet, models
from atomai.utils import average_weights


class ensemble_trainer:
    """
    Trains multiple deep learning models, each with its own unique trajectory

     Args:
        X_train: Training images
        y_train: Training labels (aka ground truth aka masks)
        X_test: Test images
        y_test: Test labels
        n_models: number of models in ensemble
        model_type: 'dilUnet' or 'dilnet'. See atomai.models for details
        training_cycles_base: Number of training iterations for baseline model
        training_cycles_ensemble: Number of training iterations for every ensemble model
        upsampling: "bilinear" or "nearest" upsampling method
        filename: Filepath for saving weights
        **kwargs:
            One can also pass kwargs to atomai.atomnet.trainer class for adjusting
            network parameters (e.g. batchnorm=True, nb_filters=25, etc.)
            and to atomai.utils.datatransform class to perform the augmentation
            "on-the-fly" (e.g. rotation=True, gauss=[20, 60], etc.)
    """
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 n_models=30, model_type: str = "dilUnet",
                 training_cycles_base: int = 1000,
                 training_cycles_ensemble: int = 50,
                 upsampling_method="bilinear",
                 filename: str = "./model.pt", **kwargs: Dict) -> None:

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.model_type, self.n_models = model_type, n_models
        self.iter_base = training_cycles_base
        self.iter_ensemble = training_cycles_ensemble
        self.upsampling_method = upsampling_method
        self.filename, self.kdict = filename, kwargs

    def train_baseline(self) -> Type[torch.nn.Module]:
        """
        Trains a single baseline model
        """
        trainer_base = atomnet.trainer(
            self.X_train, self.y_train, self.X_test, self.y_test,
            self.iter_base, self.model_type, upsampling=self.upsampling_method,
            plot_training_history=True, savename=self.filename + "_base.pt",
            **self.kdict)
        trained_basemodel = trainer_base.run()

        return trained_basemodel

    def train_ensemble(self, basemodel) -> Dict[str, torch.Tensor]:
        """
        Trains ensemble of models starting each time from baseline weights
        """
        initial_model_state_dict = copy.deepcopy(basemodel.state_dict())
        ensemble_state_dict = {}
        print('Training ensemble models:')
        for i in range(self.n_models):
            trainer_i = atomnet.trainer(
                self.X_train, self.y_train, self.X_test, self.y_test,
                self.iter_ensemble, self.model_type,
                upsampling=self.upsampling_method, batch_seed=i,
                print_loss=10, plot_training_history=False, **self.kdict)
            trainer_i.net.load_state_dict(initial_model_state_dict)
            trained_model_i = trainer_i.run()
            ensemble_state_dict[i] = trained_model_i.state_dict()
        ensemble_state_dict_aver = average_weights(ensemble_state_dict)

        ensemble_metadict = copy.deepcopy(trainer_i.meta_state_dict)
        ensemble_metadict["weights"] = ensemble_state_dict
        torch.save(ensemble_metadict, self.filename + "_ensemble.tar")
        ensemble_aver_metadict = copy.deepcopy(trainer_i.meta_state_dict)
        ensemble_aver_metadict["weights"] = ensemble_state_dict_aver
        torch.save(ensemble_metadict, self.filename + "_ensemble_aver_weights.pt")

        return ensemble_state_dict, ensemble_state_dict_aver

    def run(self) -> Tuple[Type[torch.nn.Module], Dict, Dict]:
        """
        Trains a baseline model and ensemble of models
        """
        print('Training baseline model:')
        basemodel = self.train_baseline()
        print('Training ensemble models:')
        ensemble, ensemble_aver = self.train_ensemble(basemodel)
        return basemodel, ensemble, ensemble_ave


def ensemble_predict(predictive_model: Type[torch.nn.Module],
                     ensemble: Dict[int, Dict[str, torch.Tensor]],
                     expdata: np.ndarray,num_classes: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Makes a prediction (mean and variance/uncertainty) with ensemble of models
    Args:
        predictive_model: model skeleton (can be with randomly initialized weights)
        ensemble: nested dictionary with weights of each model in the ensemble
        expdata: 2D experimental image
        num_classes: number of classes in the classification scheme # TODO: this should be inferred automatically from predictive model
    """
    use_gpu = torch.cuda.is_available()
    img_h, img_w = expdata.shape
    nn_output_ensemble = np.zeros((len(ensemble), img_h, img_w, num_classes))
    for i, w in ensemble.items():
        predictive_model.load_state_dict(w)
        predictive_model.eval()
        _, nn_output = atomnet.predictor(
            expdata, predictive_model, use_gpu=use_gpu, verbose=False).decode()
        nn_output_ensemble[i] = nn_output[0]
    nn_output_mean = np.mean(nn_output_ensemble, axis=0)
    nn_output_var = np.var(nn_output_ensemble, axis=0)
    return nn_output_mean, nn_output_var


def load_ensemble(meta_state_dict: str) -> Tuple[Type[torch.nn.Module], Dict[int, Dict[str, torch.Tensor]]]:
    """
    Loads trained ensemble models
    Args:
        meta_state_dict (str):
            filepath to dictionary with trained weights and key information
            about model's structure
    Returns:
        Model skeleton (initialized) and dictionary with weights of all the models
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
        model = models.dilUnet(
            nb_classes, nb_filters, dropout,
            batchnorm, upsampling, with_dilation,
            layers=layers)
    elif model_type == 'dilnet':
        model = models.dilnet(
            nb_classes, nb_filters, dropout,
            batchnorm, upsampling, layers=layers)
    else:
        raise NotImplementedError(
            "The network must be either 'dilUnet' or 'dilnet'"
        )
    model.load_state_dict(checkpoint[0])
    return model.eval(), checkpoint