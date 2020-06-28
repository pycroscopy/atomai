import copy
from collections import OrderedDict
from typing import Dict, Tuple, Type, Union

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from atomai.core import atomnet, atomstat, models
from atomai.utils import average_weights, Hook, mock_forward


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
        filename: Filepath for saving weights
        **kwargs:
            One can also pass kwargs to atomai.atomnet.trainer class for adjusting
            network parameters (e.g. batchnorm=True, nb_filters=25, etc.)
            and to atomai.utils.datatransform class to perform the augmentation
            "on-the-fly" (e.g. rotation=True, gauss=[20, 60], etc.)
    """
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray = None, y_test: np.ndarray = None,
                 n_models=30, model_type: str = "dilUnet",
                 training_cycles_base: int = 1000,
                 training_cycles_ensemble: int = 50,
                 filename: str = "./model", **kwargs: Dict) -> None:

        if X_test is None or y_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=kwargs.get("test_size", 0.15),
                shuffle=True, random_state=0)
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.model_type, self.n_models = model_type, n_models
        self.iter_base = training_cycles_base
        self.iter_ensemble = training_cycles_ensemble
        self.filename, self.kdict = filename, kwargs
        self.ensemble_state_dict = {}

    def train_baseline(self) -> Type[torch.nn.Module]:
        """
        Trains a single baseline model
        """
        print('Training baseline model:')
        trainer_base = atomnet.trainer(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            self.iter_base, self.model_type,
            plot_training_history=True,
            savename=self.filename + "_base.pt",
            **self.kdict)
        trained_basemodel = trainer_base.run()

        return trained_basemodel

    def train_ensemble(self,
                       basemodel: Union[OrderedDict, Type[torch.nn.Module]],
                       **kwargs: Dict
                       ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Type[torch.nn.Module]]:
        """
        Trains ensemble of models starting each time from baseline weights

        Args:
            basemodel: Baseline model or baseline weights
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
        filename = kwargs.get("filename")
        training_cycles_ensemble = kwargs.get("training_cycles_ensemble")
        if training_cycles_ensemble is not None:
            self.iter_ensemble = training_cycles_ensemble
        if filename is not None:
            self.filename = filename
        print('Training ensemble models:')
        for i in range(self.n_models):
            print('Ensemble model', i+1)
            trainer_i = atomnet.trainer(
                self.X_train, self.y_train, self.X_test, self.y_test,
                self.iter_ensemble, self.model_type, batch_seed=i,
                print_loss=10, plot_training_history=False, **self.kdict)
            self.update_weights(trainer_i.net.state_dict().values(),
                                initial_model_state_dict.values())
            trained_model_i = trainer_i.run()
            self.ensemble_state_dict[i] = trained_model_i.state_dict()

        ensemble_metadict = copy.deepcopy(trainer_i.meta_state_dict)
        ensemble_metadict["weights"] = self.ensemble_state_dict
        torch.save(ensemble_metadict, self.filename + "_ensemble.tar")

        ensemble_state_dict_aver = average_weights(self.ensemble_state_dict)
        ensemble_aver_metadict = copy.deepcopy(trainer_i.meta_state_dict)
        ensemble_aver_metadict["weights"] = ensemble_state_dict_aver
        torch.save(ensemble_aver_metadict, self.filename + "_ensemble_aver_weights.pt")

        trainer_i.net.load_state_dict(ensemble_state_dict_aver)
        
        return self.ensemble_state_dict, trainer_i.net

    @classmethod
    def update_weights(cls, statedict1, statedict2):
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

    def run(self) -> Tuple[Type[torch.nn.Module], Dict, Dict]:
        """
        Trains a baseline model and ensemble of models
        """
        basemodel = self.train_baseline()
        ensemble, ensemble_aver = self.train_ensemble(basemodel)
        return basemodel, ensemble, ensemble_aver


def ensemble_predict(predictive_model: Type[torch.nn.Module],
                     ensemble: Dict[int, Dict[str, torch.Tensor]],
                     expdata: np.ndarray, calculate_coordinates=True,
                     **kwargs: Dict
                     ) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]]:
    """
    Predicts mean and variance/uncertainty in image pixels
    and (optionally) coordinates with ensemble of models

    Args:
        predictive_model: model skeleton (can have randomly initialized weights)
        ensemble: nested dictionary with weights of each model in the ensemble
        expdata: 2D experimental image or 3D image stack
        calculate_coordinates: computes atomic coordinates for each prediction
        **eps: DBSCAN epsilon for clustering coordinates
        **threshold: value at which a neural network output is thresholded for calculating coordinates
        **num_classes: number of classes in the classification scheme
        **downsample_factor: image downsampling (max_size / min_size) in NN
    """
    use_gpu = torch.cuda.is_available()
    if np.ndim(expdata) == 2:
        expdata = np.expand_dims(expdata, axis=0)
    batch_dim, img_h, img_w = expdata.shape

    num_classes = kwargs.get("num_classes")
    if num_classes is None:
        hookF = [Hook(layer[1]) for layer in list(predictive_model._modules.items())]
        mock_forward(predictive_model)
        num_classes = [hook.output.shape for hook in hookF][-1][1]
    downsample_factor = kwargs.get("downsample_factor")
    if downsample_factor is None:
        hookF = [Hook(layer[1]) for layer in list(predictive_model._modules.items())]
        mock_forward(predictive_model)
        imsize = [hook.output.shape[-1] for hook in hookF]
        downsample_factor = max(imsize) / min(imsize)

    nn_output_ensemble = np.zeros((
        len(ensemble), batch_dim, img_h, img_w, num_classes))
    for i, w in ensemble.items():
        predictive_model.load_state_dict(w)
        predictive_model.eval()
        _, nn_output = atomnet.predictor(
            expdata, predictive_model,
            nb_classes=num_classes, downsampling=downsample_factor,
            use_gpu=use_gpu, verbose=False).decode()
        nn_output_ensemble[i] = nn_output
    nn_output_mean = np.mean(nn_output_ensemble, axis=0)
    nn_output_var = np.var(nn_output_ensemble, axis=0)
    coord_mean, coord_var = None, None
    if calculate_coordinates:
        coord_mean, coord_var = ensemble_locate(nn_output_ensemble, **kwargs)
    return (nn_output_mean, nn_output_var), (coord_mean, coord_var)


def ensemble_locate(nn_output_ensemble: np.ndarray,
                    **kwargs: Dict) -> Tuple[Tuple[np.ndarray, np.ndarray]]:
    """
    Finds coordinates for each ensemble predictions
    (basically, an atomnet.locator for ensembles)

    Args:
        nn_output_ensembles: 5D numpy array with ensemble predictions
        **eps: DBSCAN epsilon for clustering coordinates
        **threshold: threshold value for atomnet.locator

    Returns:
        Mean and variance for every detected atom/defect/particle coordinate
    """
    eps = kwargs.get("eps", 0.5)
    thresh = kwargs.get("threshold", 0.5)
    coord_mean_all = {}
    coord_var_all = {}
    for i in range(nn_output_ensemble.shape[1]):
        coordinates = {}
        nn_output = nn_output_ensemble[:, i]
        for i2, img in enumerate(nn_output):
            coord = atomnet.locator(img[None, ...], thresh).run()
            coordinates[i2] = coord[0]
        _, coord_mean, coord_var = atomstat.cluster_coord(coordinates, eps)
        coord_mean_all[i] = coord_mean
        coord_var_all[i] = coord_var
    return coord_mean_all, coord_var_all


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
