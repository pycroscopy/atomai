"""
epredictor.py
===========

Module for predicting with ensembles of pre-trained neural networks

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Dict, Tuple, Type, Union

import numpy as np
import torch
from torch.nn.functional import softmax

from ..utils import (get_downsample_factor, torch_format_image,
                     torch_format_spectra, cluster_coord)
from .predictor import BasePredictor, Locator


class EnsemblePredictor(BasePredictor):
    """
    Prediction with ensemble of models

    Args:
        skeleton: Model skeleton (cam be with randomly initialized weights)
        ensemble: Ensemble of trained weights
        data_type: Input data type (image or spectra)
        output_type: Output data type (image or spectra)
        nb_classes: Number of classes (e.g. for semantic segmentation)
        in_dim: Input data size (for models with fully-connected layers)
        out_dim: Output data size (for models with fully-connected layers)
        **output_shape: Optionally one may specify the exact output shape
        **verbose: verbosity
    
    Example:

        >>> p = aoi.predictors.EnsemblePredictor(skeleton, ensemble, nb_classes=3)
        >>> nn_out_mean, nn_out_var = p.predict(expdata)
    """
    def __init__(self,
                 skeleton: Type[torch.nn.Module],
                 ensemble: Dict[int, Dict[str, torch.Tensor]],
                 data_type: str = "image",
                 output_type: str = "image",
                 nb_classes: int = None,
                 in_dim: Tuple[int] = None,
                 out_dim: Tuple[int] = None,
                 **kwargs: Union[str, Tuple[int]]) -> None:
        """
        Initialize ensemble predictor
        """
        super(EnsemblePredictor, self).__init__()
        if output_type not in ["image", "spectra"]:
            raise TypeError("Supported output types are 'image' and 'spectra'")
        inout = [data_type, output_type]
        inout_d = not all([in_dim, out_dim])
        if inout in (["image", "spectra"], ["spectra", "image"]) and inout_d:
            raise TypeError(
                "Specify input (in_dim) & output (out_dim) dimensions")
        self.device = "cpu"
        if kwargs.get("use_gpu", True) and torch.cuda.is_available():
            if kwargs.get("device") is None:
                self.device = "cuda"
            else:
                self.device = kwargs.get("device")
        self.model = skeleton
        self.ensemble = ensemble
        self.data_type = data_type
        self.output_type = output_type
        self.nb_classes = nb_classes
        self.in_dim, self.out_dim = in_dim, out_dim
        self.downsample_factor = None
        self.logits = kwargs.get("logits", True)
        self.output_shape = kwargs.get("output_shape")
        verbose = kwargs.get("verbose", 1)
        if verbose:
            self.everbose = True
            self.verbose = True if verbose > 1 else False

    def _set_output_shape(self, data: np.ndarray) -> None:
        """
        Sets output shape
        """
        if self.data_type == self.output_type == "image":
            if self.nb_classes:  # semantic segmentation
                out_shape = (len(data), self.nb_classes, *data.shape[2:])
            else:  # image cleaning
                out_shape = (len(data), 1, *data.shape[2:])
        elif self.data_type == "spectra" and self.output_type == "image":
            if self.nb_classes:
                out_shape = (len(data), self.nb_classes, *self.out_dim)
            else:
                out_shape = (len(data), 1, *self.out_dim)
        elif self.data_type == "image" and self.output_type == "spectra":
            out_shape = (len(data), 1, *self.out_dim)
        elif self.data_type == self.output_type == "spectra":
            out_shape = (len(data), 1, *data.shape[2:])
        else:
            raise TypeError("Data not understood")

        self.output_shape = out_shape

    def preprocess(self,
                   data: np.ndarray,
                   norm: bool = True
                   ) -> torch.Tensor:
        """
        Preprocesses data depending on type (image or spectra)
        """
        if self.data_type == "image":
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            data = torch_format_image(data, norm)
        elif self.data_type == "spectra":
            if data.ndim == 1:
                data = data[np.newaxis, ...]
            data = torch_format_spectra(data, norm)
        return data

    def ensemble_forward_(self,
                          data: torch.Tensor,
                          out_shape: Tuple[int]
                          ) -> Tuple[np.ndarray]:
        """
        Computes mean and variance of prediction with ensemble models
        """
        eprediction = self.ensemble_forward(data, out_shape)

        return np.mean(eprediction, axis=0), np.var(eprediction, axis=0)

    def ensemble_forward(self,
                         data: torch.Tensor,
                         out_shape: Tuple[int],
                         num_batches: int = 1) -> np.ndarray:
        """
        Computes prediction with ensemble models.
        Returns ALL calculated predictions (n_models * n_samples).
        """
        eprediction = np.zeros(
            (len(self.ensemble), *out_shape))
        for i, m in enumerate(self.ensemble.values()):
            self.model.load_state_dict(m)
            self._model2device()
            if num_batches > 1:
                prob = self.batch_predict(
                    data, out_shape, num_batches)
            else:
                prob = self.forward_(data)
            nclasses = 0 if not self.nb_classes else self.nb_classes
            if self.logits:
                if nclasses > 1:
                    prob = softmax(prob, dim=1)
                elif self.nb_classes == 1:
                    prob = torch.sigmoid(prob)
            else:
                if nclasses > 1:
                    prob = torch.exp(prob)
            eprediction[i] = prob.cpu().numpy()

        return eprediction

    def ensemble_batch_predict(self,
                               data: np.ndarray,
                               num_batches: int = 10
                               ) -> Tuple[np.ndarray]:
        """
        Batch-by-batch prediction with ensemble models
        """
        batch_size = len(data) // num_batches
        if batch_size < 1:
            num_batches = batch_size = 1
        prediction_mean = np.zeros(shape=self.output_shape)
        prediction_var = np.zeros(shape=self.output_shape)
        for i in range(num_batches):
            if self.everbose:
                print("\rBatch {}/{}".format(i+1, num_batches), end="")
            data_i = data[i*batch_size:(i+1)*batch_size]
            pred_mean, pred_var = self.ensemble_forward_(
                data_i, (batch_size, *self.output_shape[1:]))
            prediction_mean[i*batch_size:(i+1)*batch_size] = pred_mean
            prediction_var[i*batch_size:(i+1)*batch_size] = pred_var
        data_i = data[(i+1)*batch_size:]
        if len(data_i) > 0:
            pred_mean, pred_var = self.ensemble_forward_(
                data_i, (len(data_i), *self.output_shape[1:]))
            prediction_mean[(i+1)*batch_size:] = pred_mean
            prediction_var[(i+1)*batch_size:] = pred_var
        return prediction_mean, prediction_var

    def predict(self,
                data: np.ndarray,
                num_batches: int = 10,
                format_out: str = "channel_last",
                norm: bool = True
                ) -> Tuple[np.ndarray]:
        """
        Predicts mean and variance for all the data points
        with ensemble of models

        Args:
            data: input data
            num_batches:
                number of batches for batch-by-batch prediction (Default: 10)
            format_out:
                'channel_last' of 'channel_first' dimension order in output
            norm: Normalize input data to (0, 1)

        Returns:
            Tuple of numpy arrays with predicted mean and variance
        """

        if format_out not in ["channel_first", "channel_last"]:
            raise ValueError(
                "Specify channel_last or channel_first output format")

        data = self.preprocess(data, norm)
        if not self.output_shape:
            self._set_output_shape(data)

        if (self.data_type == self.output_type == "image"
           and self.downsample_factor is None):
            self.downsample_factor = get_downsample_factor(self.model)

        prediction = self.ensemble_batch_predict(data, num_batches)
        prediction_mean, prediction_var = prediction

        # channel transpose
        if format_out == "channel_last":
            size_dim = np.arange(prediction_mean.ndim - 2) + 2
            c_tr = (0, *size_dim, 1)
        elif format_out == "channel_first":
            c_tr = np.arange(prediction_mean.ndim)

        return prediction_mean.transpose(c_tr), prediction_var.transpose(c_tr)


def ensemble_locate(nn_output_ensemble: np.ndarray,
                    **kwargs: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds coordinates for each ensemble predictions

    Args:
        nn_output_ensembles (numpy array):
            5D numpy array with ensemble predictions
        **eps (float):
            DBSCAN epsilon for clustering coordinates
        **threshold (float):
            threshold value for atomnet.locator

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
            coord = Locator(thresh).run(img[None, ...])
            coordinates[i2] = coord[0]
        _, coord_mean, coord_var = cluster_coord(coordinates, eps)
        coord_mean_all[i] = coord_mean
        coord_var_all[i] = coord_var
    return coord_mean_all, coord_var_all

