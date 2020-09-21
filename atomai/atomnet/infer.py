"""
infer.py
========

Module for making predictions with trained fully convolutional neural networks
(FCNNs) and ensemble of FCNNs.

Created by Maxim Ziatdinov (maxim.ziatdinov@ai4microscopy.com)
"""

import copy
import time
import warnings
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from atomai.utils import (Hook, cluster_coord, cv_thresh, find_com, img_pad,
                          img_resize, mock_forward, peak_refinement,
                          set_train_rng, torch_format)


class predictor:
    """
    Prediction with a trained neural network

    Args:
        trained_model (pytorch object):
            Trained pytorch model (skeleton+weights)
        refine (bool):
            Atomic positions refinement with 2d Gaussian peak fitting
        resize (tuple or 2-element list):
            Target dimensions for optional image(s) resizing
        use_gpu (bool):
            Use gpu device for inference
        logits (bool):
            Indicates that the image data is passed through
            a softmax/sigmoid layer when set to False
            (logits=True for AtomAI models)
        seed (int):
            Sets seed for random number generators (for reproducibility)
        **thresh (float):
            value between 0 and 1 for thresholding the NN output
        **d (int):
            half-side of a square around each atomic position used
            for refinement with 2d Gaussian peak fitting. Defaults to 1/4
            of average nearest neighbor atomic distance
        **nb_classes (int):
            Number of classes in the model
        **downsampled (int or float):
            Downsampling factor (equal to :math:`2^n` where *n* is a number
            of pooling operations)

    Example:

        >>> # Here we load new experimental data (as 2D or 3D numpy array)
        >>> expdata = np.load('expdata-test.npy')
        >>> # Get prediction from a trained model
        >>> # (it also returns the input to NN in case the image was resized, etc.)
        >>> nn_input, (nn_output, coords) = atomnet.predictor(trained_model).run(expdata)

    """
    def __init__(self,
                 trained_model: Type[torch.nn.Module],
                 refine: bool = False,
                 resize: Union[Tuple, List] = None,
                 use_gpu: bool = False,
                 logits: bool = True,
                 seed: int = 1,
                 **kwargs: Union[int, float, bool]) -> None:
        """
        Initializes predictive object
        """
        if seed:
            set_train_rng(seed)
        model = trained_model
        self.nb_classes = kwargs.get('nb_classes', None)
        if self.nb_classes is None:
            hookF = [Hook(layer[1]) for layer in list(model._modules.items())]
            mock_forward(model)
            self.nb_classes = [hook.output.shape for hook in hookF][-1][1]
        self.downsampling = kwargs.get('downsampling', None)
        if self.downsampling is None:
            hookF = [Hook(layer[1]) for layer in list(model._modules.items())]
            mock_forward(model)
            imsize = [hook.output.shape[-1] for hook in hookF]
            self.downsampling = max(imsize) / min(imsize)
        self.model = model
        if use_gpu and torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()

        self.resize = resize
        self.logits = logits
        self.refine = refine
        self.d = kwargs.get("d", None)
        self.thresh = kwargs.get("thresh", .5)
        self.use_gpu = use_gpu
        self.verbose = kwargs.get("verbose", True)

    def preprocess(self, image_data: np.ndarray) -> torch.Tensor:
        """
        Prepares an input for a neural network
        """
        if image_data.ndim == 2:
            image_data = image_data[np.newaxis, ...]
        elif image_data.ndim == 4:
            if image_data.shape[-1] == 1:
                image_data = image_data[..., 0]
            elif image_data.shape[1] == 1:
                image_data = image_data[:, 0, ...]
        if self.resize is not None:
            image_data = img_resize(image_data, self.resize)
        image_data = img_pad(image_data, self.downsampling)
        image_data = torch_format(image_data)
        return image_data

    def predict(self, images: torch.Tensor) -> np.ndarray:
        """
        Returns 'probability' of each pixel
        in image(s) belonging to an atom/defect
        """
        if self.use_gpu and torch.cuda.is_available():
            images = images.cuda()
        self.model.eval()
        with torch.no_grad():
            prob = self.model(images)
        if self.logits:
            if self.nb_classes > 1:
                prob = F.softmax(prob, dim=1)
            else:
                prob = torch.sigmoid(prob)
        else:
            if self.nb_classes > 1:
                prob = torch.exp(prob)
            else:
                pass
        if self.use_gpu:
            images = images.cpu()
            prob = prob.cpu()
        prob = prob.permute(0, 2, 3, 1) # reshape to have channel as a last dim
        prob = prob.numpy()
        return prob

    def decode(self,
               image_data: np.ndarray,
               return_image: bool = False,
               **kwargs: int) -> Tuple[np.ndarray]:
        """
        Make prediction

        Args:
            image_data (2D or 3D numpy array):
                Image stack or a single image (all greyscale)
            return_image (bool):
                Returns images used as input into NN
            **num_batches: number of batches
        """
        warn_msg = ("The default output of predictor.decode() and predictor.run() " +
                    "is now ```nn_output, coords``` instead of ```nn_input, (nn_output, coords)```")
        warnings.warn(warn_msg, UserWarning)
        image_data = self.preprocess(image_data)
        n, _, w, h = image_data.shape
        num_batches = kwargs.get("num_batches")
        if num_batches is None:
            if w >= 256 or h >= 256:
                num_batches = len(image_data)
            else:
                num_batches = 10
        batch_size = len(image_data) // num_batches
        if batch_size < 1:
            num_batches = batch_size = 1
        decoded_imgs = np.zeros((n, w, h, self.nb_classes))
        for i in range(num_batches):
            if self.verbose:
                print("\rBatch {}/{}".format(i+1, num_batches), end="")
            images_i = image_data[i*batch_size:(i+1)*batch_size]
            decoded_i = self.predict(images_i)
            decoded_imgs[i*batch_size:(i+1)*batch_size] = decoded_i
        images_i = image_data[(i+1)*batch_size:]
        if len(images_i) > 0:
            decoded_i = self.predict(images_i)
            decoded_imgs[(i+1)*batch_size:] = decoded_i
        if return_image:
            image_data = image_data.permute(0, 2, 3, 1).numpy()
            return image_data, decoded_imgs
        return decoded_imgs

    def run(self,
            image_data: np.ndarray,
            **kwargs: int) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
        """
        Make prediction with a trained model and calculate coordinates

        Args:
            image_data (2D or 3D numpy array):
                Image stack or a single image (all greyscale)
            **num_batches: number of batches (Default: 10)
        """
        start_time = time.time()
        images, decoded_imgs = self.decode(
            image_data, return_image=True, **kwargs)
        loc = locator(self.thresh, refine=self.refine, d=self.d)
        coordinates = loc.run(decoded_imgs, images)
        if self.verbose:
            n_images_str = " image was " if decoded_imgs.shape[0] == 1 else " images were "
            print("\n" + str(decoded_imgs.shape[0])
                  + n_images_str + "decoded in approximately "
                  + str(np.around(time.time() - start_time, decimals=4))
                  + ' seconds')
        return decoded_imgs, coordinates


class locator:
    """
    Transforms pixel data from NN output into coordinate data

    Args:
        decoded_imgs (4D numpy array):
            Output of a neural network
        threshold (float):
            Value at which the neural network output is thresholded
        dist_edge (int):
            Distance within image boundaries not to consider
        dim_order (str):
            'channel_last' or 'channel_first' (Default: 'channel last')

    Example:

        >>> # Transform output of atomnet.predictor to atomic classes and coordinates
        >>> coordinates = atomnet.locator(dist_edge=10, refine=False).run(nn_output)
    """
    def __init__(self,
                 threshold: float = 0.5,
                 dist_edge: int = 5,
                 dim_order: str = 'channel_last',
                 **kwargs: Union[bool, float]) -> None:
        """
        Initialize locator parameters
        """
        self.dim_order = dim_order
        self.threshold = threshold
        self.dist_edge = dist_edge
        self.refine = kwargs.get("refine")
        self.d = kwargs.get("d")

    def preprocess(self, nn_output: np.ndarray) -> np.ndarray:
        """
        Prepares data for coordinates extraction
        """
        if nn_output.shape[-1] == 1:  # Add background class for 1-channel data
            nn_output_b = 1 - nn_output
            nn_output = np.concatenate(
                (nn_output, nn_output_b), axis=3)
        if self.dim_order == 'channel_first':  # make channel dim the last dim
            nn_output = np.transpose(nn_output, (0, 2, 3, 1))
        elif self.dim_order == 'channel_last':
            pass
        else:
            raise NotImplementedError(
                'For dim_order, use "channel_first"',
                'or "channel_last" (e.g. tensorflow)')
        return nn_output

    def run(self, nn_output: np.ndarray, *args: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Extract all atomic coordinates in image
        via CoM method & store data as a dictionary
        (key: frame number)

        Args:
            nn_output (4D numpy array):
                Output (prediction) of a neural network
            *args: 4D input into a neural network (experimental data)

        """
        nn_output = self.preprocess(nn_output)
        d_coord = {}
        for i, decoded_img in enumerate(nn_output):
            coordinates = np.empty((0, 2))
            category = np.empty((0, 1))
            # we assume that class 'background' is always the last one
            for ch in range(decoded_img.shape[2]-1):
                decoded_img_c = cv_thresh(
                    decoded_img[:, :, ch], self.threshold)
                coord = find_com(decoded_img_c)
                coord_ch = self.rem_edge_coord(coord, *nn_output.shape[1:3])
                category_ch = np.zeros((coord_ch.shape[0], 1)) + ch
                coordinates = np.append(coordinates, coord_ch, axis=0)
                category = np.append(category, category_ch, axis=0)
            d_coord[i] = np.concatenate((coordinates, category), axis=1)
        if self.refine:
            if len(args) > 0:
                imgdata = args[0]
            else:
                raise AssertionError("Pass input image(s) for coordinates refinement")
            print('\n\rRefining atomic positions... ', end="")
            d_coord_r = {}
            for i, (img, coord) in enumerate(zip(imgdata, d_coord.values())):
                d_coord_r[i] = peak_refinement(img[..., 0], coord, self.d)
            print("Done")
            return d_coord_r
        return d_coord

    def rem_edge_coord(self, coordinates: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Removes coordinates at the image edges
        """

        def coord_edges(coordinates, h, w):
            return [coordinates[0] > h - self.dist_edge,
                    coordinates[0] < self.dist_edge,
                    coordinates[1] > w - self.dist_edge,
                    coordinates[1] < self.dist_edge]

        coord_to_rem = [
                        idx for idx, c in enumerate(coordinates)
                        if any(coord_edges(c, h, w))
                        ]
        coord_to_rem = np.array(coord_to_rem, dtype=int)
        coordinates = np.delete(coordinates, coord_to_rem, axis=0)
        return coordinates


class ensemble_predictor:

    """
    Predicts mean and variance/uncertainty in image pixels
    and (optionally) coordinates with ensemble of models

    Args:
        predictive_model (pytorch object):
            model skeleton (can have randomly initialized weights)
        ensemble (Dict):
            nested dictionary with weights of each model in the ensemble
        calculate_coordinates (bool):
            computes atomic coordinates for each prediction
        **eps (float): DBSCAN epsilon for clustering coordinates
        **threshold (float):
            value at which a neural network output is thresholded
            for calculating coordinates
        **num_classes (float): number of classes in the classification scheme
        **downsample_factor (int): image downsampling (max_size / min_size) in NN
    """

    def __init__(self,
                 predictive_model: Type[torch.nn.Module],
                 ensemble: Dict[int, Dict[str, torch.Tensor]],
                 calculate_coordinates: bool = False, **kwargs: Dict) -> None:

        self.use_gpu = torch.cuda.is_available()

        self.ensemble = ensemble
        self.predictive_model = copy.deepcopy(predictive_model)

        self.num_classes = kwargs.get("num_classes")
        if self.num_classes is None:
            hookF = [Hook(layer[1]) for layer in list(predictive_model._modules.items())]
            mock_forward(predictive_model)
            self.num_classes = [hook.output.shape for hook in hookF][-1][1]
        self.downsample_factor = kwargs.get("downsample_factor")
        if self.downsample_factor is None:
            hookF = [Hook(layer[1]) for layer in list(predictive_model._modules.items())]
            mock_forward(predictive_model)
            imsize = [hook.output.shape[-1] for hook in hookF]
            self.downsample_factor = max(imsize) / min(imsize)

        self.calculate_coordinates = calculate_coordinates
        if self.calculate_coordinates:
            self.eps = kwargs.get("eps", 0.5)
            self.thresh = kwargs.get("threshold", 0.5)

    def preprocess_data(self, imgdata: np.ndarray):
        """
        Basic preprocessing of input images
        """
        if np.ndim(imgdata) == 2:
            imgdata = np.expand_dims(imgdata, axis=0)
        if imgdata.ndim == 4 and imgdata.shape[-1] == 1:
            imgdata = imgdata[..., 0]
        elif imgdata.ndim == 4 and imgdata.shape[1] == 1:
            imgdata = imgdata[:, 0, ...]
        imgdata = img_pad(imgdata, self.downsample_factor)
        return imgdata

    def predict(self,
                x_new: np.ndarray
                ) -> Tuple[Tuple[np.ndarray, np.ndarray],
                           Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]]:
        """
        Runs ensemble decoding for a single batch

        Args:
            x_new (numpy array): batch of images
        """
        x_new = self.preprocess_data(x_new)
        batch_dim, img_h, img_w = x_new.shape
        nn_output_ensemble = np.zeros((
            len(self.ensemble), batch_dim, img_h, img_w, self.num_classes))
        for i, w in self.ensemble.items():
            self.predictive_model.load_state_dict(w)
            self.predictive_model.eval()
            nn_output = predictor(
                self.predictive_model,
                nb_classes=self.num_classes,
                downsampling=self.downsample_factor,
                use_gpu=self.use_gpu,
                verbose=False).decode(x_new, num_batches=1)
            nn_output_ensemble[i] = nn_output
        nn_output_mean = np.mean(nn_output_ensemble, axis=0)
        nn_output_var = np.var(nn_output_ensemble, axis=0)
        coord_mean, coord_var = None, None
        if self.calculate_coordinates:
            coord_mean, coord_var = ensemble_locate(
                nn_output_ensemble, eps=self.eps, threshold=self.thresh)
        return (nn_output_mean, nn_output_var), (coord_mean, coord_var)

    def run(self,
            imgdata: np.ndarray,
            **kwargs: Dict
            ) -> Tuple[Tuple[np.ndarray, np.ndarray],
                       Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]]:
        """
        Runs decoding with ensemble of models in a batch-by-batch fashion

        Args:
            imgdata (numpy array): 2D experimental image or 3D image stack
            **num_batches (int): number of batches
                (for large datasets to make sure everything fits into memory)
        """
        imgdata = self.preprocess_data(imgdata)
        num_batches = kwargs.get("num_batches", 10)
        batch_size = len(imgdata) // num_batches
        if batch_size < 1:
            batch_size = num_batches = 1
        img_mu_all = np.zeros((*imgdata.shape[0:3], self.num_classes))
        img_var_all = np.zeros(img_mu_all.shape)
        coord_mu_all, coord_var_all = None, None
        if self.calculate_coordinates:
            coord_mu_all = np.zeros((imgdata.shape[0], 3))
            coord_var_all = np.zeros(coord_mu_all.shape)

        for i in range(num_batches):
            print("\rBatch {}/{}".format(i+1, num_batches), end="")
            x_i = imgdata[i*batch_size:(i+1)*batch_size]
            (img_mu_i, img_var_i), (coord_mu_i, coord_var_i) = self.predict(x_i)
            img_mu_all[i*batch_size:(i+1)*batch_size] = img_mu_i
            img_var_all[i*batch_size:(i+1)*batch_size] = img_var_i
            if self.calculate_coordinates:
                coord_mu_all[i*batch_size:(i+1)*batch_size] = coord_mu_i
                coord_var_all[i*batch_size:(i+1)*batch_size] = coord_var_i
        x_i = imgdata[(i+1)*batch_size:]
        if len(x_i) > 0:
            (img_mu_i, img_var_i), (coord_mu_i, coord_var_i) = self.predict(x_i)
            img_mu_all[(i+1)*batch_size:] = img_mu_i
            img_var_all[(i+1)*batch_size:] = img_var_i
            if self.calculate_coordinates:
                coord_mu_all[(i+1)*batch_size:] = coord_mu_i
                coord_var_all[(i+1)*batch_size:] = coord_var_i

        return (img_mu_all, img_var_all), (coord_mu_all, coord_var_all)


def ensemble_locate(nn_output_ensemble: np.ndarray,
                    **kwargs: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds coordinates for each ensemble predictions
    (basically, an atomnet.locator for ensembles)

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
            coord = locator(thresh).run(img[None, ...])
            coordinates[i2] = coord[0]
        _, coord_mean, coord_var = cluster_coord(coordinates, eps)
        coord_mean_all[i] = coord_mean
        coord_var_all[i] = coord_var
    return coord_mean_all, coord_var_all
