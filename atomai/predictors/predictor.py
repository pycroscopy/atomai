"""
predictor.py
============

Module for making predictions with pre-trained neural networks,
including semantic segmentation models and im2spec models.

Created by Maxim Ziatdinov (maxim.ziatdinov@ai4microscopy.com)
"""

import time
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from atomai.utils import (cv_thresh, find_com, get_downsample_factor,
                          get_nb_classes, img_pad, img_resize, peak_refinement,
                          set_train_rng, torch_format_image,
                          torch_format_spectra)


class BasePredictor:
    """
    Base predictor class
    """
    def __init__(self,
                 model: Type[torch.nn.Module] = None,
                 use_gpu: bool = False,
                 **kwargs: Union[bool, str]) -> None:
        """
        Initialize predictor

        Args:
            model: trained pytorch model
            use_gpu: Use GPU accelerator (Default: False)
            **device: CUDA device, e.g. 'cuda:0'
        """
        self.model = model
        self.device = "cpu"
        if use_gpu and torch.cuda.is_available():
            if kwargs.get("device") is None:
                self.device = "cuda"
            else:
                self.device = kwargs.get("device")
        if self.model is not None:
            self.model.to(self.device)
        self.verbose = kwargs.get("verbose", False)

    def preprocess(self,
                   data: Union[torch.Tensor, np.ndarray]
                   ) -> None:
        """
        Preprocess input data
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        return data

    def _model2device(self, device: str = None) -> None:
        if device is None:
            device = self.device
        self.model.to(device)

    def _data2device(self,
                     data: torch.Tensor,
                     device: str = None) -> torch.Tensor:
        if device is None:
            device = self.device
        data = data.to(device)
        return data

    def forward_(self, xnew: torch.Tensor) -> torch.Tensor:
        """
        Pass data through a trained neural network
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(xnew.to(self.device))
        return out

    def batch_predict(self,
                      data: torch.Tensor,
                      out_shape: Tuple[int],
                      num_batches: int) -> torch.Tensor:
        """
        Make a prediction batch-by-batch (for larger datasets)
        """
        batch_size = len(data) // num_batches
        if batch_size < 1:
            num_batches = batch_size = 1
        #prediction_all = np.zeros(shape=out_shape)
        prediction_all = torch.zeros(out_shape)
        for i in range(num_batches):
            if self.verbose:
                print("\rBatch {}/{}".format(i+1, num_batches), end="")
            data_i = data[i*batch_size:(i+1)*batch_size]
            prediction_i = self.forward_(data_i)
            # We put predictions on cpu since the major point of batch-by-batch
            # prediction is to not run out of the GPU memory
            prediction_all[i*batch_size:(i+1)*batch_size] = prediction_i.cpu()
        data_i = data[(i+1)*batch_size:]
        if len(data_i) > 0:
            prediction_i = self.forward_(data_i)
            prediction_all[(i+1)*batch_size:] = prediction_i.cpu()
        return prediction_all

    def predict(self,
                data: torch.Tensor,
                out_shape: Tuple[int] = None,
                num_batches: int = 1) -> torch.Tensor:
        """
        Make a prediction on the new data with a trained model
        """
        if out_shape is None:
            out_shape = data.shape
        else:
            out_shape = (data.shape[0], *out_shape)
        data = self.preprocess(data)
        prediction = self.batch_predict(data, out_shape, num_batches)
        return prediction


class SegPredictor(BasePredictor):
    """
    Prediction with a trained fully convolutional neural network

    Args:
        trained_model:
            Trained pytorch model (skeleton+weights)
        refine:
            Atomic positions refinement with 2d Gaussian peak fitting
        resize:
            Target dimensions for optional image(s) resizing
        use_gpu:
            Use gpu device for inference
        logits:
            Indicates that the image data is passed through
            a softmax/sigmoid layer when set to False
            (logits=True for AtomAI models)
        **thresh (float):
            value between 0 and 1 for thresholding the NN output
            (Default: 0.5)
        **d (int):
            half-side of a square around each atomic position used
            for refinement with 2d Gaussian peak fitting. Defaults to 1/4
            of average nearest neighbor atomic distance
        **nb_classes (int):
            Number of classes in the model
        **downsampling (int or float):
            Downsampling factor (equal to :math:`2^n` where *n* is a number
            of pooling operations)

    Example:

        >>> # Here we load new experimental data (as 2D or 3D numpy array)
        >>> expdata = np.load('expdata-test.npy')
        >>> # Get prediction from a trained model
        >>> pseg = atomnet.SegPredictor(trained_model)
        >>> nn_output, coords = pseg.run(expdata)

    """
    def __init__(self,
                 trained_model: Type[torch.nn.Module],
                 refine: bool = False,
                 resize: Union[Tuple, List] = None,
                 use_gpu: bool = False,
                 logits: bool = True,
                 **kwargs: Union[int, float, bool]) -> None:
        """
        Initializes predictive object
        """
        super(SegPredictor, self).__init__(trained_model, use_gpu)
        set_train_rng(1)
        self.nb_classes = kwargs.get('nb_classes', None)
        if self.nb_classes is None:
            self.nb_classes = get_nb_classes(trained_model)
        self.downsampling = kwargs.get('downsampling', None)
        if self.downsampling is None:
            self.downsampling = get_downsample_factor(trained_model)

        self.resize = resize
        self.logits = logits
        self.refine = refine
        self.d = kwargs.get("d", None)
        self.thresh = kwargs.get("thresh", .5)
        self.use_gpu = use_gpu
        self.verbose = kwargs.get("verbose", True)

    def preprocess(self,
                   image_data: np.ndarray,
                   norm: bool = True) -> torch.Tensor:
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
        image_data = torch_format_image(image_data, norm)
        return image_data

    def forward_(self, images: torch.Tensor) -> np.ndarray:
        """
        Returns 'probability' of each pixel
        in image(s) belonging to an atom/defect
        """
        images = images.to(self.device)
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
        prob = prob.permute(0, 2, 3, 1)  # reshape to have channel as a last dim
        images = images.cpu()
        prob = prob.cpu()
        return prob

    def predict(self,
                image_data: np.ndarray,
                return_image: bool = False,
                **kwargs: int) -> Tuple[np.ndarray]:
        """
        Make prediction

        Args:
            image_data:
                3D image stack or a single 2D image (all greyscale)
            return_image:
                Returns images used as input into NN
            **num_batches: number of batches
            **norm (bool): Normalize data to (0, 1) during pre-processing
        """
        image_data = self.preprocess(
            image_data, kwargs.get("norm", True))
        n, _, w, h = image_data.shape
        num_batches = kwargs.get("num_batches")
        if num_batches is None:
            if w >= 256 or h >= 256:
                num_batches = len(image_data)
            else:
                num_batches = 10
        segmented_imgs = self.batch_predict(
            image_data, (n, w, h, self.nb_classes), num_batches)
        if return_image:
            image_data = image_data.permute(0, 2, 3, 1).numpy()
            return image_data, segmented_imgs.numpy()
        return segmented_imgs.numpy()

    def run(self,
            image_data: np.ndarray,
            compute_coords=True,
            **kwargs: int) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Make prediction with a trained model and calculate coordinates

        Args:
            image_data (2D or 3D numpy array):
                Image stack or a single image (all greyscale)
            compute_coords (bool):
                Computes centers of the mass of individual blobs
                in the segmented images (Default: True)
            **num_batches (int):
                number of batches for batch-by-batch prediction
                which ensures that one doesn't run out of memory
                (Default: 10)
            **norm (bool): Normalize data to (0, 1) during pre-processing
        """
        start_time = time.time()
        if not compute_coords:
            decoded_imgs = self.predict(image_data, **kwargs)
            return decoded_imgs
        images, decoded_imgs = self.predict(
            image_data, return_image=True, **kwargs)
        loc = Locator(self.thresh, refine=self.refine, d=self.d)
        coordinates = loc.run(decoded_imgs, images)
        if self.verbose:
            n_images_str = " image was " if decoded_imgs.shape[0] == 1 else " images were "
            print("\n" + str(decoded_imgs.shape[0])
                  + n_images_str + "decoded in approximately "
                  + str(np.around(time.time() - start_time, decimals=4))
                  + ' seconds')
        return decoded_imgs, coordinates


class ImSpecPredictor(BasePredictor):
    """
    Prediction with a trained im2spec or spec2im model

    Args:
        trained_model:
            Pre-trained neural network
        output_dim:
            Output dimensions. For im2spec, the output_dim is (length,).
            For spec2im, the output_dim is (height, width)
        use_gpu:
            Use GPU accelration for prediction
        verbose:
            Verbosity
    
    Example:

        >>> # Predict spectra from images with pretrained im2spec model
        >>> out_dim = (16,)  # spectra length
        >>> prediction = ImSpecPredictor(trained_model, out_dim).run(data)
    """
    def __init__(self,
                 trained_model: Type[torch.nn.Module],
                 output_dim: Tuple[int],
                 use_gpu: bool = False,
                 **kwargs: str) -> None:
        """
        Initialize predictor
        """
        super(ImSpecPredictor, self).__init__(trained_model, use_gpu)
        if isinstance(output_dim, int):
            output_dim = (output_dim,)
        if len(output_dim) not in [1, 2]:
            raise ValueError("output_dim must be a two-value tuple for images" +
                             " and a single-value tuple for spectra")
        set_train_rng(1)
        self.output_dim = output_dim
        self.verbose = kwargs.get("verbose", True)

    def preprocess(self,
                   signal: np.ndarray,
                   norm: bool = True) -> torch.Tensor:
        """
        Preprocess input signal (images or spectra)
        """
        if len(self.output_dim) == 1:
            if signal.ndim == 2:
                signal = signal[np.newaxis, ...]
            signal = torch_format_image(signal, norm)
        elif len(self.output_dim) == 2:
            if signal.ndim == 1:
                signal = signal[np.newaxis, ...]
            signal = torch_format_spectra(signal, norm)
        return signal

    def predict(self,
                signal: np.ndarray,
                **kwargs: int) -> np.ndarray:
        """
        Predict spectra from images or vice versa

        Args:
            signal (numpy array): Input image/spectrum or batch of images/spectra
            **num_batches (int): number of batches (Default: 10)
            **norm (bool): Normalize data to (0, 1) during pre-processing
        """
        signal = self.preprocess(signal, kwargs.get("norm", True))
        num_batches = kwargs.get("num_batches", 10)
        output = self.batch_predict(
            signal, (len(signal), 1, *self.output_dim), num_batches)
        return output[:, 0].numpy()

    def run(self,
            signal: np.ndarray,
            **kwargs: int) -> np.ndarray:
        """
        Make prediction with a trained model

        Args:
            signal (numpy array): Input image/spectrum or batch of images/spectra
            **num_batches (int): number of batches (Default: 10)
            **norm (bool): Normalize data to (0, 1) during pre-processing
        """
        start_time = time.time()
        prediction = self.predict(signal, **kwargs)
        if self.verbose:
            if len(self.output_dim) == 1:
                str_ = " image was " if prediction.shape[0] == 1 else " images were "
            else:
                str_ = " spectrum was " if prediction.shape[0] == 1 else " spectra were "
            print("\n" + str(prediction.shape[0])
                  + str_ + "decoded in approximately "
                  + str(np.around(time.time() - start_time, decimals=4))
                  + ' seconds')
        return prediction


class Locator:
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
        >>> coordinates = locator(dist_edge=10, refine=False).run(nn_output)
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
