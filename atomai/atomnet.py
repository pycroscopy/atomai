"""
atomnet.py
==========

Module for training neural networks
and making predictions with trained models

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import copy
import os
import time
import warnings
from collections import OrderedDict
from typing import Dict, List, Tuple, Type, Union

import atomai.losses_metrics as losses_metrics_
import numpy as np
import torch
import torch.nn.functional as F
from atomai.nets import dilnet, dilUnet
from atomai.utils import (Hook, average_weights, cluster_coord, cv_thresh,
                          datatransform, find_com, gpu_usage_map, img_pad,
                          img_resize, mock_forward, peak_refinement,
                          plot_losses, preprocess_training_data, torch_format,
                          unsqueeze_channels)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", module="torch.nn.functional")


class trainer:
    """
    Class for training a fully convolutional neural network
    for semantic segmentation of noisy experimental data

    Args:
        images_all (list or dict or 4D numpy array):
            Training images in the form of list/dictionary of
            small 4D numpy arrays (batches) or larger 4D numpy array
            representing all the training images. For dictionary with N batches,
            the keys must be 0, 1, 2, ... *N*. Both small and large 4D numpy arrays
            represent 3D images :math:`(height \\times width \\times 1)` stacked
            along the zeroth ("batch") dimension.
        labels_all (list or dict or 4D numpy array):
            Training labels (aka ground truth aka masks) in the form of
            list/dictionary of small 3D (binary classification) or 4D (multiclass)
            numpy arrays or larger 4D (binary) / 3D (multiclass) numpy array
            containing all the training labels.
            For dictionary with N batches, the keys must be 0, 1, 2, ... *N*.
            Both small and large numpy arrays are 3D (binary) / 2D (multiclass) images
            stacked along the zeroth ("batch") dimension. The reason why in the
            multiclass case the images have 4 dimensions while the labels have only 3 dimensions
            is because of how the cross-entropy loss is calculated in PyTorch
            (see https://pytorch.org/docs/stable/nn.html#nllloss).
        images_test_all (list or dict or 4D numpy array):
            Test images in the form of list/dictionary of
            small 4D numpy arrays (batches) or larger 4D numpy array
            representing all the test images. For dictionary with N batches,
            the keys must be 0, 1, 2, ... *N*. Both small and large 4D numpy arrays
            represent 3D images :math:`(height \\times width \\times 1)` stacked
            along the zeroth ("batch") dimension.
        labels_test_all (list or dict or 4D numpy array):
            Test labels (aka ground truth aka masks) in the form of
            list/dictionary of small 3D (binary classification) or 4D (multiclass)
            numpy arrays or larger 4D (binary) / 3D (multiclass) numpy array
            containing all the test labels.
            For dictionary with N batches, the keys must be 0, 1, 2, ... *N*.
            Both small and large numpy arrays are 3D (binary) / 2D (multiclass) images
            stacked along the zeroth ("batch") dimenstion.
        training_cycles (int):
            Number of training 'epochs' (1 epoch == 1 batch)
        model_type (str):
            Type of model to train: 'dilUnet' or 'dilnet' (Default: 'dilUnet').
            See atomai.nets for more details.
        seed (int):
            Deterministic mode for model training (Default: 1)
        batch_seed (int):
            Separate seed for generating a sequence of batches
            for training/testing. Equal to 'seed' if set to None (default)
        **batch_size (int):
            Size of training and test batches
        **use_batchnorm (bool):
            Apply batch normalization after each convolutional layer
            (Default: True)
        **use_dropouts (bool):
            Apply dropouts in the three inner blocks in the middle of a network
            (Default: False)
        **loss (str):
            Type of loss for model training ('ce', 'dice' or 'focal')
            (Default: 'ce')
        **upsampling_mode (str):
            "bilinear" or "nearest" upsampling method (Default: "bilinear")
        **nb_filters (int):
            Number of convolutional filters in the first convolutional block
            (this number doubles in the consequtive block(s),
            see definition of dilUnet and dilnet models for details)
        **with_dilation (bool):
            Use dilated convolutions in the bottleneck of dilUnet
            (Default: True)
        **layers (list):
            List with a number of layers in each block.
            For U-Net the first 4 elements in the list
            are used to determine the number of layers
            in each block of the encoder (including bottleneck layer),
            and the number of layers in the decoder  is chosen accordingly
            (to maintain symmetry between encoder and decoder)
        **print_loss (int):
            Prints loss every *n*-th epoch
        **savedir (str):
            Directory to automatically save intermediate and final weights
        **savename (str):
            Filename for model weights
            (appended with "_test_weights_best.pt" and "_weights_final.pt")
        **plot_training_history (bool):
            Plots training and test curves vs epochs at the end of training
        **kwargs:
            One can also pass kwargs for utils.datatransform class
            to perform the augmentation "on-the-fly" (e.g. rotation=True,
            gauss=[20, 60], ...)

    Example:

    >>> # Load 4 numpy arrays with training and test data
    >>> dataset = np.load('training_data.npz')
    >>> images_all = dataset['X_train']
    >>> labels_all = dataset['y_train']
    >>> images_test_all = dataset['X_test']
    >>> labels_test_all = dataset['y_test']
    >>> # Train a model
    >>> netr = atomnet.trainer(
    >>>     images_all, labels_all,
    >>>     images_test_all, labels_test_all,
    >>>     training_cycles=500)
    >>> trained_model = netr.run()
    """
    def __init__(self,
                 images_all: Union[np.ndarray, List[np.ndarray], Dict[int, np.ndarray]],
                 labels_all: Union[np.ndarray, List[np.ndarray], Dict[int, np.ndarray]],
                 images_test_all: Union[np.ndarray, List[np.ndarray], Dict[int, np.ndarray]],
                 labels_test_all: Union[np.ndarray, List[np.ndarray], Dict[int, np.ndarray]],
                 training_cycles: int,
                 model_type: str = 'dilUnet',
                 seed: int = 1,
                 batch_seed: int = None,
                 **kwargs: Union[int, List, str, bool]) -> None:
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        if batch_seed is None:
            np.random.seed(seed)
        else:
            np.random.seed(batch_seed)

        self.batch_size = kwargs.get("batch_size", 32)
        (self.images_all, self.labels_all,
         self.images_test_all, self.labels_test_all,
         self.num_classes) = preprocess_training_data(
                                images_all, labels_all,
                                images_test_all, labels_test_all,
                                self.batch_size)
        use_batchnorm = kwargs.get('use_batchnorm', True)
        use_dropouts = kwargs.get('use_dropouts', False)
        upsampling = kwargs.get('upsampling', "bilinear")
        if model_type == 'dilUnet':
            with_dilation = kwargs.get('with_dilation', True)
            nb_filters = kwargs.get('nb_filters', 16)
            layers = kwargs.get("layers", [1, 2, 2, 3])
            self.net = dilUnet(
                self.num_classes, nb_filters, use_dropouts,
                use_batchnorm, upsampling, with_dilation,
                layers=layers
            )
        elif model_type == 'dilnet':
            nb_filters = kwargs.get('nb_filters', 25)
            layers = kwargs.get("layers", [1, 3, 3, 3])
            self.net = dilnet(
                self.num_classes, nb_filters,
                use_dropouts, use_batchnorm, upsampling,
                layers=layers
            )
        else:
            raise NotImplementedError(
                "Currently implemented models are 'dilUnet' and 'dilnet'"
            )
        if torch.cuda.is_available():
            self.net.cuda()
        else:
            warnings.warn(
                "No GPU found. The training can be EXTREMELY slow",
                UserWarning
            )
        loss = kwargs.get('loss', "ce")
        if loss == 'dice':
            self.criterion = losses_metrics_.dice_loss()
        elif loss == 'focal':
            self.criterion = losses_metrics_.focal_loss()
        elif loss == 'ce' and self.num_classes == 1:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif loss == 'ce' and self.num_classes > 2:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(
                "Select Dice loss ('dice'), focal loss ('focal') or"
                " cross-entropy loss ('ce')"
            )
        self.batch_idx_train = np.random.randint(
            0, len(self.images_all), training_cycles)
        self.batch_idx_test = np.random.randint(
            0, len(self.images_test_all), training_cycles)
        auglist = ["zoom", "gauss", "jitter", "poisson", "contrast",
                   "salt_and_pepper", "blur", "resize", "rotation",
                   "background"]
        self.augdict = {k: kwargs[k] for k in auglist if k in kwargs.keys()}
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.training_cycles = training_cycles
        self.print_loss = kwargs.get("print_loss", 100)
        self.savedir = kwargs.get("savedir", "./")
        self.savename = kwargs.get("savename", "model")
        self.plot_training_history = kwargs.get("plot_training_history", True)
        self.train_loss, self.test_loss = [], []
        self.meta_state_dict = {
            'model_type': model_type,
            'batchnorm': use_batchnorm,
            'dropout': use_dropouts,
            'upsampling': upsampling,
            'nb_filters': nb_filters,
            'layers': layers,
            'nb_classes': self.num_classes,
            'weights': self.net.state_dict()
        }
        if "with_dilation" in locals():
            self.meta_state_dict["with_dilation"] = with_dilation

    def dataloader(self, batch_num: int, mode: str = 'train') -> Tuple[torch.Tensor]:
        """
        Generates 2 batches images (training and test)
        """
        # Generate batch of training images with corresponding ground truth
        if mode == 'test':
            images = self.images_test_all[batch_num][:self.batch_size]
            labels = self.labels_test_all[batch_num][:self.batch_size]
        else:
            images = self.images_all[batch_num][:self.batch_size]
            labels = self.labels_all[batch_num][:self.batch_size]
        # "Augment" data if applicable
        if len(self.augdict) > 0:
            dt = datatransform(
                self.num_classes, "channel_first", 'channel_first',
                True, len(self.train_loss), **self.augdict)
            images, labels = dt.run(
                images[:, 0, ...], unsqueeze_channels(labels, self.num_classes))
        # Transform images and ground truth to torch tensors and move to GPU
        images = torch.from_numpy(images).float()
        if self.num_classes == 1:
            labels = torch.from_numpy(labels).float()
        else:
            labels = torch.from_numpy(labels).long()
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        return images, labels

    def train_step(self, img: torch.Tensor, lbl: torch.Tensor) -> float:
        """
        Propagates image(s) through a network to get model's prediction
        and compares predicted value with ground truth; then performs
        backpropagation to compute gradients and optimizes weights.
        """
        self.net.train()
        self.optimizer.zero_grad()
        prob = self.net(img)
        loss = self.criterion(prob, lbl)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_step(self, img: torch.Tensor, lbl: torch.Tensor) -> float:
        """
        Forward path for test data with deactivated autograd engine
        """
        self.net.eval()
        with torch.no_grad():
            prob = self.net(img)
            loss = self.criterion(prob, lbl)
        return loss.item()

    def run(self) -> None:
        """
        Trains a neural network for *N* epochs by passing a single pair of
        training images and labels and a single pair of test images
        and labels at each epoch. Saves the best (on test data)
        and final model weights.
        """
        for e in range(self.training_cycles):
            # Get training images/labels
            images, labels = self.dataloader(
                self.batch_idx_train[e], mode='train')
            # Training step
            loss = self.train_step(images, labels)
            self.train_loss.append(loss)
            images_, labels_ = self.dataloader(
                self.batch_idx_test[e], mode='test')
            loss_ = self.test_step(images_, labels_)
            self.test_loss.append(loss_)
            # Print loss info
            if e == 0 or (e+1) % self.print_loss == 0:
                if torch.cuda.is_available():
                    gpu_usage = gpu_usage_map(torch.cuda.current_device())
                else:
                    gpu_usage = ['N/A ', ' N/A']
                print('Epoch {} ...'.format(e+1),
                      'Training loss: {} ...'.format(
                          np.around(self.train_loss[-1], 4)),
                      'Test loss: {} ...'.format(
                    np.around(self.test_loss[-1], 4)),
                      'GPU memory usage: {}/{}'.format(
                          gpu_usage[0], gpu_usage[1]))
        # Save final model weights
        torch.save(self.meta_state_dict,
                   os.path.join(self.savedir,
                   self.savename+'_metadict_final_weights.tar'))
        # Run evaluation (by passing all the test data) on the final model
        running_loss_test = 0
        for idx in range(len(self.images_test_all)):
            images_, labels_ = self.dataloader(idx, mode='test')
            loss = self.test_step(images_, labels_)
            running_loss_test += loss
        print('Model (final state) evaluation loss:',
              np.around(running_loss_test / len(self.images_test_all), 4))
        if self.plot_training_history:
            plot_losses(self.train_loss, self.test_loss)
        return self.net


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
        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
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
        if use_gpu:
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
            image_data = np.expand_dims(image_data, axis=0)
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
        if self.use_gpu:
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
               **kwargs: int) -> Tuple[np.ndarray]:
        """
        Make prediction

        Args:
            image_data (2D or 3D numpy array):
                Image stack or a single image (all greyscale)
            **num_batches: number of batches
        """
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
        images_data = image_data.permute(0, 2, 3, 1).numpy()
        return images_data, decoded_imgs

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
        images, decoded_imgs = self.decode(image_data, **kwargs)
        loc = locator(self.thresh, refine=self.refine, d=self.d)
        coordinates = loc.run(decoded_imgs, images)
        if self.verbose:
            n_images_str = " image was " if decoded_imgs.shape[0] == 1 else " images were "
            print("\n" + str(decoded_imgs.shape[0])
                  + n_images_str + "decoded in approximately "
                  + str(np.around(time.time() - start_time, decimals=4))
                  + ' seconds')
        return images, (decoded_imgs, coordinates)


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
                (nn_output[:, :, :, None], nn_output_b[:, :, :, None]), axis=3)
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

    def rem_edge_coord(self, coordinates: np.ndarray, w: int, h: int) -> np.ndarray:
        """
        Removes coordinates at the image edges
        """

        def coord_edges(coordinates, w, h):
            return [coordinates[0] > h - self.dist_edge,
                    coordinates[0] < self.dist_edge,
                    coordinates[1] > w - self.dist_edge,
                    coordinates[1] < self.dist_edge]

        coord_to_rem = [
                        idx for idx, c in enumerate(coordinates)
                        if any(coord_edges(c, w, h))
                        ]
        coord_to_rem = np.array(coord_to_rem, dtype=int)
        coordinates = np.delete(coordinates, coord_to_rem, axis=0)
        return coordinates


class ensemble_trainer:
    """
    Trains multiple deep learning models, each with its own unique trajectory

    Args:
        X_train (numpy array): Training images
        y_train (numpy array): Training labels (aka ground truth aka masks)
        X_test (numpy array): Test images
        y_test (numpy array): Test labels
        n_models (int): number of models in ensemble
        model_type (str): 'dilUnet' or 'dilnet'. See atomai.models for details
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
        trainer_base = trainer(
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
        filename = kwargs.get("filename")
        training_cycles_ensemble = kwargs.get("training_cycles_ensemble")
        if training_cycles_ensemble is not None:
            self.iter_ensemble = training_cycles_ensemble
        if filename is not None:
            self.filename = filename
        print('Training ensemble models:')
        for i in range(self.n_models):
            print('Ensemble model', i+1)
            trainer_i = trainer(
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

    def run(self) -> Tuple[Type[torch.nn.Module], Dict, Dict]:
        """
        Trains a baseline model and ensemble of models
        """
        basemodel = self.train_baseline()
        ensemble, ensemble_aver = self.train_ensemble(basemodel)
        return basemodel, ensemble, ensemble_aver


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
        self.predictive_model = predictive_model

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

    def predict(self,
                x_new: np.ndarray
                ) -> Tuple[Tuple[np.ndarray, np.ndarray],
                           Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]]:
        """
        Runs ensemble decoding for a single batch

        Args:
            x_new (numpy array): batch of images
        """
        x_new = img_pad(x_new, self.downsample_factor)
        batch_dim, img_h, img_w = x_new.shape
        nn_output_ensemble = np.zeros((
            len(self.ensemble), batch_dim, img_h, img_w, self.num_classes))
        for i, w in self.ensemble.items():
            self.predictive_model.load_state_dict(w)
            self.predictive_model.eval()
            _, nn_output = predictor(
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
        if np.ndim(imgdata) == 2:
            imgdata = np.expand_dims(imgdata, axis=0)
        imgdata = img_pad(imgdata, self.downsample_factor)
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
