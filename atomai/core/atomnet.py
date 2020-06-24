"""
atomnet.py
==========

Module for training neural networks
and making predictions with trained models

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

import atomai.core.losses_metrics as losses_metrics_
from atomai.core.models import dilnet, dilUnet
from atomai.utils import (Hook, cv_thresh, find_com, gpu_usage_map, img_pad,
                          img_resize, mock_forward, peak_refinement, datatransform,
                          plot_losses, preprocess_training_data, torch_format,
                          unsqueeze_channels)

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
            Type of model to train ('dilUnet' or 'dilnet'). See atomai.models
            for more details.
        seed (int):
            Deterministic mode for model training
        batch_seed (int):
            Separate seed for generating a sequence of batches
            for training/testing. Equal to 'seed' if set to None (default)
        **batch_size (int):
            Size of training and test batches
        **use_batchnorm (bool):
            Apply batch normalization after each convolutional layer
        **use_dropouts (bool):
            Apply dropouts in the three inner blocks in the middle of a network
        **loss (str):
            Type of loss for model training ('ce', 'dice' or 'focal')
        **upsampling_mode (str):
            "bilinear" or "nearest" upsampling method
        **nb_filters (int):
            Number of convolutional filters in the first convolutional block
            (this number doubles in the consequtive block(s),
            see definition of dilUnet and dilnet models for details)
        **with_dilation (bool):
            Use dilated convolutions in the bottleneck of dilUnet
        **layers (list):
            List with a number of layers in each block.
            For U-Net the first 4 elements in the list
            are used to determine the number of layers
            in each block of the encoder (incluidng bottleneck layer),
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
            or gauss=[20, 60])

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
                 images_all,
                 labels_all,
                 images_test_all,
                 labels_test_all,
                 training_cycles,
                 model_type='dilUnet',
                 seed=1,
                 batch_seed=None,
                 **kwargs):
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
        loss = kwargs.get('loss', "dice")
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

    def dataloader(self, batch_num, mode='train'):
        """
        Generates a batch of training/test images
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

    def train_step(self, img, lbl):
        """
        Propagates image(s) through a network to get model's prediction
        and compares predicted value with ground truth; then performs
        backpropagation to compute gradients and optimizes weights.
        """
        self.net.train()
        self.optimizer.zero_grad()
        prob = self.net.forward(img)
        loss = self.criterion(prob, lbl)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_step(self, img, lbl):
        """
        Forward path for test data with deactivated autograd engine
        """
        self.net.eval()
        with torch.no_grad():
            prob = self.net.forward(img)
            loss = self.criterion(prob, lbl)
        return loss.item()

    def run(self):
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
            # Save model weights (if test loss decreased)
            if e > 0 and self.test_loss[-1] < min(self.test_loss[: -1]):
                torch.save(self.meta_state_dict,
                   os.path.join(self.savedir,
                   self.savename+'_metadict_best_test_weights.tar'))
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
        image_data (2D or 3D numpy array):
            Image stack or a single image (all greyscale)
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
        >>> nn_input, (nn_output, coordinates) = atomnet.predictor(expdata, trained_model).run()

    """
    def __init__(self,
                 image_data,
                 trained_model,
                 refine=False,
                 resize=None,
                 use_gpu=False,
                 logits=True,
                 seed=1,
                 **kwargs):
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
        downsampling = kwargs.get('downsampling', None)
        if downsampling is None:
            hookF = [Hook(layer[1]) for layer in list(model._modules.items())]
            mock_forward(model)
            imsize = [hook.output.shape[-1] for hook in hookF]
            downsampling = max(imsize) / min(imsize)
        self.model = model
        if use_gpu:
            self.model.cuda()
        else:
            self.model.cpu()
        if image_data.ndim == 2:
            image_data = np.expand_dims(image_data, axis=0)
        if resize is not None:
            image_data = img_resize(image_data, resize)
        image_data = img_pad(image_data, downsampling)
        self.image_data = torch_format(image_data)
        self.logits = logits
        self.refine = refine
        self.d = kwargs.get("d", None)
        self.thresh = kwargs.get("thresh", .5)
        self.use_gpu = use_gpu
        self.verbose = kwargs.get("verbose", True)

    def predict(self, images):
        """
        Returns 'probability' of each pixel
        in image(s) belonging to an atom/defect
        """
        if self.use_gpu:
            images = images.cuda()
        self.model.eval()
        with torch.no_grad():
            prob = self.model.forward(images)
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

    def decode(self):
        """
        Make prediction
        """
        if self.image_data.shape[0] < 20 and min(self.image_data.shape[2:4]) < 512:
            decoded_imgs = self.predict(self.image_data)
        else:
            n, _, w, h = self.image_data.shape
            decoded_imgs = np.zeros((n, w, h, self.nb_classes))
            for i in range(n):
                decoded_imgs[i, :, :, :] = self.predict(self.image_data[i:i+1])
        images_numpy = self.image_data.permute(0, 2, 3, 1).numpy()
        return images_numpy, decoded_imgs

    def run(self):
        start_time = time.time()
        images, decoded_imgs = self.decode()
        coordinates = locator(decoded_imgs, self.thresh).run()
        if self.verbose:
            n_images_str = " image was " if decoded_imgs.shape[0] == 1 else " images were "
            print(str(decoded_imgs.shape[0])
                  + n_images_str + "decoded in approximately "
                  + str(np.around(time.time() - start_time, decimals=4))
                  + ' seconds')
        if self.refine:
            print('\rRefining atomic positions... ', end="")
            coordinates_r = {}
            for i, (img, coord) in enumerate(zip(images, coordinates.values())):
                coordinates_r[i] = peak_refinement(img[...,0], coord, self.d)
            print("Done")
            return images, (decoded_imgs, coordinates_r)
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
        >>> coordinates = atomnet.locator(nn_output).run()
    """
    def __init__(self,
                 nn_output,
                 threshold=0.5,
                 dist_edge=5,
                 dim_order='channel_last'):

        if nn_output.shape[-1] == 1: # Add background class for 1-channel data
            nn_output_b = 1 - nn_output
            nn_output = np.concatenate(
                (nn_output[:, :, :, None], nn_output_b[:, :, :, None]), axis=3)
        if dim_order == 'channel_first':  # make channel dim the last dim
            nn_output = np.transpose(nn_output, (0, 2, 3, 1))
        elif dim_order == 'channel_last':
            pass
        else:
            raise NotImplementedError(
                'For dim_order, use "channel_first"',
                'or "channel_last" (e.g. tensorflow)')
        self.nn_output = nn_output
        self.threshold = threshold
        self.dist_edge = dist_edge

    def run(self):
        """
        Extract all atomic coordinates in image
        via CoM method & store data as a dictionary
        (key: frame number)
        """

        d_coord = {}
        for i, decoded_img in enumerate(self.nn_output):
            coordinates = np.empty((0, 2))
            category = np.empty((0, 1))
            # we assume that class 'background' is always the last one
            for ch in range(decoded_img.shape[2]-1):
                decoded_img_c = cv_thresh(
                    decoded_img[:, :, ch], self.threshold)
                coord = find_com(decoded_img_c)
                coord_ch = self.rem_edge_coord(coord)
                category_ch = np.zeros((coord_ch.shape[0], 1)) + ch
                coordinates = np.append(coordinates, coord_ch, axis=0)
                category = np.append(category, category_ch, axis=0)
            d_coord[i] = np.concatenate((coordinates, category), axis=1)
        return d_coord

    def rem_edge_coord(self, coordinates):
        """
        Removes coordinates at the image edges
        """

        def coord_edges(coordinates, w, h):
            return [coordinates[0] > w - self.dist_edge,
                    coordinates[0] < self.dist_edge,
                    coordinates[1] > h - self.dist_edge,
                    coordinates[1] < self.dist_edge]

        w, h = self.nn_output.shape[1:3]
        coord_to_rem = [
                        idx for idx, c in enumerate(coordinates)
                        if any(coord_edges(c, w, h))
                        ]
        coord_to_rem = np.array(coord_to_rem, dtype=int)
        coordinates = np.delete(coordinates, coord_to_rem, axis=0)
        return coordinates
