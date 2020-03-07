"""
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

import atomai.losses as losses_
from atomai.models import dilnet, dilUnet
from atomai.utils import (Hook, cv_thresh, find_com, gpu_usage_map, img_pad,
                          img_resize, mock_forward, plot_losses, torch_format)

warnings.filterwarnings("ignore", module="torch.nn.functional")


class trainer:
    """
    Class for training a fully convolutional neural network
    for semantic segmentation of noisy experimental data

    Args:
        images_all: list / dict / 4D numpy array
            list or dictionary of 4D numpy arrays or 4D numpy array
            (3D image tensors stacked along the first dim)
            representing training images
        labels_all: list / dict / 4D numpy array
            list or dictionary of 3D numpy arrays or 
            4D (binary) / 3D (multiclass) numpy array
            where 3D / 2D image are tensors stacked along the first dim
            which represent training labels (aka masks aka ground truth)
        images_test_all: list / dict / 4D numpy array
            list or dictionary of 4D numpy arrays or 4D numpy array
            (3D image tensors stacked along the first dim)
            representing test images
        labels_test_all: list / dict / 4D numpy array
            list or dictionary of 3D numpy arrays or 
            4D (binary) / 3D (multiclass) numpy array
            where 3D / 2D image are tensors stacked along the first dim
            which represent test labels (aka masks aka ground truth)
        training_cycles: int
            number of training 'epochs' (1 epoch == 1 batch)
        model_type: str
            Type of mode to choose from the package ('dilUnet' or 'dilnet')
        seed: int
            deterministic mode for model training
        batch_seed: int
            separate seed for generating a sequence of batches
            for training/testing. Equal to 'seed' if set to None (default)

    **Kwargs:
        batch_size: int
            size of training and test batches
        use_batchnorm: bool
            Apply batch normalization after each convolutional layer
        use_dropouts: bool
            Apply dropouts in the three inner blocks in the middle of a network
        loss: str
            type of loss for model training ('ce', 'dice' or 'focal')
        upsampling mode: str
            "bilinear" or "nearest" upsampling method
        nb_filters: int
            number of convolutional filters in the first convolutional block
            (this number doubles in the consequtive block(s),
            see definition of dilUnet and dilnet models for details)
        with_dilation: bool
            use / not use dilated convolutions in the bottleneck of dilUnet
        print_loss: int
            prints loss every n-th epoch
        savedir: str
            directory to automatically save intermediate and final weights
        savename: str
            filename for model weights
            (appended with "_test_weights_best.pt" and "_weights_final.pt")
        plot_training_history: bool
            Plot training and test curves vs epochs at the end of training
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
        assert type(images_all) == type(labels_all)\
        == type(images_test_all) == type(labels_test_all),\
        "Provide all training and test data in the same format"
        if type(labels_all) == list:
            num_classes = max(set([len(np.unique(lab)) for lab in labels_all]))
        elif type(labels_all) == dict:
            num_classes = max(
                set([len(np.unique(lab)) for lab in labels_all.values()]))
        elif type(labels_all) == np.ndarray:
            batch_size = kwargs.get("batch_size", 32)
            n_train_batches, _ = np.divmod(labels_all.shape[0], batch_size)
            n_test_batches, _ = np.divmod(labels_test_all.shape[0], batch_size)
            images_all = np.split(
                images_all[:n_train_batches*batch_size], n_train_batches)
            labels_all = np.split(
                labels_all[:n_train_batches*batch_size], n_train_batches)
            images_test_all = np.split(
                images_test_all[:n_test_batches*batch_size], n_test_batches)
            labels_test_all = np.split(
                labels_test_all[:n_test_batches*batch_size], n_test_batches)
            num_classes = max(set([len(np.unique(lab)) for lab in labels_all]))    
        else:
            raise NotImplementedError(
                "Provide training and test data as python list (or dictionary)",
                "of numpy arrays or as 4D (images)",
                "and 4D/3D (labels for single/multi class) numpy arrays"
            )
        assert num_classes != 1,\
        "Confirm that you have a class corresponding to background"
        num_classes = num_classes - 1 if num_classes == 2 else num_classes

        imshapes_train = set([len(im.shape) for im in images_all])
        assert len(imshapes_train) == 1,\
        "All training images must have the same dimensionality"
        imshapes_test = set([len(im.shape) for im in images_test_all])
        assert len(imshapes_test) == 1,\
        "All test images must have the same dimensionality"
        if imshapes_train.pop() == 3:
            warnings.warn(
                'Adding a channel dimension of 1 to training images',
                UserWarning
            )
            images_all_e = [
                np.expand_dims(im, axis=1) for im in images_all]
            images_all = images_all_e
        if imshapes_test.pop() == 3:
            warnings.warn(
                'Adding a channel dimension of 1 to test images',
                UserWarning
            )
            images_test_all_e = [
                np.expand_dims(im, axis=1) for im in images_test_all]
            images_test_all = images_test_all_e
        
        lshapes_train = set([len(l.shape) for l in labels_all])
        assert len(lshapes_train) == 1,\
        "All labels must have the same dimensionality"
        lshapes_test = set([len(l.shape) for l in labels_test_all])
        assert len(lshapes_test) == 1,\
        "All labels must have the same dimensionality"
        if num_classes == 1 and lshapes_train.pop() == 3:
            warnings.warn(
                'Adding a channel dimension of 1 to training labels',
                UserWarning
            )
            labels_all_e = [
                np.expand_dims(l, axis=1) for l in labels_all]
            labels_all = labels_all_e
        if num_classes == 1 and lshapes_test.pop() == 3:
            warnings.warn(
                'Adding a channel dimension of 1 to test labels',
                UserWarning
            )
            labels_test_all_e = [
                np.expand_dims(l, axis=1) for l in labels_test_all]
            labels_test_all = labels_test_all_e
        use_batchnorm = kwargs.get('use_batchnorm', True)
        use_dropouts = kwargs.get('use_dropouts', False)
        upsampling = kwargs.get('upsampling', "bilinear")
        if model_type == 'dilUnet':
            with_dilation = kwargs.get('with_dilation', True)
            nb_filters = kwargs.get('nb_filters', 16)
            self.net = dilUnet(
                num_classes, nb_filters, with_dilation,
                use_dropouts, use_batchnorm, upsampling
            )
        elif model_type == 'dilnet':
            nb_filters = kwargs.get('nb_filters', 25)
            self.net = dilnet(
                num_classes, nb_filters, 
                use_dropouts, use_batchnorm, upsampling
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
            self.criterion = losses_.dice_loss()
        elif loss == 'focal':
            self.criterion = losses_.focal_loss()
        elif loss == 'ce' and num_classes == 1:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif loss == 'ce' and num_classes > 2:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(
                "Choose between Dice loss ('dice') and cross-entropy loss ('ce')"
            )
        self.batch_idx_train = np.random.randint(
            0, len(images_all), training_cycles)
        self.batch_idx_test = np.random.randint(
            0, len(images_test_all), training_cycles)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.images_all = images_all
        self.labels_all = labels_all
        self.images_test_all = images_test_all
        self.labels_test_all = labels_test_all
        self.training_cycles = training_cycles
        self.num_classes = num_classes
        self.batch_size = kwargs.get("batch_size", 32)
        self.print_loss = kwargs.get("print_loss", 100)
        self.savedir = kwargs.get("savedir", "./")
        self.savename = kwargs.get("savename", "model")
        self.plot_training_history = kwargs.get("plot_training_history", True)
        self.train_loss, self.test_loss = [], []

    def dataloader(self, batch_num, mode='train'):
        # Generate batch of training images with corresponding ground truth
        if mode == 'test':
            images = self.images_test_all[batch_num][:self.batch_size]
            labels = self.labels_test_all[batch_num][:self.batch_size]
        else:
            images = self.images_all[batch_num][:self.batch_size]
            labels = self.labels_all[batch_num][:self.batch_size]
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
        """Forward --> Backward --> Optimize"""
        self.net.train()
        self.optimizer.zero_grad()
        prob = self.net.forward(img)
        loss = self.criterion(prob, lbl)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_step(self, img, lbl):
        """Forward path for test data"""
        self.net.eval()
        with torch.no_grad():
            prob = self.net.forward(img)
            loss = self.criterion(prob, lbl)
        return loss.item()

    def run(self):
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
                torch.save(self.net.state_dict(),
                   os.path.join(self.savedir, self.savename+'_test_weights_best.pt'))
        # Save final model weights
        torch.save(self.net.state_dict(),
                   os.path.join(self.savedir, self.savename+'_weights_final.pt'))
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
    Predictions with a trained neural network

    Args:
        image_data: 2D or 3D numpy array
            image stack or a single image (all greyscale)
        trained_model: object
            trained pytorch model (skeleton+weights)
        resize: tuple / 2-element list
            target dimensions for optional image(s) resizing
        use_gpu: bool
            use gpu device for inference
        seed: int
            sets seed for random number generators
    Kwargs:
        **nb_classes: int
            number of classes in the model
        **downsampled: int or float
            downsampling factor
     """
    def __init__(self,
                 image_data,
                 trained_model,
                 resize=None,
                 use_gpu=False,
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
        self.use_gpu = use_gpu

    def predict(self, images):
        '''Returns 'probability' of each pixel
           in image belonging to an atom'''
        if self.use_gpu:
            images = images.cuda()
        self.model.eval()
        with torch.no_grad():
            prob = self.model.forward(images)
            if self.nb_classes > 1:
                prob = F.softmax(prob, dim=1)
            else:
                prob = torch.sigmoid(prob)
        if self.use_gpu:
            images = images.cpu()
            prob = prob.cpu()
        prob = prob.permute(0, 2, 3, 1) # reshape to have channel as a last dim
        prob = prob.numpy()
        return prob

    def run(self):
        '''Make prediction'''
        start_time = time.time()
        if self.image_data.shape[0] < 20 and min(self.image_data.shape[2:4]) < 512:
            decoded_imgs = self.predict(self.image_data)
        else:
            n, _, w, h = self.image_data.shape
            decoded_imgs = np.zeros((n, w, h, self.nb_classes))
            for i in range(n):
                decoded_imgs[i, :, :, :] = self.predict(self.image_data[i:i+1])
        n_images_str = " image was " if decoded_imgs.shape[0] == 1 else " images were "
        print(str(decoded_imgs.shape[0])
                + n_images_str + "decoded in approximately "
                + str(np.around(time.time() - start_time, decimals=4))
                + ' seconds')
        images_numpy = self.image_data.permute(0, 2, 3, 1).numpy()
        return images_numpy, decoded_imgs


class locator:
    """
    Transforms pixel data from NN output into coordinate data

    Args:
        decoded_imgs: 4D numpy array
            the output of a neural network (softmax/sigmoid layer)
        threshold: float
            value at which the neural network output is thresholded
        dist_edge: int
            distance within image boundaries not to consider
        dim_order: str
            'channel_last' or 'channel_first' (Default: 'channel last')
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
        '''Extract all atomic coordinates in image
        via CoM method & store data as a dictionary
        (key: frame number)'''

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
        '''Remove coordinates at the image edges'''

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
