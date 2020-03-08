"""
utils.py
========

Utility functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import subprocess
import random
import torch
import numpy as np
import cv2
from scipy import ndimage, fftpack, spatial
from sklearn.feature_extraction.image import extract_patches_2d
from skimage import exposure
from skimage.util import random_noise
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings


#####################
# Model weights #
#####################


def load_model(model, weights_path):
    """
    Loads weights saved as pytorch state dictionary into a model skeleton

    Args:
        model (pytorch object):
            Initialized pytorch model
        weights_path (str):
            Filepath to trained weights (pytorch state dict)

    Returns:
        Model with trained weights loaded in evaluation state

    Example:

        >>> from atomai.utils import load_model
        >>> # Path to file with trained weights
        >>> weights_path = '/content/simple_model_weights.pt'
        >>> # initialize model (by default all models trained are 'dilUnet')
        >>> model = models.dilUnet(nb_classes=3)
        >>> # load the weights into the model skeleton
        >>> model = load_model(model, weights_path)
    """
    torch.manual_seed(0)
    if torch.cuda.device_count() > 0:
        checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model.eval()


def average_weights(ensemble):
    """
    Averages weights of all models in the ensemble

    Args:
        ensemble (dict):
            dictionary with trained weights (model's state_dict)
            of models with exact same architecture.

    Returns:
        Average weights (as model's state_dict)
    """
    ensemble_state_dict = ensemble[0]
    names = [name for name in ensemble_state_dict.keys()]
    for name in names:
        w_aver = []
        for model in ensemble.values():
            for n, p in model.items():
                if n == name:
                    w_aver.append(p)
        ensemble_state_dict[name].copy_(sum(w_aver) / len(w_aver))
    return ensemble_state_dict


#######################
# GPU characteristics #
#######################


def gpu_usage_map(cuda_device):
    """
    Get the current GPU memory usage
    Adapted with changes from
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--id=' + str(cuda_device),
            '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_usage = [int(y) for y in result.split(',')]
    return gpu_usage[0:2]


#####################
# Image preprocessing #
#####################

def preprocess_training_data(images_all,
                             labels_all,
                             images_test_all,
                             labels_test_all,
                             batch_size):
    """
    Preprocess training and test data

    Args:
        images_all (list / dict / 4D numpy array):
            list or dictionary of 4D numpy arrays or 4D numpy array
            (3D image tensors stacked along the first dim)
            representing training images
        labels_all (list / dict / 4D numpy array):
            list or dictionary of 3D numpy arrays or
            4D (binary) / 3D (multiclass) numpy array
            where 3D / 2D image are tensors stacked along the first dim
            which represent training labels (aka masks aka ground truth)
        images_test_all (list / dict / 4D numpy array):
            list or dictionary of 4D numpy arrays or 4D numpy array
            (3D image tensors stacked along the first dim)
            representing test images
        labels_test_all (list / dict / 4D numpy array):
            list or dictionary of 3D numpy arrays or
            4D (binary) / 3D (multiclass) numpy array
            where 3D / 2D image are tensors stacked along the first dim
            which represent test labels (aka masks aka ground truth)
        batch_size (int):
            size of training and test batches

    Returns:
        4 lists processed with preprocessed training and test data,
        number of classes inferred from the data
    """
    assert type(images_all) == type(labels_all)\
    == type(images_test_all) == type(labels_test_all),\
    "Provide all training and test data in the same format"
    if type(labels_all) == list:
        num_classes = max(set([len(np.unique(lab)) for lab in labels_all]))
    elif type(labels_all) == dict:
        num_classes = max(
            set([len(np.unique(lab)) for lab in labels_all.values()]))
    elif type(labels_all) == np.ndarray:
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
    return (images_all, labels_all,
            images_test_all, labels_test_all,
            num_classes)


def torch_format(image_data):
    """
    Reshapes, normalizes and converts image data
    to pytorch format for model training and prediction

    Args:
        image_data (3D numpy array):
            Image stack with dimensions (n_batches x height x width)
    """
    image_data = np.expand_dims(image_data, axis=1)
    image_data = (image_data - np.amin(image_data))/np.ptp(image_data)
    image_data = torch.from_numpy(image_data).float()
    return image_data


def img_resize(image_data, rs):
    """
    Resizes a stack of images

    Args:
        image_data (3D numpy array):
            Image stack with dimensions (n_batches x height x width)
        rs (tuple):
            target width and height

    Returns:
        Resized stack of images
    """
    if image_data.shape[1:3] == rs:
        return image_data.copy()
    image_data_r = np.zeros(
        (image_data.shape[0], rs[0], rs[1]))
    for i, img in enumerate(image_data):
        img = cv2.resize(img, (rs[0], rs[1]))
        image_data_r[i, :, :] = img
    return image_data_r


def img_pad(image_data, pooling):
    """
    Pads the image if its size (w, h)
    is not divisible by 2**n, where n is a number
    of pooling layers in a network

    Args:
        image_data (3D numpy array):
            Image stack with dimensions (n_batches x height x width)
        pooling (int):
            number of pooling operations
    """
    # Pad image rows (height)
    while image_data.shape[1] % pooling != 0:
        d0, _, d2 = image_data.shape
        image_data = np.concatenate(
            (image_data, np.zeros((d0, 1, d2))), axis=1)
    # Pad image columns (width)
    while image_data.shape[2] % pooling != 0:
        d0, d1, _ = image_data.shape
        image_data = np.concatenate(
            (image_data, np.zeros((d0, d1, 1))), axis=2)
    return image_data


######################
# Atomic coordinates #
######################


def find_com(image_data):
    """
    Find atoms via center of mass methods

    Args:
        image_data (2D numpy array):
            2D image (usually an output of neural network)
    """
    labels, nlabels = ndimage.label(image_data)
    coordinates = np.array(
        ndimage.center_of_mass(
            image_data, labels, np.arange(nlabels) + 1))
    coordinates = coordinates.reshape(coordinates.shape[0], 2)
    return coordinates


def compare_coordinates(coordinates1,
                        coordinates2,
                        d_max,
                        plot_results=False,
                        **kwargs):
    """
    Finds difference between predicted ('coordinates1')
    and "true" ('coordinates2') coordinates using scipy.spatialcKDTree method.
    Use 'd_max' to set maximum search radius. If plotting, pass figure size
    and experimental image using keyword arguments 'fsize' and 'expdata'.
    """
    coordinates1_ = np.empty((0, 3))
    coordinates2_ = np.empty((0, 3))
    delta_r = []
    for c in coordinates1:
        dist, idx = spatial.cKDTree(coordinates2).query(c)
        if dist < d_max:
            coordinates1_ = np.append(coordinates1_, [c], axis=0)
            coordinates2_ = np.append(
                coordinates2_, [coordinates2[idx]], axis=0)
            delta_r.append(dist)
    if plot_results:
        fsize = kwargs.get('fsize', 20)
        expdata = kwargs.get('expdata')
        assert expdata is not None,\
        "For plotting, provide 2D image via 'expdata' keyword"
        plt.figure(figsize=(int(fsize*1.25), fsize))
        plt.imshow(expdata, cmap='gray')
        im = plt.scatter(
            coordinates1_[:, 1], coordinates1_[:, 0],
            c=np.array(delta_r), cmap='jet', s=5)
        clrbar = plt.colorbar(im)
        clrbar.set_label('Position deviation (px)')
        plt.show()
    return coordinates1_, coordinates2_, np.array(delta_r)


def cv_thresh(imgdata,
              threshold=.5):
    """
    Wrapper for opencv binary threshold method.
    Returns thresholded image.
    """
    _, thresh = cv2.threshold(
                    imgdata,
                    threshold, 1,
                    cv2.THRESH_BINARY)
    return thresh


def filter_cells_(imgdata,
                  im_thresh=.5,
                  blob_thresh=150,
                  filter_='above'):
    """
    Filters out blobs above/below cetrain size
    in the thresholded neural network output
    """
    imgdata = cv_thresh(imgdata, im_thresh)
    label_img, cc_num = ndimage.label(imgdata)
    cc_areas = ndimage.sum(imgdata, label_img, range(cc_num + 1))
    if filter == 'above':
        area_mask = (cc_areas > blob_thresh)
    else:
        area_mask = (cc_areas < blob_thresh)
    label_img[area_mask[label_img]] = 0
    label_img[label_img > 0] = 1
    return label_img


def filter_cells(imgdata,
                 im_thresh=0.5,
                 blob_thresh=50,
                 filter_='above'):
    """
    Filters blobs above/below certain size
    for each image in the stack.
    The 'imgdata' must have dimensions (n x h x w).

    Args:
        imgdata (3D numpy array):
            stack of images (without channel dimension)
        im_thresh (float):
            value at which each image in the stack will be thresholded
        blob_thresh (int):
            maximum/mimimun blob size for thresholding
        filter_ (string):
            Select 'above' or 'below' to remove larger or smaller blobs,
            respectively

    Returns:
        Image stack with the same dimensions as the input data
    """
    filtered_stack = np.zeros_like(imgdata)
    for i, img in enumerate(imgdata):
        filtered_stack[i] = filter_cells_(
            img, im_thresh, blob_thresh, filter_)
    return filtered_stack


##########################
# NN structure inference #
##########################

class Hook():
    """
    Returns the input and output of a
    layer during forward/backward pass
    see https://www.kaggle.com/sironghuang/
        understanding-pytorch-hooks/notebook

    Args:
        module (torch module): single layer or sequential block)
        backward (bool): replace forward_hook with backward_hook
    """
    def __init__(self, module, backward=False):
        if backward is False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input_, output_):
        self.input = input_
        self.output = output_

    def close(self):
        self.hook.remove()


def mock_forward(model, dims=(1, 64, 64)):
    """
    Passes a dummy variable throuh a network
    """
    x = torch.randn(1, dims[0], dims[1], dims[2])
    if next(model.parameters()).is_cuda:
        x = x.cuda()
    out = model(x)
    return out


def get_nb_classes(weights_path):
    """
    Returns the number of classes used in trained AtomAI models
    from the loaded weights.

    Args:
        weight_path (str):
            Path to file with saved weights (.pt extension)

    """
    checkpoint = torch.load(weights_path, map_location='cpu')
    last_layer = [k for k in checkpoint.keys()][-1]
    return list(checkpoint[last_layer].size())[0]


#####################
# Vizualization #
#####################

def plot_losses(train_loss, test_loss):
    """
    Plots train and test losses
    """
    print('Plotting training history')
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(train_loss, label='Train')
    ax.plot(test_loss, label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()


def plot_coord(img, coord, fsize=6):
    """
    Plots coordinates (colored according to atom class)
    """
    y, x, c = coord.T
    plt.figure(figsize=(fsize, fsize))
    plt.imshow(img, cmap='gray')
    plt.scatter(x, y, c=c, cmap='RdYlGn', s=8)
    plt.show()


def draw_boxes(imgdata, defcoord, bbox=16, fsize=6):
    """
    Draws boxes cetered around the extracted dedects
    """
    _, ax = plt.subplots(1, 1, figsize=(fsize, fsize))
    ax.imshow(imgdata, cmap='gray')
    for point in defcoord:
        startx = int(round(point[0] - bbox))
        starty = int(round(point[1] - bbox))
        p = patches.Rectangle(
            (starty, startx), bbox*2, bbox*2,
            fill=False, edgecolor='orange', lw=2)
        ax.add_patch(p)
    ax.grid(False)
    plt.show()


#############################
# Training data preparation #
#############################

class MakeAtom:
    """
    Creates an image of atom modelled as
    2D Gaussian and a corresponding mask

    Args:
        sc (float):
            scale parameter, which determines Gaussian width
        intensity (float):
            parameter of 2D gaussian function
        theta (float):
            parameter of 2D gaussian function
        offset (float):
            parameter of 2D gaussian function
    """
    def __init__(self, sc, cfp=2, intensity=1, theta=0, offset=0):
        if sc % 2 == 0:
            sc += 1
        self.xo, self.yo = sc/2, sc/2
        x = np.linspace(0, sc, sc)
        y = np.linspace(0, sc, sc)
        self.x, self.y = np.meshgrid(x, y)
        self.sigma_x, self.sigma_y = sc/4, sc/4
        self.theta = theta
        self.offset = offset
        self.intensity = intensity

    def atom2dgaussian(self):
        """
        Models atom as 2d Gaussian
        """
        a = (np.cos(self.theta)**2)/(2*self.sigma_x**2) +\
            (np.sin(self.theta)**2)/(2*self.sigma_y**2)
        b = -(np.sin(2*self.theta))/(4*self.sigma_x**2) +\
             (np.sin(2*self.theta))/(4*self.sigma_y**2)
        c = (np.sin(self.theta)**2)/(2*self.sigma_x**2) +\
            (np.cos(self.theta)**2)/(2*self.sigma_y**2)
        g = self.offset + self.intensity*np.exp(
            -(a*((self.x-self.xo)**2) + 2*b*(self.x-self.xo)*(self.y-self.yo) +\
            c*((self.y-self.yo)**2)))
        return g


def create_lattice_mask(lattice, xy_atoms, *args, **kwargs):
    """
    Given experimental image and xy atomic coordinates
    creates ground truth image. Currently works only for the case
    where all atoms are one class. Notice that it will round fractional pixels.

    Args:
        lattice (2D numpy array):
            experimental image as 2D numpy array
        xy_atoms (2 x N numpy array):
            position of atoms in the experimental data
        *arg (python function):
            function that creates a 2D numpy array with atom and
            corresponding mask for each atomic coordinate.

            Example:

            >>> def create_atomic_mask(r=7, thresh=.2):
            >>>     atom = MakeAtom(r).atom2dgaussian()
            >>>     _, mask = cv2.threshold(atom, thresh, 1, cv2.THRESH_BINARY)
            >>>     return atom, mask

        **scale: int
            controls the atomic mask size

    Returns:
        2D numpy array with ground truth data
    """
    if len(args) == 1:
        create_mask_func = args[0]
    else:
        create_mask_func = create_atom_mask_pair
    scale = kwargs.get("scale", 7)
    lattice_mask = np.zeros_like(lattice)
    for i in range(xy_atoms.shape[-1]):
        x, y = xy_atoms[:, i]
        x = int(np.around(x))
        y = int(np.around(y))
        _, mask = create_mask_func(scale)
        r_m = mask.shape[0] / 2
        r_m1 = int(r_m + .5)
        r_m2 = int(r_m - .5)
        lattice_mask[x-r_m1:x+r_m2, y-r_m1:y+r_m2] = mask
    return lattice_mask


def create_atom_mask_pair(r=7, thresh=.2):
    """
    Helper function for creating atomic masks
    from 2D gaussian via simple thresholding
    """
    atom = MakeAtom(r).atom2dgaussian()
    _, mask = cv2.threshold(atom, thresh, 1, cv2.THRESH_BINARY)
    mask = mask[np.min(np.where(mask == 1)[0]):
                np.max(np.where(mask == 1)[0] + 1),
                np.min(np.where(mask == 1)[1]):
                np.max(np.where(mask == 1)[1]) + 1]
    return atom, mask


def extract_patches_(lattice_im, lattice_mask, patch_size, num_patches):
    """
    Extracts subimages of the selected size from the 'mother images' of
    atomic lattice and the corresponding mask
    (atomic contours with constant pixel values)
    """
    if type(patch_size) == int:
        patch_size = (patch_size, patch_size)
    images = extract_patches_2d(
        lattice_im, patch_size, max_patches=num_patches, random_state=0)
    labels = extract_patches_2d(
        lattice_mask, patch_size, max_patches=num_patches, random_state=0)
    return images, labels


class augmentor:
    """
    Applies a sequence of pre-defined operations for data augmentation.

    Args:
        batch_size (int):
            number of images in the batch,
        width (int):
            width of images in the batch,
        height (int):
            height of images in the batch,
        channels (int):
            number of classes (channels) in the ground truth
        dim_order_in (str):
            channel first or channel last ordering in the input masks
        dim_order_out (str):
            channel first or channel last ordering in the output masks
        norm (bool):
            normalization to (0, 1)
        seed (int):
            determenism
        **flip (bool):
            image vertical/horizonal flipping,
        **rotate90 (bool):
            rotating image by +- 90 deg
        **zoom (tuple):
            values for zooming-in (min height, max height, step);
            assumes height==width
        **noise (dict):
            dictionary of with range of noise values for each type of noise;
            dictionary keys are:
            'poisson', 'gauss', 'blur', 'contrast', 'salt and pepper'.
            For each case, you need to specify the range of values.
            Notice that for poisson noise,
            smaller values result in larger noise
        **resize (tuple):
            values for image resizing (min height, max height, step);
            assumes heght==width

    Examples:
        Suppose we have a stack of images
        and a stack of masks (aka labels aka ground truth)
        with dimensions (n_images, height, width)
        and (n_images, height, width, channels).
        We can use the augmentor as follows.

        >>> # Specify size, dimensions
        >>> batch_size = len(labels_all) # here we will pass through the augmentor all data at once
        >>> dim1, dim2, ch = labels_all.shape[1:]
        >>> # Define image distortion/noise parameters
        >>> zoom = (256-128, 256+1, 8)
        >>> noise_dict = {}
        >>> noise_dict['poisson'] = (80, 130)
        >>> noise_dict['gauss'] = (1, 400)
        >>> noise_dict['blur'] = (1, 40)
        >>> noise_dict['contrast'] = (50, 400)
        >>> noise_dict['salt and pepper'] = (1, 50)
        >>> # Run the augmentor
        >>> imaug = augmentor(
        >>>    batch_size=batch_size, width=dim1, height=dim2, n_channels=ch,
        >>>    dim_order_in='channel_last', dim_order_out='channel_first',
        >>>    noise=noise_dict, zoom=zoom, flip=True, squeeze=True)
        >>> images_all, labels_all = imaug.run(images_all, labels_all)
    """
    def __init__(self, batch_size, width, height,
                 n_channels, dim_order_in='channel_last',
                 dim_order_out='channel_first', squeeze=False,
                 seed=None, **kwargs):
        self.n, self.w, self.h = batch_size, width, height
        self.ch = n_channels
        self.dim_order_in = dim_order_in
        self.dim_order_out = dim_order_out
        self.squeeze = squeeze
        self.flip = kwargs.get('flip')
        self.rotate90 = kwargs.get('rotate90')
        self.zoom = kwargs.get('zoom')
        self.noise = kwargs.get('noise')
        self.resize = kwargs.get('resize')
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def run(self, images, masks):
        """
        Applies a sequence of augmentation procedures
        to images and (except for noise) ground truth
        """
        if self.dim_order_in == 'channel_first':
            masks = np.transpose(masks, [0, 2, 3, 1])
        elif self.dim_order_in == 'channel_last':
            pass
        else:
            raise NotImplementedError("Use 'channel_first' or 'channel_last'")
        images = (images - np.amin(images))/np.ptp(images)
        if self.flip:
            images, masks = self.batch_flip(images, masks)
        if self.noise is not None:
            images, masks = self.batch_noise(images, masks)
        if self.zoom is not None:
            images, masks = self.batch_zoom(images, masks)
        if self.resize is not None:
            images, masks = self.batch_resize(images, masks)
        if self.squeeze:
            images, masks = self.squeeze_data(images, masks)
        if self.dim_order_out == 'channel_first':
            images = np.expand_dims(images, axis=1)
            if self.squeeze is None or self.ch == 1:
                masks = np.transpose(masks, (0, 3, 1, 2))
        elif self.dim_order_out == 'channel_last':
            images = np.expand_dims(images, axis=3)
        else:
            raise NotImplementedError("Use 'channel_first' or 'channel_last'")
        images = (images - np.amin(images))/np.ptp(images)
        return images, masks

    def batch_noise(self, X_batch, y_batch,):
        """
        Takes an image stack and applies
        various types of noise to each image
        """
        def make_pnoise(image, l):
            vals = len(np.unique(image))
            vals = (l/50) ** np.ceil(np.log2(vals))
            image_n_filt = np.random.poisson(image * vals) / float(vals)
            return image_n_filt
        pnoise_range = self.noise['poisson']
        spnoise_range = self.noise['salt and pepper']
        gnoise_range = self.noise['gauss']
        blevel_range = self.noise['blur']
        c_level_range = self.noise['contrast']
        X_batch_a = np.zeros((self.n, self.w, self.h))
        for i, img in enumerate(X_batch):
            pnoise = random.randint(pnoise_range[0], pnoise_range[1])
            spnoise = random.randint(spnoise_range[0], spnoise_range[1])
            gnoise = random.randint(gnoise_range[0], gnoise_range[1])
            blevel = random.randint(blevel_range[0], blevel_range[1])
            clevel = random.randint(c_level_range[0], c_level_range[1])
            img = ndimage.filters.gaussian_filter(img, blevel*1e-2)
            img = make_pnoise(img, pnoise)
            img = random_noise(img, mode='gaussian', var=gnoise*1e-4)
            img = random_noise(img, mode='pepper', amount=spnoise*1e-4)
            img = random_noise(img, mode='salt', amount=spnoise*5e-5)
            img = exposure.adjust_gamma(img, clevel*1e-2)
            X_batch_a[i, :, :] = img
        return X_batch_a, y_batch

    def batch_zoom(self, X_batch, y_batch):
        """
        Crops and then resizes to the original size
        all images in one batch
        """
        zoom_list = np.arange(self.zoom[0], self.zoom[1], self.zoom[2])
        X_batch_a = np.zeros((self.n, self.w, self.h))
        y_batch_a = np.zeros((self.n, self.w, self.h, self.ch))
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            rs = random.choice(zoom_list)
            w1 = int((self.w-rs)/2)
            w2 = int(rs + (self.w-rs)/2)
            h1 = int((self.h-rs)/2)
            h2 = int(rs + (self.h-rs)/2)
            img = img[w1:w2, h1:h2]
            gt = gt[w1:w2, h1:h2]
            img = cv2.resize(img, (self.w, self.h))
            gt = cv2.resize(gt, (self.w, self.h))
            #_, gt = cv2.threshold(gt, 0.5, 1, cv2.THRESH_BINARY)
            gt = np.around(gt)
            if len(gt.shape) != 3:
                gt = np.expand_dims(gt, axis=2)
            X_batch_a[i, :, :] = img
            y_batch_a[i, :, :, :] = gt
        return X_batch_a, y_batch_a

    def batch_flip(self, X_batch, y_batch):
        """
        Flips and rotates all images and in one batch
        and correponding labels (ground truth)
        """
        X_batch_a = np.zeros((self.n, self.w, self.h))
        y_batch_a = np.zeros((self.n, self.w, self.h, self.ch))
        int_r = (-1, 3) if self.rotate90 else (-1, 1)
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            flip_type = random.randint(int_r[0], int_r[1])
            if flip_type == 3:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            elif flip_type == 2:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                img = cv2.flip(img, flip_type)
                gt = cv2.flip(gt, flip_type)
            if len(gt.shape) != 3:
                gt = np.expand_dims(gt, axis=2)
            X_batch_a[i, :, :] = img
            y_batch_a[i, :, :, :] = gt
        return X_batch_a, y_batch_a

    def batch_resize(self, X_batch, y_batch):
        """
        Resize all images in one batch and
        corresponding labels (ground truth)
        """
        rs_arr = np.arange(self.resize[0], self.resize[1], self.resize[2])
        rs = random.choice(rs_arr)
        if X_batch.shape[1:3] == (rs, rs):
            return X_batch, y_batch
        X_batch_a = np.zeros((self.n, rs, rs))
        y_batch_a = np.zeros((self.n, rs, rs, self.ch))
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            img = cv2.resize(img, (rs, rs), cv2.INTER_CUBIC)
            gt = cv2.resize(gt, (rs, rs), cv2.INTER_CUBIC)
            #_, gt = cv2.threshold(gt, 0.5, 1, cv2.THRESH_BINARY)
            gt = np.around(gt)
            if len(gt.shape) < 3:
                gt = np.expand_dims(gt, axis=-1)
            X_batch_a[i, :, :] = img
            y_batch_a[i, :, :, :] = gt
        return X_batch_a, y_batch_a

    @classmethod
    def squeeze_data(cls, images, labels):
        """
        Squeezes channels in each training image and
        filters out image-label pairs where some pixels have multiple values.
        As a result the number of image-label-pairs returned may be different
        from the number of image-label pairs in the original data.
        """

        def squeeze_channels(label):
            """
            Squeezes multiple channel into a single channel for a single label
            """
            label_ = np.zeros((1, label.shape[0], label.shape[1]))
            for c in range(label.shape[-1]):
                label_ += label[:, :, c] * c
            return label_

        if labels.shape[-1] == 1:
            return images, labels
        images_valid, labels_valid = [], []
        for label, image in zip(labels, images):
            label = squeeze_channels(label)
            if len(np.unique(label)) == labels.shape[-1]:
                labels_valid.append(label)
                images_valid.append(image[None, ...])
        return np.concatenate(images_valid), np.concatenate(labels_valid)


def squeeze_channels_(y_train):
    """
    Squeezes multiple channel into a single channel for a batch of labels.
    Assumes 'channel first ordering'
    """
    y_train_ = np.zeros((y_train.shape[0], y_train.shape[2], y_train.shape[3]))
    for c in range(y_train.shape[1]):
        y_train_ += y_train[:, c] * c
    return y_train_


def squeeze_data_(images, labels):
    """
    Squeezes channels in each training image and
    filters out image-label pairs where some pixels have multiple values.
    As a result the number of image-label-pairs returned may be different
    from the number of image-label pairs in the original data.
    Assumes 'channel first' ordering
    """
    if labels.shape[1] == 1:
        return images, labels
    images_valid, labels_valid = [], []
    for label, image in zip(labels, images):
        label = squeeze_channels_(label[None, ...])
        unique_labels = len(np.unique(label))
        if unique_labels == labels.shape[1]:
            labels_valid.append(label)
            images_valid.append(image[None, ...])
    return np.concatenate(images_valid), np.concatenate(labels_valid)


def FFTmask(imgsrc, maskratio=10):
    """
    Takes a square real space image and filter out a disk with radius equal to:
    1/maskratio * image size.
    Retruns FFT transform of the image and the filtered FFT transform
    """
    # Take the fourier transform of the image.
    F1 = fftpack.fft2((imgsrc))
    # Now shift so that low spatial frequencies are in the center.
    F2 = (fftpack.fftshift((F1)))
    # copy the array and zero out the center
    F3 = F2.copy()
    l = int(imgsrc.shape[0]/maskratio)
    m = int(imgsrc.shape[0]/2)
    y, x = np.ogrid[1: 2*l + 1, 1:2*l + 1]
    mask = (x - l)*(x - l) + (y - l)*(y - l) <= l*l
    F3[m-l:m+l, m-l:m+l] = F3[m-l:m+l, m-l:m+l] * (1 - mask)
    return F2, F3


def FFTsub(imgsrc, imgfft):
    """
    Takes real space image and filtred FFT.
    Reconstructs real space image and subtracts it from the original.
    Returns normalized image.
    """
    reconstruction = np.real(fftpack.ifft2(fftpack.ifftshift(imgfft)))
    diff = np.abs(imgsrc - reconstruction)
    # normalization
    diff = diff - np.amin(diff)
    diff = diff/np.amax(diff)
    return diff


def threshImg(diff, threshL=0.25, threshH=0.75):
    """
    Takes in difference image, low and high thresold values,
    and outputs a map of all defects.
    """
    threshIL = diff < threshL
    threshIH = diff > threshH
    threshI = threshIL + threshIH
    return threshI
