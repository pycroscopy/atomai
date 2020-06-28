"""
utils.py
========

Utility functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import copy
import subprocess
import warnings
from collections import OrderedDict

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import fftpack, ndimage, optimize, spatial, stats
from skimage import exposure
from skimage.util import random_noise
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", module="scipy.optimize")


#####################
# Model weights #
#####################


def load_weights(model, weights_path):
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

        >>> from atomai.utils import load_weights
        >>> # Path to file with trained weights
        >>> weights_path = '/content/simple_model_weights.pt'
        >>> # Initialize model (by default all trained models are 'dilUnet')
        >>> # You can also use nb_classes=utils.nb_filters_classes(weights_path)[1]
        >>> model = models.dilUnet(nb_classes=3)
        >>> # Load the weights into the model skeleton
        >>> model = load_weights(model, weights_path)
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
            Dictionary with trained weights (model's state_dict)
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
            List or dictionary of 4D numpy arrays or 4D numpy array
            (3D image tensors stacked along the first dim)
            representing training images
        labels_all (list / dict / 4D numpy array):
            List or dictionary of 3D numpy arrays or
            4D (binary) / 3D (multiclass) numpy array
            where 3D / 2D image are tensors stacked along the first dim
            which represent training labels (aka masks aka ground truth)
        images_test_all (list / dict / 4D numpy array):
            List or dictionary of 4D numpy arrays or 4D numpy array
            (3D image tensors stacked along the first dim)
            representing test images
        labels_test_all (list / dict / 4D numpy array):
            List or dictionary of 3D numpy arrays or
            4D (binary) / 3D (multiclass) numpy array
            where 3D / 2D image are tensors stacked along the first dim
            which represent test labels (aka masks aka ground truth)
        batch_size (int):
            Size of training and test batches

    Returns:
        4 lists processed with preprocessed training and test data,
        number of classes inferred from the data
    """
    if not (type(images_all) == type(labels_all) ==
            type(images_test_all) == type(labels_test_all)):
        raise AssertionError(
            "Provide all training and test data in the same format")
    if isinstance(labels_all, list):
        pass
    elif isinstance(labels_all, dict):
        images_all = [i for i in images_all.values()]
        labels_all = [i for i in labels_all.values()]
        images_test_all = [i for i in images_test_all.values()]
        labels_test_all = [i for i in labels_test_all.values()]
    elif isinstance(labels_all, np.ndarray):
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
    else:
        raise NotImplementedError(
            "Provide training and test data as python list (or dictionary)",
            "of numpy arrays or as 4D (images)",
            "and 4D/3D (labels for single/multi class) numpy arrays"
        )
    num_classes = max(set([len(np.unique(lab)) for lab in labels_all]))
    if num_classes == 1:
        raise AssertionError(
            "Confirm that you have a class corresponding to background")
    num_classes = num_classes - 1 if num_classes == 2 else num_classes

    imshapes_train = set([len(im.shape) for im in images_all])
    if len(imshapes_train) != 1:
        raise AssertionError(
            "All training images must have the same dimensionality")
    imshapes_test = set([len(im.shape) for im in images_test_all])
    if len(imshapes_test) != 1:
        raise AssertionError(
            "All test images must have the same dimensionality")
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
    if len(lshapes_train) != 1:
        raise AssertionError(
         "All labels must have the same dimensionality")
    lshapes_test = set([len(l.shape) for l in labels_test_all])
    if len(lshapes_test) != 1:
        raise AssertionError(
            "All labels must have the same dimensionality")
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


def img_resize(image_data, rs, round_=False):
    """
    Resizes a stack of images

    Args:
        image_data (3D numpy array):
            Image stack with dimensions (n_batches x height x width)
        rs (tuple):
            Target height and width
        round_(bool):
            rounding (in case of labeled pixels)

    Returns:
        Resized stack of images
    """
    if image_data.shape[1:3] == rs:
        return image_data.copy()
    image_data_r = np.zeros(
        (image_data.shape[0], rs[0], rs[1]))
    for i, img in enumerate(image_data):
        img = cv_resize(img, rs, round_)
        image_data_r[i, :, :] = img
    return image_data_r


def cv_resize(img, rs, round_=False):
    """
    Wrapper for open-cv resize function

    Args:
        img (2D numpy array): input 2D image
        rs (tuple): target height and width
        round_(bool): rounding (in case of labeled pixels)

    Returns:
        Resized image
    """
    if img.shape == rs:
        return img
    rs_method = cv2.INTER_AREA if img.shape[0] < rs[0] else cv2.INTER_CUBIC
    img_rs = cv2.resize(img, rs, interpolation=rs_method)
    if round_:
        img_rs = np.round(img_rs)
    return img_rs


def cv_resize_stack(imgdata, rs, round_=False):
    """
    Resizes a 3D stack of images

    Args:
        imgdata (3D numpy array): stack of 3D images to be resized
        rs (tuple or int): target height and width
        round_(bool): rounding (in case of labeled pixels)

    Returns:
        Resized image
    """
    rs = [rs, rs] if isinstance(rs, int) else rs
    imgdata_rs = np.zeros((imgdata.shape[0], rs[0], rs[1]))
    for i, img in enumerate(imgdata):
        img_rs = cv_resize(img, rs, round_)
        imgdata_rs[i] = img_rs
    return imgdata_rs


def img_pad(image_data, pooling):
    """
    Pads the image if its size (w, h)
    is not divisible by :math:`2^n`, where *n* is a number
    of pooling layers in a network

    Args:
        image_data (3D numpy array):
            Image stack with dimensions (n_batches x height x width)
        pooling (int):
            Downsampling factor (equal to :math:`2^n`, where *n* is a number
            of pooling operations)
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


def get_nn_distances_(coordinates, nn=2, upper_bound=None):
    """
    Calculates nearest-neighbor distances for a single image

    Args:
        coordinates (numpy array):
            :math:`N \\times 3` array with atomic coordinates where first two
            columns are *xy* coordinates and the third column is atom class
        nn (int): Number of nearest neighbors to search for.
        upper_bound (float or int, non-negative):
            Upper distance bound for Query the kd-tree for nearest neighbors.
            Only di
    Returns:
        Tuple with :math:`atoms \\times nn` array of distances to nearest
        neighbors and :math:`atoms \\times (nn+1) \\times 3` array of coordinates
        (including coordinates of the "center" atom), where n_atoms is less or
        equal to the total number of atoms in the 'coordinates'
        (due to 'upper_bound' criterion)
    """
    upper_bound = np.inf if upper_bound is None else upper_bound
    tree = spatial.cKDTree(coordinates[:, :2])
    d, nn = tree.query(
        coordinates[:, :2], k=nn+1, distance_upper_bound=upper_bound)
    idx_to_del = np.where(d == np.inf)[0]
    nn = np.delete(nn, idx_to_del, axis=0)
    d = np.delete(d, idx_to_del, axis=0)
    return d[:, 1:], coordinates[nn]


def transform_coordinates(coord, phi, coord_dx):
    """
    Pytorch-based 2D rotation of coordinates followed by translation.
    Operates on batches.

    Args:
        coord (numpy array or torch tensor): batch with initial coordinates
        phi (float): rotational angle in rad
        coord_dx (numpy array or torch tensor): translation vector

    Returns:
        Transformed coordinates batch
    """

    if isinstance(coord, np.ndarray):
        coord = torch.from_numpy(coord).float()
    if isinstance(coord_dx, np.ndarray):
        coord_dx = torch.from_numpy(coord_dx).float()
    rotmat_r1 = torch.stack([torch.cos(phi), torch.sin(phi)], 1)
    rotmat_r2 = torch.stack([-torch.sin(phi), torch.cos(phi)], 1)
    rotmat = torch.stack([rotmat_r1, rotmat_r2], axis=1)
    coord = torch.bmm(coord, rotmat)

    return coord + coord_dx


def get_nn_distances(coordinates, nn=2, upper_bound=None):
    """
    Calculates nearest-neighbor distances for a stack of images

    Args:
        coordinates (dict):
            Dictionary where keys are frame numbers and values are
            :math:`N \\times 3` numpy arrays with atomic coordinates.
            In each array the first two columns are *xy* coordinates and
            the third column is atom class.
        nn (int): Number of nearest neighbors to search for.
        upper_bound (float or int, non-negative):
            Upper distance bound for Query the kd-tree for nearest neighbors.
            Only distances below this value will be counted.
    Returns:
        Tuple with list of :math:`atoms \\times nn` arrays of distances
        to nearest neighbors and list of :math:`atoms \\times (nn+1) \\times 3`
        array of coordinates (including coordinates of the "center" atom),
        where n_atoms is less or equal to the total number of atoms in the
        'coordinates' (due to 'upper_bound' criterion)
    """
    distances_all, atom_pairs_all = [], []
    for coord in coordinates.values():
        distances, atom_pairs = get_nn_distances_(coord, nn, upper_bound)
        distances_all.append(distances)
        atom_pairs_all.append(atom_pairs)
    return distances_all, atom_pairs_all


def gaussian_2d(xy, amp, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Models 2D Gaussian

    Args:
        xy (tuple): two M x N arrays
        amp (float): peak amplitude
        xo (float): x-coordinate of peak center
        yo (float): y-coordinate of peak center
        sigma_x (float): peak width (x-projection)
        sigma_y (float): peak height (y-projection)
        theta (float): parameter of 2D Gaussian
        offset (float): parameter of 2D Gaussian
    """
    x, y = xy
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amp*np.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.flatten()


def peak_refinement(imgdata, coordinates, d=None):
    """
    Performs a refinement of atomic postitions by fitting
    2d Gaussian where the neural network predictions serve
    as initial guess.

    Args:
        imgdata (2D numpy array):
            Single experimental image/frame
        coordinates (N x 3 numpy array):
            Atomic coordinates where first two columns are *xy* coordinates
            and the third column is atom class
        d (int):
            Half-side of a square around the identified atom for peak fitting
            If d is not specified, it is set to 1/4 of average nearest neighbor
            distance in the lattice.
    Returns:
        Refined array of coordinates
    """
    if d is None:
        d = get_nn_distances_(coordinates)[0]
        d = np.concatenate((d))
        d = int(np.mean(d)*0.25)
    xyc_all = []
    for i, c in enumerate(coordinates[:, :2]):
        cx = int(np.around(c[0]))
        cy = int(np.around(c[1]))
        img = imgdata[cx-d:cx+d, cy-d:cy+d]
        if img.shape == (int(2*d), int(2*d)):
            e1, e2 = img.shape
            x, y = np.mgrid[:e1:1, :e2:1]
            initial_guess = (img[d, d], d, d, 1, 1, 0, 0)
            try:
                popt, pcov = optimize.curve_fit(
                        gaussian_2d, (x, y), img.flatten(), p0=initial_guess)
                if np.linalg.norm(popt[1:3] - d) < 3:
                    xyc = popt[1:3] + np.around(c) - d
                else:
                    xyc = c
            except RuntimeError:
                xyc = c
        else:
            xyc = c
        xyc_all.append(xyc)
    xyc_all = np.concatenate(
        (np.array(xyc_all), coordinates[:, 2:3]), axis=-1)
    return xyc_all


def get_intensities_(coordinates, img):
    """
    Calculates intensities in a 3x3 square around each predicted position
    for a single image
    """
    intensities_all = []
    for c in coordinates:
        cx = int(np.around(c[0]))
        cy = int(np.around(c[1]))
        intensity = np.mean(img[cx-1:cx+2, cy-1:cy+2])
        intensities_all.append(intensity)
    intensities_all = np.array(intensities_all)
    return intensities_all


def get_intensities(coordinates_all, nn_input):
    """
    Calculates intensities in a 3x3 square around each predicted position
    for a stack of images
    """
    intensities_all = []
    for k, coord in coordinates_all.items():
        intensities_all.append(get_intensities_(coord, nn_input[k]))
    return intensities_all


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
        if expdata is None:
            raise AssertionError(
                "For plotting, provide 2D image via 'expdata' keyword")
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
                  filter_='below'):
    """
    Filters out blobs above/below cetrain size
    in the thresholded neural network output
    """
    imgdata = cv_thresh(imgdata, im_thresh)
    label_img, cc_num = ndimage.label(imgdata)
    cc_areas = ndimage.sum(imgdata, label_img, range(cc_num + 1))
    if filter_ == 'above':
        area_mask = (cc_areas > blob_thresh)
    else:
        area_mask = (cc_areas < blob_thresh)
    label_img[area_mask[label_img]] = 0
    label_img[label_img > 0] = 1
    return label_img


def get_contours(imgdata):
    """
    Extracts object contours from image data
    (image data must be binary thresholded)
    """
    imgdata_ = cv2.convertScaleAbs(imgdata)
    contours = cv2.findContours(
        imgdata_.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    return contours


def filter_cells(imgdata,
                 im_thresh=0.5,
                 blob_thresh=50,
                 filter_='below'):
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


def get_blob_params(nn_output, im_thresh, blob_thresh, filter_='below'):
    """
    Extracts position and angle of particles in each movie frame

    Args:
        nn_output (4D numpy array):
            out of neural network returned by atomnet.predictor
        im_thresh (float):
            value at which each image in the stack will be thresholded
        blob_thresh (int):
            maximum/mimimun blob size for thresholding
        filter_ (string):
            Select 'above' or 'below' to remove larger or smaller blobs,
            respectively

    Returns:
        Nested dictionary where for each frame there is an ordered dictionary
        with values of centers of the mass and angle for each detected particle
        in that frame.
    """
    blob_dict = {}
    nn_output = nn_output[..., 0] if np.ndim(nn_output) == 4 else nn_output
    for i, frame in enumerate(nn_output):
        contours = get_contours(frame)
        dictionary = OrderedDict()
        com_arr, angles = [], []
        for cnt in contours:
            if len(cnt) < 5:
                continue
            (com), _, angle = cv2.fitEllipse(cnt)
            com_arr.append(np.array(com)[None, ...])
            angles.append(angle)
        if len(com_arr) > 0:
            com_arr = np.concatenate(com_arr, axis=0)
        else:
            com_arr = None
        angles = np.array(angles)
        dictionary['decoded'] = frame
        dictionary['coordinates'] = com_arr
        dictionary['angles'] = angles
        blob_dict[i] = dictionary
    return blob_dict


class subimg_trajectories:
    """
    Extracts a trajectory of a single defect/atom from image stack
    together with the associated subimages

    Args:
        imgdata (np.ndarray):
            Stack of images (can be raw data or NN output)
        coord_class_dict (dict):
            Dictionary of atomic coordinates
            (same format as produced by atomnet.locator)
        window_size (int):
            size of window for subimage cropping
        min_length (int):
            Minimal length of trajectory to return
        rmax (int):
            Max allowed distance (projected on xy plane) between defect
            in one frame and the position of its nearest neighbor in the next one
    """
    def __init__(self, imgdata, coord_class_dict, window_size, min_length=0, rmax=10):
        self.imgdata = imgdata
        self.coord_class_dict = coord_class_dict
        self.r = window_size
        self.min_length = min_length
        self.rmax = rmax

    def get_trajectory(self, img, start_coord):

        def crop_(img_, c_):
            cx = int(np.around(c_[0]))
            cy = int(np.around(c_[1]))
            img_cr = img_[cx-self.r//2:cx+self.r//2, cy-self.r//2:cy+self.r//2]
            return img_cr

        flow, frames, img_cr_all = [], [], []
        c0 = start_coord
        for k, c in self.coord_class_dict.items():
            d, index = spatial.cKDTree(
                c[:, :2]).query(c0, distance_upper_bound=self.rmax)
            if d != np.inf:
                img_cr = crop_(self.imgdata[k], c[index])
                if img_cr.shape[0:2] == (self.r, self.r):
                    flow.append(c[index])
                    img_cr_all.append(img_cr)
                    frames.append(k)
                    c0 = c[index][:2]
        return np.array(flow), np.array(frames), np.array(img_cr_all)

    def get_all_trajectories(self):
        trajectories_all, frames_all = [], []
        subimgs_all = []
        for ck in self.coord_class_dict[list(self.coord_class_dict.keys())[0]][:,:2]:
            flow, frames, subimgs = self.get_trajectory(self.coord_class_dict, ck)
            if len(flow) > self.min_length:
                trajectories_all.append(flow)
                frames_all.append(frames)
                subimgs_all.append(subimgs)
        return trajectories_all, frames_all, subimgs_all


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


def nb_filters_classes(weights_path):
    """
    Inferes the number of filters and the number of classes
    used in trained AtomAI models from the loaded weights.

    Args:
        weight_path (str):
            Path to file with saved weights (.pt extension)

    """
    checkpoint = torch.load(weights_path, map_location='cpu')
    tensor_shapes = [v.shape for v in checkpoint.values() if len(v.shape) > 1]
    nb_classes = tensor_shapes[-1][0]
    nb_filters = tensor_shapes[0][0]
    return nb_filters, nb_classes


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


def plot_trajectories(traj, frames, **kwargs):
    """
    Plots individual trajectory (as position (radius) vector)

    Args:
        traj (n x 3 ndarray):
            numpy array where first two columns are coordinates
            and the 3rd columd are classes
        frames ((n,) ndarray):
            numpy array with frame numbers
        **lv (int):
            latent variable value to visualize (Default: 1)
        **fov (int or list):
            field of view or scan size
        **fsize (int):
            figure size (Default: 6)
        **cmap (str):
            colormap (Default: jet)
    """
    fov = kwargs.get("fov")
    cmap = kwargs.get("cmap", "jet")
    fsize = kwargs.get("fsize", 6)
    r_coord = np.linalg.norm(traj[:, :2], axis=1)
    if traj.shape[1] == 3:
        c_ = traj[:, -1]
    elif traj.shape[1] > 3:
        lv = kwargs.get("lv", 1)
        c_ = traj[:, 1 + lv]
    plt.figure(figsize=(fsize*2, fsize))
    plt.scatter(frames, r_coord, c=c_, cmap=cmap)
    if fov:
        if isinstance(fov, list) and len(fov) == 2:
            fov = np.sqrt(fov[0]**2 + fov[1]**2)
        elif isinstance(fov, int):
            fov = np.sqrt(2*fov**2)
        else:
            raise ValueError("Pass 'fov' argument as integer or 2-element list")
        plt.ylim(0, fov)
    plt.xlabel("Time step (a.u.)", fontsize=18)
    plt.ylabel("Position vector", fontsize=18)
    cbar = plt.colorbar()
    cbar_lbl = "States" if traj.shape[1] == 3 else "Latent variable {}".format(lv)
    cbar.set_label(cbar_lbl, fontsize=16)
    plt.clabel
    plt.title("Trajectory", fontsize=20)
    plt.show()


def plot_transitions(matrix,
                     states=None,
                     gmm_components=None,
                     plot_values=False,
                     **kwargs):
    """
    Plots transition matrix and (optionally) most frequent/probable transitions

    Args:
        m (2D numpy array):
            Transition matrix
        gmm_components (4D numpy array):
            GMM components (optional)
        plot_values (bool):
            Show calculated transtion rates
        **transitions_to_plot (int):
            number of transitions (associated with largest prob values) to plot
        **plot_toself (bool):
            Skips transitions into self when plotting transitions with largest probs
        **fsize (int): figure size
        **cmap (str): color map
    """
    fsize = kwargs.get("fsize", 6)
    cmap = kwargs.get("cmap", "Reds")
    transitions_to_plot = kwargs.get("transitions_to_plot", 6)
    plot_toself = kwargs.get("plot_toself", True)
    m = matrix
    _, ax = plt.subplots(1, 1, figsize=(fsize, fsize))
    ax.matshow(m, cmap=cmap)
    if states is None:
        states = np.arange(len(m)) + 1
    xt = states
    ax.set_xticks(np.arange(len(xt)))
    ax.set_yticks(np.arange(len(xt)))
    ax.set_xticklabels((xt).tolist(), rotation='horizontal', fontsize=14)
    ax.set_yticklabels((xt).tolist(), rotation='horizontal', fontsize=14)
    ax.set_title('Transition matrix', y=1.1, fontsize=20)
    if plot_values:
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, np.around(v, 2), ha='center', va='center', c='b')
    ax.set_xlabel('Transition class', fontsize=18)
    ax.set_ylabel('Starting class', fontsize=18)
    plt.show()
    if gmm_components is not None:
        idx_ = np.unravel_index(np.argsort(m.ravel()), m.shape)
        idx_ = np.dstack(idx_)[0][::-1]
        print()
        i_ = 0
        for i in idx_:
            if plot_toself is False and i[0] == i[1]:
                continue
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(fsize, fsize//2))
            if gmm_components.shape[-1] == 3:
                start_comp = gmm_components[states[i[0]]-1]
                trans_comp = gmm_components[states[i[1]]-1]
            else:
                start_comp = np.sum(gmm_components[states[i[0]]-1], axis=-1)
                trans_comp = np.sum(gmm_components[states[i[1]]-1], axis=-1)
            print("Starting class  --->  Transition class (Prob: {})".
                  format(m[tuple(i)]))
            ax1.imshow(start_comp, cmap=cmap)
            ax1.set_title("GMM component {}".format(states[i[0]]))
            ax2.imshow(trans_comp, cmap=cmap)
            ax2.set_title("GMM_component {}".format(states[i[1]]))
            plt.show()
            i_ = i_ + 1
            if i_ == transitions_to_plot - 1:
                break
    return


def plot_trajectories_transitions(trans_dict, k, plot_values=False, **kwargs):
    """
    Plots trajectory witht he associated transitions.

    Args:
        trans_dict (dict):
            Python dictionary containing trajectories, frame numbers,
            transitions and the averaged GMM components. Usually this is
            an output of atomstat.transition_matrix
        k (int): Number of trajectory to vizualize
        plot_values (bool): Show calculated transtion rates
        **transitions_to_plot (int):
            number of transitions (associated with largerst prob values) to plot
        **fsize (int): figure size
        **cmap (str): color map
        **fov (int or list): field of view (scan size)
    """
    traj = trans_dict["trajectories"][k]
    frames = trans_dict["frames"][k]
    trans = trans_dict["transitions"][k]
    plot_trajectories(traj, frames, **kwargs)
    print()
    s_true = np.unique(traj[:, -1]).astype(np.int64)
    plot_transitions(
        trans, s_true, trans_dict["gmm_components"],
        plot_values, **kwargs)
    return


#############################
# Training data preparation #
#############################

def get_imgstack(imgdata, coord, r):
    """
    Extracts subimages centered at specified coordinates
    for a single image

    Args:
        imgdata (3D numpy array):
            Prediction of a neural network with dimensions
            :math:`height \\times width \\times n channels`
        coord (N x 2 numpy array):
            (x, y) coordinates
        r (int):
            Window size

    Returns:
        2-element tuple containing
        i) Stack of subimages and
        ii) (x, y) coordinates of their centers
    """
    img_cr_all = []
    com = []
    for c in coord:
        cx = int(np.around(c[0]))
        cy = int(np.around(c[1]))
        img_cr = np.copy(
            imgdata[cx-r//2:cx+r//2, cy-r//2:cy+r//2])
        if img_cr.shape[0:2] == (int(r), int(r)):
            img_cr_all.append(img_cr[None, ...])
            com.append(c[None, ...])
    if len(img_cr_all) == 0:
        return None, None
    img_cr_all = np.concatenate(img_cr_all, axis=0)
    com = np.concatenate(com, axis=0)
    return img_cr_all, com


def imcrop_randpx(img, window_size, num_images, random_state=0):
    """
    Extracts subimages at random pixels

    Returns:
        2-element tuple containing
        i) Stack of subimages and
        ii) (x, y) coordinates of their centers
    """
    list_xy = []
    com_x, com_y = [], []
    n = 0
    while n < num_images:
        x = np.random.randint(
            window_size // 2 + 1, img.shape[0] - window_size // 2 - 1)
        y = np.random.randint(
            window_size // 2 + 1, img.shape[1] - window_size // 2 - 1)
        if (x, y) not in list_xy:
            com_x.append(x)
            com_y.append(y)
            list_xy.append((x, y))
            n += 1
    com_xy = np.concatenate(
        (np.array(com_x)[:, None], np.array(com_y)[:, None]),
        axis=1)
    subimages, com = get_imgstack(img, com_xy, window_size)
    return subimages, com


def imcrop_randcoord(img, coord, window_size, num_images, random_state=0):
    """
    Extracts subimages at random coordinates

    Returns:
        2-element tuple containing
        i) Stack of subimages and
        ii) (x, y) coordinates of their centers
    """
    list_idx, com_xy = [], []
    n = 0
    while n < num_images:
        i = np.random.randint(len(coord))
        if i not in list_idx:
            com_xy.append(coord[i].tolist())
            list_idx.append(i)
            n += 1
    com_xy = np.array(com_xy)
    subimages, com = get_imgstack(img, com_xy, window_size)
    return subimages, com


def extract_random_subimages(imgdata, window_size, num_images,
                             coordinates=None, **kwargs):
    """
    Extracts randomly subimages centered at certain atom class/type
    (usually from a neural network output) or just at random pixels
    (if coordinates are not known/available)

    Args:
        imgdata (numpy array): 4D stack of images (n, height, width, channel)
        window_size (int):
            Side of the square for subimage cropping
        num_images (int): number of images to extract from each "frame" in the stack
        coordinates (dict): Optional. Prediction from atomnet.locator
            (can be from other source but must be in the same format)
            Each element is a :math:`N \\times 3` numpy array,
            where *N* is a number of detected atoms/defects,
            the first 2 columns are *xy* coordinates
            and the third columns is class (starts with 0)
        **coord_class (int):
            Class of atoms/defects around around which the subimages
            will be cropped (3rd column in the atomnet.locator output)

    Returns:
        3-element tuple containing
        i) stack of subimages,
        ii) (x, y) coordinates of their centers,
        iii) frame number associated with each subimage

    """
    if coordinates:
        coord_class = kwargs.get("coord_class", 0)
    if np.ndim(imgdata) < 4:
        imgdata = imgdata[..., None]
    subimages_all = np.zeros(
        (num_images * imgdata.shape[0],
         window_size, window_size, imgdata.shape[-1]))
    com_all = np.zeros((num_images * imgdata.shape[0], 2))
    frames_all = np.zeros((num_images * imgdata.shape[0]))
    for i, img in enumerate(imgdata):
        if coordinates is None:
            stack_i, com_i = imcrop_randpx(
                img, window_size, num_images, random_state=i)
        else:
            coord = coordinates[i]
            coord = coord[coord[:, -1] == coord_class]
            coord = coord[:, :2]
            coord = remove_edge_coord(coord, imgdata.shape[1:3], window_size // 2 + 1)
            if num_images > len(coord):
                raise ValueError(
                    "Number of images cannot be greater than the available coordinates")
            stack_i, com_i = imcrop_randcoord(
                img, coord, window_size, num_images, random_state=i)
        subimages_all[i * num_images: (i + 1) * num_images] = stack_i
        com_all[i * num_images: (i + 1) * num_images] = com_i
        frames_all[i * num_images: (i + 1) * num_images] = np.ones(len(com_i), int) * i
    return subimages_all, com_all, frames_all


def remove_edge_coord(coordinates, dim, dist_edge):
    """
    Removes coordinates at the image edges
    """

    def coord_edges(coordinates, h, w):
        return [coordinates[0] > w - dist_edge,
                coordinates[0] < dist_edge,
                coordinates[1] > h - dist_edge,
                coordinates[1] < dist_edge]

    h, w = dim
    coord_to_rem = [
                    idx for idx, c in enumerate(coordinates)
                    if any(coord_edges(c, h, w))
                    ]
    coord_to_rem = np.array(coord_to_rem, dtype=int)
    coordinates = np.delete(coordinates, coord_to_rem, axis=0)
    return coordinates


def extract_subimages(imgdata, coordinates, window_size, coord_class=0):
    """
    Extracts subimages centered at certain atom class/type
    (usually from a neural network output)

    Args:
        imgdata (numpy array): 4D stack of images (n, height, width, channel)
        coordinates (dict): Prediction from atomnet.locator
            (can be from other source but must be in the same format)
            Each element is a :math:`N \\times 3` numpy array,
            where *N* is a number of detected atoms/defects,
            the first 2 columns are *xy* coordinates
            and the third columns is class (starts with 0)
        window_size (int):
            Side of the square for subimage cropping
        coord_class (int):
            Class of atoms/defects around around which the subimages
            will be cropped (3rd column in the atomnet.locator output)

    Returns:
        3-element tuple containing
        i) stack of subimages,
        ii) (x, y) coordinates of their centers,
        iii) frame number associated with each subimage
    """
    subimages_all, com_all, frames_all = [], [], []
    for i, (img, coord) in enumerate(
            zip(imgdata, coordinates.values())):
        coord_i = coord[np.where(coord[:, 2] == coord_class)][:, :2]
        stack_i, com_i = get_imgstack(img, coord_i, window_size)
        if stack_i is None:
            continue
        subimages_all.append(stack_i)
        com_all.append(com_i)
        frames_all.append(np.ones(len(com_i), int) * i)
    subimages_all = np.concatenate(subimages_all, axis=0)
    com_all = np.concatenate(com_all, axis=0)
    frames_all = np.concatenate(frames_all, axis=0)
    return subimages_all, com_all, frames_all


def combine_classes(coord_class_dict, classes_to_combine, renumerate=True):
    """
    Combines classes in a dictionary from atomnet.locator or atomnet.predictor output
    """
    coord_class_dict_ = copy.deepcopy(coord_class_dict)
    for i in range(len(coord_class_dict_)):
        coord_class_dict_[i][:, -1] = combine_classes_(
            coord_class_dict_[i][:, -1], classes_to_combine)
    if renumerate:
        coord_class_dict_ = renumerate_classes(coord_class_dict_)
    return coord_class_dict_


def combine_classes_(classes_all, classes_to_combine):
    """
    Given a list of classes to combine substitutes listed classes
    with a minimum value from the list
    """
    for comb in classes_to_combine:
        cls_min = min(comb)
        for c in comb:
            classes_all[classes_all == c] = cls_min
    return classes_all


def renumerate_classes_(classes, start_from_1=True):
    """
    Renumerate classes such that they are ordered starting from 1 or 0
    with an increment of 1
    """
    diff = np.unique(classes) - np.arange(len(np.unique(classes)))
    diff_d = {cl: d for d, cl in zip(diff, np.unique(classes))}
    classes_renum = [cl - diff_d[cl] for cl in classes]
    classes_renum = np.array(classes_renum, dtype=np.float)
    if start_from_1:
        classes_renum = classes_renum + 1
    return classes_renum


def renumerate_classes(coord_class_dict, start_from_1=True):
    """
    Renumerate classes in a dictionary from atomnet.locator or atomnet.predictor output
    such that they are ordered starting from 1 or 0 with an increment of 1
    """
    coord_class_dict_ = copy.deepcopy(coord_class_dict)
    for i in range(len(coord_class_dict)):
        coord_class_dict_[i][:, -1] = renumerate_classes_(
            coord_class_dict_[i][:, -1], start_from_1=True)
    return coord_class_dict_


class MakeAtom:
    """
    Creates an image of atom modelled as
    2D Gaussian and a corresponding mask
    """
    def __init__(self, sc=5, r_mask=3, intensity=1, theta=0, offset=0):
        """
        Args:
            sc (float): scale parameter, which determines Gaussian width
            r_mask (float): radius of mask corresponding to atom
            theta (float): parameter of 2D gaussian function
            offset (float): parameter of 2D gaussian function
        """
        if sc % 2 == 0:
            sc += 1
        self.xo, self.yo = sc/2, sc/2
        x = np.linspace(0, sc, sc)
        y = np.linspace(0, sc, sc)
        self.x, self.y = np.meshgrid(x, y)
        self.sigma_x, self.sigma_y = sc/4, sc/4
        self.intensity = intensity
        self.theta = theta
        self.offset = offset
        self.r_mask = r_mask

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

    def circularmask(self, image, radius):
        """
        Returns a mask with specified radius
        """
        h, w = self.x.shape
        X, Y = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X-self.xo+0.5)**2 + (Y-self.yo+0.5)**2)
        mask = dist_from_center <= radius
        image[~mask] = 0
        return image

    def gen_atom_mask(self):
        """
        Creates a mask for specific type of atom
        """
        atom = self.atom2dgaussian()
        mask = self.circularmask(atom.copy(), self.r_mask/2)
        mask = mask[np.min(np.where(mask > 0)[0]):
                    np.max(np.where(mask > 0)[0]+1),
                    np.min(np.where(mask > 0)[1]):
                    np.max(np.where(mask > 0)[1])+1]
        mask[mask > 0] = 1

        return atom, mask


def create_lattice_mask(lattice, xy_atoms, *args, **kwargs):
    """
    Given experimental image and *xy* atomic coordinates
    creates ground truth image. Currently works only for the case
    where all atoms are one class. Notice that it will round fractional pixels.

    Args:
        lattice (2D numpy array):
            Experimental image as 2D numpy array
        xy_atoms (2 x N numpy array):
            Position of atoms in the experimental data
        *arg (python function):
            Function that creates a 2D numpy array with atom and
            corresponding mask for each atomic coordinate. It must have
            two parameters, 'scale' and 'rmask' that control sizes of simulated
            atom and corresponding mask

            Example:

            >>> def create_atomic_mask(scale=7, rmask=5):
            >>>     atom = MakeAtom(r).atom2dgaussian()
            >>>     _, mask = cv2.threshold(atom, thresh, 1, cv2.THRESH_BINARY)
            >>>     return atom, mask

        **scale: int
            Controls the atom size (width of 2D Gaussian)
        **rmask: int
            Controls the atomic mask size
    Returns:
        2D numpy array with ground truth data
    """
    if len(args) == 1:
        create_mask_func = args[0]
    else:
        create_mask_func = create_atom_mask_pair
    scale = kwargs.get("scale", 7)
    rmask = kwargs.get("rmask", 5)
    lattice_mask = np.zeros_like(lattice)
    for i in range(xy_atoms.shape[-1]):
        x, y = xy_atoms[:, i]
        x = int(np.around(x))
        y = int(np.around(y))
        _, mask = create_mask_func(scale, rmask)
        r_m = mask.shape[0] / 2
        r_m1 = int(r_m + .5)
        r_m2 = int(r_m - .5)
        lattice_mask[x-r_m1:x+r_m2, y-r_m1:y+r_m2] = mask
    return lattice_mask


def create_multiclass_lattice_mask(imgdata, coord_class_dict, *args, **kwargs):
    """
    Given a stack of experimental images and dictionary with atomic coordinates and classes
    creates a ground truth image. Notice that it will round fractional pixels.

    Args:
        lattice (3D numpy array):
            Experimental image as 2D numpy array
        coord_class_dict (dict or N x 3 numpy array):
            Dictionary with arrays containing coordiantes and classes for each atom/defect
            In each array, the first two columns are position of atoms.
            The third column is the "intensity"/class of each atom.
            It is also possible to pass a single N x 3 ndarray, which will be
            wrapped into a dictioanry automatically.
        *arg (python function):
            Function that creates two 2D numpy arrays with atom and
            corresponding mask for each atomic coordinate. It must have
            three parameters, 'scale', 'rmask', and 'intensity' that control
            size and intensity of simulated atom and corresponding atomic mask
        **scale: int
            Controls the atom size (width of 2D Gaussian)
        **rmask: int
            Controls the atomic mask size

    Returns:
        4D numpy array with ground truth data or list of 3D numpy arrays
    """
    if np.ndim(imgdata) == 2:
        imgdata = imgdata[None, ...]
    if isinstance(coord_class_dict, np.ndarray):
        coord_class_dict = {0: coord_class_dict}
    masks = []
    for i, img in enumerate(imgdata):
        masks.append(create_multiclass_lattice_mask_(
                        img, coord_class_dict[i], *args, **kwargs))
    shapes = [m.shape for m in masks]
    if len(set(shapes)) <= 1:
        masks = np.array(masks)
    return masks


def create_multiclass_lattice_mask_(lattice, xyz_atoms, *args, **kwargs):
    """
    Given experimental image and *xyz* atomic coordinates
    creates ground truth image. Notice that it will round fractional pixels.

    Args:
        lattice (2D numpy array):
            Experimental image as 2D numpy array
        xyz_atoms (N x 3 numpy array):
            The first two columns are position of atoms.
            The third column is the intensity of each atom.
        *arg (python function):
            Function that creates two 2D numpy arrays with atom and
            corresponding mask for each atomic coordinate. It must have
            three parameters, 'scale', 'rmask', and 'intensity' that control
            size and intensity of simulated atom and corresponding atomic mask
        **scale: int
            Controls the atom size (width of 2D Gaussian)
        **rmask: int
            Controls the atomic mask size

    Returns:
        3D numpy array with ground truth data
    """
    if len(args) == 1:
        create_mask_func = args[0]
    else:
        create_mask_func = create_atom_mask_pair
    scale = kwargs.get("scale", 7)
    rmask = kwargs.get("rmask", 7)
    lattice_mask = np.zeros(
        (lattice.shape[0], lattice.shape[1], len(np.unique(xyz_atoms[:, -1]))))
    if 0 in np.unique(xyz_atoms[:, -1]):
        xyz_atoms[:, -1] = xyz_atoms[:, -1] + 1
    atom_ch_d = {}
    for i, s in enumerate(np.unique(xyz_atoms[:, -1])):
        atom_ch_d[s] = i
    for atom in xyz_atoms:
        x, y, z = atom
        x = int(np.around(x))
        y = int(np.around(y))
        _, mask = create_mask_func(scale, rmask, z)
        r_m = mask.shape[0] / 2
        r_m1 = int(r_m + .5)
        r_m2 = int(r_m - .5)
        lattice_mask[x-r_m1:x+r_m2, y-r_m1:y+r_m2, atom_ch_d[z]] = mask
    lattice_mask_b = 1 - np.sum(lattice_mask, axis=-1)
    lattice_mask = np.concatenate((lattice_mask, lattice_mask_b[..., None]), axis=-1)
    lattice_mask[lattice_mask < 0] = 0
    return lattice_mask


def create_atom_mask_pair(sc=5, r_mask=5, intensity=1):
    """
    Helper function for creating atom-label pair
    """
    amaker = MakeAtom(sc, r_mask, intensity)
    atom, mask = amaker.gen_atom_mask()
    return atom, mask


def extract_patches_(lattice_im, lattice_mask, patch_size, num_patches, **kwargs):
    """
    Extracts subimages of the selected size from the 'mother" image and mask
    """
    rs = kwargs.get("random_state", 0)
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    images = extract_patches_2d(
        lattice_im, patch_size, max_patches=num_patches, random_state=rs)
    labels = extract_patches_2d(
        lattice_mask, patch_size, max_patches=num_patches, random_state=rs)
    return images, labels


def extract_patches(images, masks, patch_size, num_patches, **kwargs):
    """
    Takes batch of images and batch of corresponding masks as an input
    and for each image-mask pair it extracts stack of subimages (patches)
    of the selected size.
    """
    if np.ndim(images) == 2:
        images = images[None, ...]
    images_aug, masks_aug = [], []
    for im, ma in zip(images, masks):
        im_aug, ma_aug = extract_patches_(
            im, ma, patch_size, num_patches, **kwargs)
        images_aug.append(im_aug)
        masks_aug.append(ma_aug)
    images_aug = np.concatenate(images_aug, axis=0)
    masks_aug = np.concatenate(masks_aug, axis=0)
    return images_aug, masks_aug


class datatransform:
    """
    Applies a sequence of pre-defined operations for data augmentation.

    Args:
        n_channels (int):
            Number of classes (channels) in the ground truth
        dim_order_in (str):
            Channel first or channel last ordering in the input masks
        dim_order_out (str):
            Channel first or channel last ordering in the output masks
        seed (int):
            Determenism
        **rotation (bool):
            Rotating image by +- 90 deg (if image is square)
            and horizontal/vertical flipping.
        **zoom (bool or int):
            Zooming-in by a specified zoom factor (Default: 2)
            Note that a zoom window is always square
        **gauss (bool or list ot tuple):
            Gaussian noise. You can pass min and max values as a list/tuple
            (Default [min, max] range: [0, 50])
        **poisson (bool or list ot tuple):
            Poisson noise. You can pass min and max values as a list/tuple
            (Default [min, max] range: [30, 40])
        **salt_and_pepper (bool or list ot tuple):
            Salt and pepper noise. You can pass min and max values as a list/tuple
            (Default [min, max] range: [0, 50])
        **blur (bool or list ot tuple):
            Gaussian blurring. You can pass min and max values as a list/tuple
            (Default [min, max] range: [1, 50])
        **contrast (bool or list ot tuple):
            Contrast level. You can pass min and max values as a list/tuple
            (Default [min, max] range: [5, 20])
        **background (bool):
            Adds/substracts asymmetric 2D gaussian of random width and intensity
            from the image
        **resize (tuple):
            Values for image resizing
            [downscale factor (default: 2), upscale factor (default:1.5)]
    """
    def __init__(self,
                 n_channels,
                 dim_order_in='channel_last',
                 dim_order_out='channel_first',
                 squeeze_channels=False,
                 seed=None,
                 **kwargs):

        self.ch = n_channels
        self.dim_order_in = dim_order_in
        self.dim_order_out = dim_order_out
        self.squeeze = squeeze_channels
        self.rotation = kwargs.get('rotation')
        self.background = kwargs.get('background')
        self.gauss = kwargs.get('gauss')
        if self.gauss is True:
            self.gauss = [0, 50]
        self.jitter = kwargs.get("jitter")
        if self.jitter is True:
            self.jitter = [0, 50]
        self.poisson = kwargs.get('poisson')
        if self.poisson is True:
            self.poisson = [30, 40]
        self.salt_and_pepper = kwargs.get('salt_and_pepper')
        if self.salt_and_pepper is True:
            self.salt_and_pepper = [0, 50]
        self.blur = kwargs.get('blur')
        if self.blur is True:
            self.blur = [1, 50]
        self.contrast = kwargs.get('contrast')
        if self.contrast is True:
            self.contrast = [5, 20]
        self.zoom = kwargs.get('zoom')
        if self.zoom is True:
            self.zoom = 2  # [min, max] zoom
        self.resize = kwargs.get('resize')
        if self.resize is True:
            self.resize = [2, 1.5]
        if seed is not None:
            np.random.seed(seed)

    def apply_gauss(self, X_batch, y_batch):
        """
        Random application of gaussian noise to each training inage in a stack
        """
        n, h, w = X_batch.shape[0:3]
        X_batch_noisy = np.zeros((n, h, w))
        for i, img in enumerate(X_batch):
            gauss_var = np.random.randint(self.gauss[0], self.gauss[1])
            img_ = random_noise(
                img, mode='gaussian', var=1e-4*gauss_var)
            X_batch_noisy[i] = img_
        return X_batch_noisy, y_batch

    def apply_jitter(self, X_batch, y_batch):
        """
        Random application of jitter noise to each training image in a stack
        """
        n, h, w = X_batch.shape[0:3]
        X_batch_noisy = np.zeros((n, h, w))
        for i, img in enumerate(X_batch):
            jitter_amount = np.random.randint(self.jitter[0], self.jitter[1]) / 10
            shift_arr = stats.poisson.rvs(jitter_amount, loc=0, size=h)
            img_ = np.array([np.roll(row, z) for row, z in zip(img, shift_arr)])
            X_batch_noisy[i] = img_
        return X_batch_noisy, y_batch

    def apply_poisson(self, X_batch, y_batch):
        """
        Random application of poisson noise to each training inage in a stack
        """
        def make_pnoise(image, l):
            vals = len(np.unique(image))
            vals = (50/l) ** np.ceil(np.log2(vals))
            image_n_filt = np.random.poisson(image * vals) / float(vals)
            return image_n_filt
        n, h, w = X_batch.shape[0:3]
        X_batch_noisy = np.zeros((n, h, w))
        for i, img in enumerate(X_batch):
            poisson_l = np.random.randint(self.poisson[0], self.poisson[1])
            img = make_pnoise(img, poisson_l)
            X_batch_noisy[i] = img
        return X_batch_noisy, y_batch

    def apply_sp(self, X_batch, y_batch):
        """
        Random application of salt & pepper noise to each training inage in a stack
        """
        n, h, w = X_batch.shape[0:3]
        X_batch_noisy = np.zeros((n, h, w))
        for i, img in enumerate(X_batch):
            sp_amount = np.random.randint(
                self.salt_and_pepper[0], self.salt_and_pepper[1])
            img = random_noise(img, mode='s&p', amount=sp_amount*1e-3)
            X_batch_noisy[i] = img
        return X_batch_noisy, y_batch

    def apply_blur(self, X_batch, y_batch):
        """
        Random blurring of each training image in a stack
        """
        n, h, w = X_batch.shape[0:3]
        X_batch_noisy = np.zeros((n, h, w))
        for i, img in enumerate(X_batch):
            blur_amount = np.random.randint(self.blur[0], self.blur[1])
            img = ndimage.filters.gaussian_filter(img, blur_amount*5e-2)
            X_batch_noisy[i] = img
        return X_batch_noisy, y_batch

    def apply_contrast(self, X_batch, y_batch):
        """
        Randomly change level of contrast of each training image on a stack
        """
        n, h, w = X_batch.shape[0:3]
        X_batch_noisy = np.zeros((n, h, w))
        for i, img in enumerate(X_batch):
            clevel = np.random.randint(self.contrast[0], self.contrast[1])
            img = exposure.adjust_gamma(img, clevel/10)
            X_batch_noisy[i] = img
        return X_batch_noisy, y_batch

    def apply_zoom(self, X_batch, y_batch):
        """
        Zoom-in achieved by cropping image and then resizing
        to the original size. The zooming window is a square.
        """
        n, h, w = X_batch.shape[0:3]
        shortdim = min([w, h])
        zoom_values = np.arange(int(shortdim // self.zoom), shortdim + 8, 8)
        X_batch_z = np.zeros((n, shortdim, shortdim))
        y_batch_z = np.zeros((n, shortdim, shortdim, self.ch))
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            zv = np.random.choice(zoom_values)
            img = img[
                (h // 2) - (zv // 2): (h // 2) + (zv // 2),
                (w // 2) - (zv // 2): (w // 2) + (zv // 2)]
            gt = gt[
                (h // 2) - (zv // 2): (h // 2) + (zv // 2),
                (w // 2) - (zv // 2): (w // 2) + (zv // 2)]
            img = cv2.resize(
                img, (shortdim, shortdim), interpolation=cv2.INTER_CUBIC)
            gt = cv2.resize(
                gt, (shortdim, shortdim), interpolation=cv2.INTER_CUBIC)
            img = np.clip(img, 0, 1)
            gt = np.around(gt)
            if len(gt.shape) != 3:
                gt = np.expand_dims(gt, axis=2)
            X_batch_z[i] = img
            y_batch_z[i] = gt
        return X_batch_z, y_batch_z

    def apply_background(self, X_batch, y_batch):
        """
        Emulates thickness variation in STEM or height variation in STM
        """
        def gauss2d(xy, x0, y0, a, b, fwhm):
            return np.exp(-np.log(2)*(a*(xy[0]-x0)**2 + b*(xy[1]-y0)**2) / fwhm**2)
        n, h, w = X_batch.shape[0:3]
        X_batch_b = np.zeros((n, h, w))
        x, y = np.meshgrid(
            np.linspace(0, h, h), np.linspace(0, w, w), indexing='ij')
        for i, img in enumerate(X_batch):
            x0 = np.random.randint(0, h - h // 4)
            y0 = np.random.randint(0, w - w // 4)
            a, b = np.random.randint(10, 20, 2) / 10
            fwhm = np.random.randint(min([h, w]) // 4, min([h, w]) - min([h, w]) // 2)
            Z = gauss2d([x, y], x0, y0, a, b, fwhm)
            img = img + 0.05 * np.random.randint(-10, 10) * Z
            X_batch_b[i] = img
        return X_batch_b, y_batch

    def apply_rotation(self, X_batch, y_batch):
        """
        Flips and rotates training images and correponding ground truth images
        """
        n, h, w = X_batch.shape[0:3]
        X_batch_r = np.zeros((n, h, w))
        y_batch_r = np.zeros((n, h, w, self.ch))
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            flip_type = np.random.randint(-1, 3)
            if flip_type == 3 and h == w:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            elif flip_type == 2 and h == w:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                img = cv2.flip(img, flip_type)
                gt = cv2.flip(gt, flip_type)
            if len(gt.shape) != 3:
                gt = np.expand_dims(gt, axis=2)
            X_batch_r[i] = img
            y_batch_r[i] = gt
        return X_batch_r, y_batch_r

    def apply_imresize(self, X_batch, y_batch):
        """
        Resizes training images and corresponding ground truth images
        """
        rs_factor_d = 1 / self.resize[0]
        rs_factor_u = self.resize[1]
        n, h, w = X_batch.shape[0:3]
        s, p = 0.03, 8
        while (np.round((h * s), 7) % p != 0
               and np.round((w * s), 7) % p != 0):
            s += 1e-5
        rs_h = (np.arange(rs_factor_d, rs_factor_u, s) * h).astype(np.int64)
        rs_w = (np.arange(rs_factor_d, rs_factor_u, s) * w).astype(np.int64)
        rs_idx = np.random.randint(len(rs_h))
        if X_batch.shape[1:3] == (rs_h[rs_idx], rs_w[rs_idx]):
            return X_batch, y_batch
        X_batch_r = np.zeros((n, rs_h[rs_idx], rs_w[rs_idx]))
        y_batch_r = np.zeros((n, rs_h[rs_idx], rs_w[rs_idx], self.ch))
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            rs_method = cv2.INTER_AREA if rs_h[rs_idx] < h else cv2.INTER_CUBIC
            img = cv2.resize(img, (rs_w[rs_idx], rs_h[rs_idx]), rs_method)
            gt = cv2.resize(gt, (rs_w[rs_idx], rs_h[rs_idx]), rs_method)
            gt = np.around(gt)
            if len(gt.shape) < 3:
                gt = np.expand_dims(gt, axis=-1)
            X_batch_r[i] = img
            y_batch_r[i] = gt
        return X_batch_r, y_batch_r

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
        images = (images - images.min()) / images.ptp()
        if self.rotation:
            images, masks = self.apply_rotation(images, masks)
        if isinstance(self.zoom, int):
            images, masks = self.apply_zoom(images, masks)
        if isinstance(self.resize, list) or isinstance(self.resize, tuple):
            images, masks = self.apply_imresize(images, masks)
        if isinstance(self.gauss, list) or isinstance(self.gauss, tuple):
            images, masks = self.apply_gauss(images, masks)
        if isinstance(self.jitter, list) or isinstance(self.jitter, tuple):
            images, masks = self.apply_jitter(images, masks)
        if isinstance(self.poisson, list) or isinstance(self.poisson, tuple):
            images, masks = self.apply_poisson(images, masks)
        if isinstance(self.salt_and_pepper, list) or isinstance(self.salt_and_pepper, tuple):
            images, masks = self.apply_sp(images, masks)
        if isinstance(self.blur, list) or isinstance(self.blur, tuple):
            images, masks = self.apply_blur(images, masks)
        if isinstance(self.contrast, list) or isinstance(self.contrast, tuple):
            images, masks = self.apply_contrast(images, masks)
        if self.background:
            images, masks = self.apply_background(images, masks)
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
        images = (images - images.min()) / images.ptp()
        return images, masks


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


def unsqueeze_channels(labels, n_channels):
    """
    Separates pixels with different values into different channels
    """
    if n_channels == 1:
        return labels
    n, h, w = labels.shape
    lbl_all = np.zeros((n, h, w, n_channels))
    for i, lbl in enumerate(labels):
        lbl = np.array(OneHotEncoder().fit_transform(lbl.reshape(-1,1)).todense())
        lbl = lbl.reshape(h, w, n_channels)
        lbl_all[i] = lbl
    return lbl_all.transpose([0, -1, 1, 2])


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
