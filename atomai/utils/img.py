"""
img.py
======

Helper functions for working with images. Includes image pre- and post-processing
for model training and prediction and generation of stack of subimages

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Tuple, Optional, Dict, Union, List, Type
from collections import OrderedDict
import warnings
import numpy as np
import cv2
from scipy import fftpack, ndimage
from sklearn.feature_extraction.image import extract_patches_2d
import torch
from .coords import remove_edge_coord

input_data_types = Union[np.ndarray, List[np.ndarray], Dict[int, np.ndarray]]


def preprocess_training_data(images_all: input_data_types,
                             labels_all: input_data_types,
                             images_test_all: input_data_types,
                             labels_test_all: input_data_types,
                             batch_size: int) -> Tuple[List[np.ndarray]]:
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


def init_torch_dataloaders(X_train: Union[List, np.ndarray],
                           y_train: Union[List, np.ndarray],
                           X_test: Union[List, np.ndarray],
                           y_test: Union[List, np.ndarray],
                           batch_size: int, num_classes: int
                           ) -> Tuple[Type[torch.utils.data.DataLoader]]:
    """
    Returns train and test dataloaders in a native PyTorch format
    """
    if not (type(X_train) == type(y_train) ==
            type(X_test) == type(y_test)):
        raise AssertionError(
            "Provide all training and test data in the same format")
    if isinstance(X_train, list):
        tor = lambda x: torch.from_numpy(np.concatenate(x))
        X_train, y_train = tor(X_train).float(), tor(y_train)
        X_test, y_test = tor(X_test).float(), tor(y_test)
    if num_classes > 1:
        y_train = y_train.long()
        y_test = y_test.long()
    else:
        y_train = y_train.float()
        y_test = y_test.float()
    if torch.cuda.is_available():
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=batch_size)
    return train_loader, test_loader


def torch_format(image_data: np.ndarray) -> torch.Tensor:
    """
    Reshapes (if needed), normalizes and converts image data
    to pytorch format for model training and prediction

    Args:
        image_data (3D or 4D numpy array):
            Image stack with dimensions (n_batches x height x width)
            or (n_batches x 1 x height x width)
    """
    if image_data.ndim not in [3, 4]:
        raise AssertionError(
            "Provide image(s) as 3D (n, h, w) or 4D (n, 1, h, w) tensor")
    if np.ndim(image_data) == 3:
        image_data = np.expand_dims(image_data, axis=1)
    elif np.ndim(image_data) == 4 and image_data.shape[1] != 1:
        raise AssertionError(
            "4D image tensor must have (n, 1, h, w) dimensions")
    else:
        pass
    image_data = (image_data - np.amin(image_data))/np.ptp(image_data)
    image_data = torch.from_numpy(image_data).float()
    return image_data


def img_resize(image_data: np.ndarray, rs: Tuple[int],
               round_: bool = False) -> np.ndarray:
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
    if rs[0] != rs[1]:
        rs = (rs[1], rs[0])
    if image_data.shape[1:3] == rs:
        return image_data.copy()
    image_data_r = np.zeros(
        (image_data.shape[0], rs[0], rs[1]))
    for i, img in enumerate(image_data):
        img = cv_resize(img, rs, round_)
        image_data_r[i, :, :] = img
    return image_data_r


def cv_resize(img: np.ndarray, rs: Tuple[int],
              round_: bool = False) -> np.ndarray:
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
    rs = (rs[1], rs[0])
    rs_method = cv2.INTER_AREA if img.shape[0] < rs[0] else cv2.INTER_CUBIC
    img_rs = cv2.resize(img, rs, interpolation=rs_method)
    if round_:
        img_rs = np.round(img_rs)
    return img_rs


def cv_resize_stack(imgdata: np.ndarray, rs: Union[int, Tuple[int]],
                    round_: bool = False) -> np.ndarray:
    """
    Resizes a 3D stack of images

    Args:
        imgdata (3D numpy array): stack of 3D images to be resized
        rs (tuple or int): target height and width
        round_(bool): rounding (in case of labeled pixels)

    Returns:
        Resized image
    """
    rs = (rs, rs) if isinstance(rs, int) else rs
    if imgdata.shape[1:3] == rs:
        return imgdata
    imgdata_rs = np.zeros((imgdata.shape[0], rs[0], rs[1]))
    for i, img in enumerate(imgdata):
        img_rs = cv_resize(img, rs, round_)
        imgdata_rs[i] = img_rs
    return imgdata_rs


def img_pad(image_data: np.ndarray, pooling: int) -> np.ndarray:
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


def get_imgstack(imgdata: np.ndarray,
                 coord: np.ndarray,
                 r: int) -> Tuple[np.ndarray]:
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

        - Stack of subimages
        - (x, y) coordinates of their centers
    """
    img_cr_all = []
    com = []
    for c in coord:
        cx = int(np.around(c[0]))
        cy = int(np.around(c[1]))
        if r % 2 != 0:
            img_cr = np.copy(
                imgdata[cx-r//2:cx+r//2+1,
                        cy-r//2:cy+r//2+1])
        else:
            img_cr = np.copy(
                imgdata[cx-r//2:cx+r//2,
                        cy-r//2:cy+r//2])
        if img_cr.shape[0:2] == (int(r), int(r)):
            img_cr_all.append(img_cr[None, ...])
            com.append(c[None, ...])
    if len(img_cr_all) == 0:
        return None, None
    img_cr_all = np.concatenate(img_cr_all, axis=0)
    com = np.concatenate(com, axis=0)
    return img_cr_all, com


def imcrop_randpx(img: np.ndarray, window_size: int, num_images: int,
                  random_state: int = 0) -> Tuple[np.ndarray]:
    """
    Extracts subimages at random pixels

    Returns:
        2-element tuple containing

        - Stack of subimages
        - (x, y) coordinates of their centers
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


def imcrop_randcoord(img: np.ndarray, coord: np.ndarray,
                     window_size: int, num_images: int,
                     random_state: int = 0) -> Tuple[np.ndarray]:
    """
    Extracts subimages at random coordinates

    Returns:
        2-element tuple containing

        - Stack of subimages
        - (x, y) coordinates of their centers
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


def extract_random_subimages(imgdata: np.ndarray, window_size: int, num_images: int,
                             coordinates: Optional[Dict[int, np.ndarray]] = None,
                             **kwargs: int) -> Tuple[np.ndarray]:
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

        - stack of subimages
        - (x, y) coordinates of their centers
        - frame number associated with each subimage
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


def extract_subimages(imgdata: np.ndarray,
                      coordinates: Union[Dict[int, np.ndarray], np.ndarray],
                      window_size: int, coord_class: int = 0) -> Tuple[np.ndarray]:
    """
    Extracts subimages centered at certain atom class/type
    (usually from a neural network output)

    Args:
        imgdata (numpy array):
            4D stack of images (n, height, width, channel).
            It is also possible to pass a single 2D image.
        coordinates (dict or N x 2 numpy arry): Prediction from atomnet.locator
            (can be from other source but must be in the same format)
            Each element is a :math:`N \\times 3` numpy array,
            where *N* is a number of detected atoms/defects,
            the first 2 columns are *xy* coordinates
            and the third columns is class (starts with 0).
            It is also possible to pass N x 2 numpy array if the corresponding
            imgdata is a single 2D image.
        window_size (int):
            Side of the square for subimage cropping
        coord_class (int):
            Class of atoms/defects around around which the subimages
            will be cropped (3rd column in the atomnet.locator output)

    Returns:
        3-element tuple containing

        - stack of subimages,
        - (x, y) coordinates of their centers,
        - frame number associated with each subimage
    """
    if isinstance(coordinates, np.ndarray):
        coordinates = np.concatenate((
            coordinates, np.zeros((coordinates.shape[0], 1))), axis=-1)
        coordinates = {0: coordinates}
    if np.ndim(imgdata) == 2:
        imgdata = imgdata[None, ..., None]
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
    if len(subimages_all) > 0:
        subimages_all = np.concatenate(subimages_all, axis=0)
        com_all = np.concatenate(com_all, axis=0)
        frames_all = np.concatenate(frames_all, axis=0)
    return subimages_all, com_all, frames_all


def extract_patches_(lattice_im: np.ndarray, lattice_mask: np.ndarray,
                     patch_size: int, num_patches: int, **kwargs: int
                     ) -> Tuple[np.ndarray]:
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


def extract_patches(images: np.ndarray, masks: np.ndarray,
                    patch_size: int, num_patches: int, **kwargs: int
                    ) -> Tuple[np.ndarray]:
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


def FFTmask(imgsrc: np.ndarray, maskratio: int = 10) -> Tuple[np.ndarray]:
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


def FFTsub(imgsrc: np.ndarray, imgfft: np.ndarray) -> np.ndarray:
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


def threshImg(diff: np.ndarray,
              threshL: float = 0.25,
              threshH: float = 0.75) -> np.ndarray:
    """
    Takes in difference image, low and high thresold values,
    and outputs a map of all defects.
    """
    threshIL = diff < threshL
    threshIH = diff > threshH
    threshI = threshIL + threshIH
    return threshI


def crop_borders(imgdata: np.ndarray, thresh: float = 0) -> np.ndarray:
    """
    Crops image border where all values are zeros

    Args:
        imgdata (numpy array): 3D numpy array (h, w, c)
        thresh: border values to crop

    Returns: Cropped array
    """
    def crop(img):
        mask = img > thresh
        img = img[np.ix_(mask.any(1), mask.any(0))]
        return img

    imgdata_cr = [crop(imgdata[..., i]) for i in range(imgdata.shape[-1])]

    return np.array(imgdata_cr).transpose(1, 2, 0)


def get_coord_grid(imgdata: np.ndarray, step: int,
                   return_dict: bool = True
                   ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Generate a square coordinate grid for every image in a stack. Returns coordinates
    in a dictionary format (same format as generated by atomnet.predictor)
    that can be used as an input for utility functions extracting subimages
    and atomstat.imlocal class

    Args:
        imgdata (numpy array): 2D or 3D numpy array
        step (int): distance between grid points
        return_dict (bool): returns coordiantes as a dictionary (same format as atomnet.predictor)

    Returns:
        Dictionary or numpy array with coordinates
    """
    if np.ndim(imgdata) == 2:
        imgdata = np.expand_dims(imgdata, axis=0)
    coord = []
    for i in range(0, imgdata.shape[1], step):
        for j in range(0, imgdata.shape[2], step):
            coord.append(np.array([i, j]))
    coord = np.array(coord)
    if return_dict:
        coord = np.concatenate((coord, np.zeros((coord.shape[0], 1))), axis=-1)
        coordinates_dict = {i: coord for i in range(imgdata.shape[0])}
        return coordinates_dict
    coordinates = [coord for _ in range(imgdata.shape[0])]
    return np.concatenate(coordinates, axis=0)


def cv_thresh(imgdata: np.ndarray,
              threshold: float = .5):
    """
    Wrapper for opencv binary threshold method.
    Returns thresholded image.
    """
    _, thresh = cv2.threshold(
                    imgdata,
                    threshold, 1,
                    cv2.THRESH_BINARY)
    return thresh


def filter_cells_(imgdata: np.ndarray,
                  im_thresh: float = .5,
                  blob_thresh: int = 150,
                  filter_: str = 'below') -> np.ndarray:
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


def get_contours(imgdata: np.ndarray) -> List[np.ndarray]:
    """
    Extracts object contours from image data
    (image data must be binary thresholded)
    """
    imgdata_ = cv2.convertScaleAbs(imgdata)
    contours = cv2.findContours(
        imgdata_.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    return contours


def filter_cells(imgdata: np.ndarray,
                 im_thresh: float = 0.5,
                 blob_thresh: int = 50,
                 filter_: str = 'below') -> np.ndarray:
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


def get_blob_params(nn_output: np.ndarray, im_thresh: float,
                    blob_thresh: int, filter_: str = 'below') -> Dict:
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

