"""
preproc.py
======

Helper functions for prerocessing training/validation data.

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Tuple, Optional, Union, List, Type
import warnings
import numpy as np
import torch

from sklearn.model_selection import train_test_split


def num_classes_from_labels(labels: np.ndarray) -> int:
    """
    Gets number of classes from labels (aka ground truth aka masks)

    Args:
        labels (numpy array):
            ground truth (aka masks aka labels) for semantic segmentation

    Returns:
        number of classes
    """
    uval = np.unique(labels)
    if min(uval) != 0:
        raise AssertionError("Labels should start from 0")
    for i, j in zip(uval, uval[1:]):
        if j - i != 1:
            raise AssertionError("Mask values should be in range between "
                                 "0 and total number of classes "
                                 "with an increment of 1")
    num_classes = len(uval)
    if num_classes == 2:
        num_classes = num_classes - 1
    return num_classes


def check_image_dims(X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     num_classes: int
                     ) -> Tuple[np.ndarray]:
    """
    Adds if necessary pseudo-dimension of 1 (channel dimensions)
    to images and masks
    """
    if X_train.ndim == 3:
        warnings.warn(
            'Adding a channel dimension of 1 to training images',
            UserWarning)
        X_train = X_train[:, np.newaxis]
    if X_test.ndim == 3:
        warnings.warn(
            'Adding a channel dimension of 1 to test images',
            UserWarning)
        X_test = X_test[:, np.newaxis]
    if num_classes == 1 and y_train.ndim == 3:
        warnings.warn(
            'Adding a channel dimension of 1 to training labels',
            UserWarning)
        y_train = y_train[:, np.newaxis]
    if num_classes == 1 and y_test.ndim == 3:
        warnings.warn(
            'Adding a channel dimension of 1 to test labels',
            UserWarning)
        y_test = y_test[:, np.newaxis]

    return X_train, y_train, X_test, y_test


def check_signal_dims(X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> Tuple[np.ndarray]:
    """
    Adds if necessary a pseudo-dimension of 1 (channel dimensions)
    to images and spectra
    """
    if X_train.ndim > y_train.ndim:
        if X_train.ndim == 3:
            warnings.warn(
                'Adding a channel dimension of 1 to training images',
                UserWarning)
            X_train = X_train[:, np.newaxis]
        if X_test.ndim == 3:
            warnings.warn(
                'Adding a channel dimension of 1 to test images',
                UserWarning)
            X_test = X_test[:, np.newaxis]
        if y_train.ndim == 2:
            warnings.warn(
                'Adding a channel dimension of 1 to training spectra',
                UserWarning)
            y_train = y_train[:, np.newaxis]
        if y_test.ndim == 2:
            warnings.warn(
                'Adding a channel dimension of 1 to test spectra',
                UserWarning)
            y_test = y_test[:, np.newaxis]

    elif X_train.ndim < y_train.ndim:
        if X_train.ndim == 2:
            warnings.warn(
                'Adding a channel dimension of 1 to training images',
                UserWarning)
            X_train = X_train[:, np.newaxis]
        if X_test.ndim == 2:
            warnings.warn(
                'Adding a channel dimension of 1 to test images',
                UserWarning)
            X_test = X_test[:, np.newaxis]
        if y_train.ndim == 3:
            warnings.warn(
                'Adding a channel dimension of 1 to training spectra',
                UserWarning)
            y_train = y_train[:, np.newaxis]
        if y_test.ndim == 3:
            warnings.warn(
                'Adding a channel dimension of 1 to test spectra',
                UserWarning)
            y_test = y_test[:, np.newaxis]

        same_dim1 = X_train.shape[1:] == X_test.shape[1:]
        same_dim2 = y_train.shape[1:] == y_test.shape[1:]
        if not all([same_dim1, same_dim2]):
            raise ValueError("The image/spectra dimensions must be" +
                             " the same for training and test data")

    return X_train, y_train, X_test, y_test


def get_array_memsize(X_arr: Union[np.ndarray, torch.Tensor],
                      precision: str = "single") -> float:
    """
    Returns memory size of numpy array or torch tensor
    """
    if X_arr is None:
        return 0
    if isinstance(X_arr, torch.Tensor):
        X_arr = X_arr.cpu().numpy()
    arrsize = X_arr.nbytes
    if precision == "single":
        if X_arr.dtype in ["float64", "int64"]:
            arrsize = arrsize / 2
        elif X_arr.dtype in ["float32", "int32"]:
            pass
        else:
            warnings.warn(
                "Data type is not understood", UserWarning)
    elif precision == "double":
        if X_arr.dtype in ["float32", "int32"]:
            arrsize = arrsize * 2
        elif X_arr.dtype in ["float64", "int64"]:
            pass
        else:
            warnings.warn(
                "Data type is not understood", UserWarning)
    else:
        raise NotImplementedError(
            "Specify 'single' or 'double' precision type")
    return arrsize


def array2list_(x: Union[np.ndarray, torch.Tensor],
                batch_size: int, store_on_cpu: bool = False
                ) -> Union[List[torch.Tensor], List[np.ndarray]]:
    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise TypeError("Provide data as numpy array or torch tensor")

    if isinstance(x, torch.Tensor):
        device = 'cuda' if torch.cuda.is_available() and not store_on_cpu else 'cpu'
        x = x.to(device)
    n_batches = int(np.divmod(x.shape[0], batch_size)[0])
    split = np.split if isinstance(x, np.ndarray) else torch.chunk
    return split(x[:n_batches*batch_size], n_batches)


def array2list(X_train: Union[np.ndarray, torch.Tensor],
               y_train: Union[np.ndarray, torch.Tensor],
               X_test: Union[np.ndarray, torch.Tensor],
               y_test: Union[np.ndarray, torch.Tensor],
               batch_size: int, memory_alloc: float = 4
               ) -> Union[Tuple[List[np.ndarray]], Tuple[List[torch.Tensor]]]:
    """
    Splits train and test numpy arrays or torch tensors into lists of
    arrays/tensors of a specified size. The remainders are not included.
    """
    all_data = [X_train, y_train, X_test, y_test]
    arrsize = sum([get_array_memsize(x) for x in all_data])
    store_on_cpu = (arrsize / 1e9) > memory_alloc
    X_train = array2list_(X_train, batch_size, store_on_cpu)
    y_train = array2list_(y_train, batch_size, store_on_cpu)
    X_test = array2list_(X_test, batch_size, store_on_cpu)
    y_test = array2list_(y_test, batch_size, store_on_cpu)
    return X_train, y_train, X_test, y_test


def preprocess_training_image_data_(images_all: Union[np.ndarray, torch.Tensor],
                                    labels_all: Union[np.ndarray, torch.Tensor],
                                    images_test_all: Union[np.ndarray, torch.Tensor],
                                    labels_test_all: Union[np.ndarray, torch.Tensor],
                                    ) -> Tuple[torch.Tensor]:
    """
    Preprocess training and test image data
    """
    all_data = (images_all, labels_all, images_test_all, labels_test_all)
    all_numpy = all([isinstance(i, np.ndarray) for i in all_data])
    all_torch = all([isinstance(i, torch.Tensor) for i in all_data])
    if not all_numpy and not all_torch:
        raise TypeError(
            "Provide training and test data in the form" +
            " of numpy arrays or torch tensors")
    num_classes = num_classes_from_labels(labels_all)
    (images_all, labels_all,
     images_test_all, labels_test_all) = check_image_dims(*all_data, num_classes)
    if all_numpy:
        images_all = torch.from_numpy(images_all)
        images_test_all = torch.from_numpy(images_test_all)
        labels_all = torch.from_numpy(labels_all)
        labels_test_all = torch.from_numpy(labels_test_all)
    images_all, images_test_all = images_all.float(), images_test_all.float()
    if num_classes > 1:
        labels_all, labels_test_all = labels_all.long(), labels_test_all.long()
    else:
        labels_all, labels_test_all = labels_all.float(), labels_test_all.float()

    return (images_all, labels_all, images_test_all,
            labels_test_all, num_classes)



def preprocess_training_image_data(images_all: Union[np.ndarray, torch.Tensor],
                                   labels_all: Union[np.ndarray, torch.Tensor],
                                   images_test_all: Union[np.ndarray, torch.Tensor],
                                   labels_test_all: Union[np.ndarray, torch.Tensor],
                                   batch_size: int, memory_alloc: float = 4
                                   ) -> Tuple[Union[List[np.ndarray], List[torch.Tensor]], int]:
    """
    Preprocess training and test image data

    Args:
        images_all (numpy array):
            4D numpy array (3D image tensors stacked along the first dim)
            representing training images
        labels_all (numpy array):
            4D (binary) / 3D (multiclass) numpy array
            where 3D / 2D images stacked along the first array dimension
            represent training labels (aka masks aka ground truth)
        images_test_all (numpy array):
            4D numpy array (3D image tensors stacked along the first dim)
            representing test images
        labels_test_all (numpy array):
            4D (binary) / 3D (multiclass) numpy array
            where 3D / 2D images stacked along the first array dimension
            represent test labels (aka masks aka ground truth)
        batch_size (int):
            Size of training and test batches
        memory_alloc (float or int):
            Threshold (in GB) for holding all training data on GPU

    Returns:
        5-element tuple containing lists of numpy arrays ot torch tensors with
        training and test data, and an integer corresponding to the number of
        classes inferred from the data.
    """
    data_all = preprocess_training_image_data_(
        images_all, labels_all, images_test_all, labels_test_all)
    num_classes = data_all[-1]
    images_all, labels_all, images_test_all, labels_test_all = array2list(
        *data_all[:-1], batch_size, memory_alloc)

    return (images_all, labels_all, images_test_all,
            labels_test_all, num_classes)


def preprocess_training_imspec_data_(X_train: Union[np.ndarray, torch.Tensor],
                                     y_train: Union[np.ndarray, torch.Tensor],
                                     X_test: Union[np.ndarray, torch.Tensor],
                                     y_test: Union[np.ndarray, torch.Tensor],
                                     ) -> Tuple[Union[List[np.ndarray], List[torch.Tensor]], Tuple[Tuple[int]]]:
        """
        Preprocesses training and test data for im2spec/spec2im models
        """
        all_data = (X_train, y_train, X_test, y_test)
        all_numpy = all([isinstance(i, np.ndarray) for i in all_data])
        all_torch = all([isinstance(i, torch.Tensor) for i in all_data])
        if not all_numpy and not all_torch:
            raise TypeError(
                "Provide training and test data in the form" +
                " of numpy arrays or torch tensors")

        X_train, y_train, X_test, y_test = check_signal_dims(
            X_train, y_train, X_test, y_test)
        in_dim = X_train.shape[2:]
        out_dim = y_train.shape[2:]

        if all_numpy:
            X_train = torch.from_numpy(X_train).float()
            y_train = torch.from_numpy(y_train).float()
            X_test = torch.from_numpy(X_test).float()
            y_test = torch.from_numpy(y_test).float()
        else:
            X_train = X_train.float()
            y_train = y_train.float()
            X_test = X_test.float()
            y_test = y_test.float()

        return X_train, y_train, X_test, y_test, (in_dim, out_dim)


def preprocess_training_imspec_data(X_train: Union[np.ndarray, torch.Tensor],
                                    y_train: Union[np.ndarray, torch.Tensor],
                                    X_test: Union[np.ndarray, torch.Tensor],
                                    y_test: Union[np.ndarray, torch.Tensor],
                                    batch_size: int, memory_alloc: float = 4
                                    ) -> Tuple[Union[List[np.ndarray], List[torch.Tensor]], Tuple[Tuple[int]]]:
        """
        Preprocesses training and test data for im2spec/spec2im models

        Args:
            X_train (numpy array):
                4D numpy array with image data (n_samples x 1 x height x width)
                or 3D numpy array with spectral data (n_samples x 1 x signal_length).
                It is also possible to pass 3D and 2D arrays by ignoring the channel dim,
                which will be added automatically.
            y_train (numpy array):
                3D numpy array with spectral data (n_samples x 1 x signal_length)
                or 4D numpy array with image data (n_samples x 1 x height x width).
                It is also possible to pass 2D and 3D arrays by ignoring the channel dim,
                which will be added automatically. Note that if your X_train data are images,
                then your y_train must be spectra and vice versa.
            X_test (numpy array):
                4D numpy array with image data (n_samples x 1 x height x width)
                or 3D numpy array with spectral data (n_samples x 1 x signal_length).
                It is also possible to pass 3D and 2D arrays by ignoring the channel dim,
                which will be added automatically.
            y_test (numpy array):
                3D numpy array with spectral data (n_samples x 1 x signal_length)
                or 4D numpy array with image data (n_samples x 1 x height x width).
                It is also possible to pass 2D and 3D arrays by ignoring the channel dim,
                which will be added automatically. Note that if your X_train data are images,
                then your y_train must be spectra and vice versa.
            memory_alloc (int or float):
                Threshold (in GB) for holding all training data on GPU

        Returns:
        4-element tuple containing lists of numpy arrays or torch tensors
        with training and test data
        """
        data_all = preprocess_training_imspec_data_(
            X_train, y_train, X_test, y_test)
        dims = data_all[-1]

        X_train, y_train, X_test, y_test = array2list(
                *data_all[:-1], batch_size, memory_alloc)

        return X_train, y_train, X_test, y_test, dims


def init_dataloaders(X_train: torch.Tensor,
                     y_train: torch.Tensor,
                     X_test: torch.Tensor,
                     y_test: torch.Tensor,
                     batch_size: int,
                     memory_alloc: float = 4
                     ) -> Tuple[Type[torch.utils.data.DataLoader]]:
    """
    Returns two pytorch dataloaders for training and test data
    """
    device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_data = [X_train, y_train, X_test, y_test]
    arrsize = sum([get_array_memsize(x) for x in all_data])
    if arrsize / 1e9 > memory_alloc:
        device_ = 'cpu'
    X_train, y_train = X_train.to(device_), y_train.to(device_)
    X_test, y_test = X_test.to(device_), y_test.to(device_)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=batch_size, drop_last=True)
    return train_loader, test_loader


def init_fcnn_dataloaders(X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          batch_size: int,
                          num_classes: Optional[int] = None,
                          memory_alloc: float = 4
                          ) -> Tuple[Type[torch.utils.data.DataLoader], int]:
    """
    Returns two pytorch dataloaders for training and test data
    for semantic segmentation tasks, and the number of classes
    """
    data_all = preprocess_training_image_data_(
        X_train, y_train, X_test, y_test)
    num_classes = data_all[-1]
    train_loader, test_loader = init_dataloaders(
        *data_all[:-1], batch_size, memory_alloc)

    return train_loader, test_loader, num_classes


def init_imspec_dataloaders(X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            batch_size: int,
                            memory_alloc: float = 4
                            ) -> Tuple[Type[torch.utils.data.DataLoader], Tuple[Tuple[int]]]:
    """
    Returns train and test dataloaders for training images/spectra
    in a native PyTorch format and the (input, output) dimensions
    """

    data_all = preprocess_training_imspec_data_(
            X_train, y_train, X_test, y_test)
    dims = data_all[-1]
    train_loader, test_loader = init_dataloaders(
        *data_all[:-1], batch_size, memory_alloc)
    return train_loader, test_loader, dims


def init_vae_dataloaders(X_train: np.ndarray,
                         X_test: np.ndarray,
                         y_train: Optional[np.ndarray] = None,
                         y_test: Optional[np.ndarray] = None,
                         batch_size: int = 100,
                         ) -> Tuple[Type[torch.utils.data.DataLoader]]:
    """
    Returns train and test dataloaders for training images
    in a native PyTorch format
    """
    labels_ = y_train is not None and y_test is not None
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    if labels_:
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

    if torch.cuda.is_available():
        X_train = X_train.cuda()
        X_test = X_test.cuda()
    if labels_:
        y_train = y_train.cuda()
        y_test = y_test.cuda()

    if labels_:
        data_train = torch.utils.data.TensorDataset(X_train, y_train)
        data_test = torch.utils.data.TensorDataset(X_test, y_test)
    else:
        data_train = torch.utils.data.TensorDataset(X_train)
        data_test = torch.utils.data.TensorDataset(X_test)
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size)
    return train_loader, test_loader


def torch_format_image(image_data: np.ndarray,
                       norm: bool = True) -> torch.Tensor:
    """
    Reshapes (if needed), normalizes and converts image data
    to pytorch format for model training and prediction

    Args:
        image_data (3D or 4D numpy array):
            Image stack with dimensions (n_batches x height x width)
            or (n_batches x 1 x height x width)
        norm (bool):
            Normalize to (0, 1) (Default: True)
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
    if norm:
        image_data = (image_data - image_data.min()) / image_data.ptp()
    image_data = torch.from_numpy(image_data).float()
    return image_data


def torch_format_spectra(spectra: np.ndarray,
                         norm: bool = False) -> torch.Tensor:
    """
    Reshapes (if needed), normalizes and converts image data
    to pytorch format for model training and prediction

    Args:
        image_data (3D or 4D numpy array):
            Image stack with dimensions (n_batches x height x width)
            or (n_batches x 1 x height x width)
        norm (bool):
            Normalize data to (0, 1) (Default: False)
    """
    if spectra.ndim not in [2, 3]:
        raise AssertionError(
            "Provide spectrum(s) as 2D (n, length) or 3D (n, 1, length) tensor")
    if np.ndim(spectra) == 2:
        spectra = np.expand_dims(spectra, axis=1)
    elif np.ndim(spectra) == 3 and spectra.shape[1] != 1:
        raise AssertionError(
            "3D spectra tensor must have (n, 1, length) dimensions")
    else:
        pass
    if norm:
        spectra = (spectra - spectra.min()) / spectra.ptp()
    spectra = torch.from_numpy(spectra).float()
    return spectra


def torch_format(image_data: np.ndarray) -> torch.Tensor:
    """
    Reshapes (if needed), normalizes and converts image data
    to pytorch format for model training and prediction

    Args:
        image_data (3D or 4D numpy array):
            Image stack with dimensions (n_batches x height x width)
            or (n_batches x 1 x height x width)
    """
    warnings.warn("torch_format is deprecated. Use torch_format_image instead",
                  UserWarning)
    return torch_format_image(image_data)


def data_split(X_train: np.ndarray,
               y_train: np.ndarray,
               test_size: float = 0.15,
               random_state: int = 1,
               channel: Optional[str] = None,
               format_out: str = "numpy"
               ) -> Tuple[Union[np.ndarray, torch.Tensor]]:
    """
    Wrapper for sklearn's train_test_split, which also takes care
    (optionally) of pseudo-channel dimension and numpy-to-torch conversion
    """
    if channel == "first":
        X_train = X_train[:, np.newaxis]
        y_train = y_train[:, np.newaxis]
    elif channel == "last":
        X_train = X_train[..., np.newaxis]
        y_train = y_train[..., np.newaxis]
    elif channel is None:
        pass
    else:
        raise NotImplementedError(
            "{} channel format is not implemented".format(channel) +
            " Choose between 'first', 'last'")
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=test_size,
        shuffle=True, random_state=random_state)
    torf = lambda x: torch.from_numpy(x).float()
    torl = lambda x: torch.from_numpy(x).long()
    if format_out == "torch_float_long":
        X_train, X_test = torf(X_train), torf(X_test)
        y_train, y_test = torl(y_train), torl(y_test)
    elif format_out == "torch_float":
        X_train, X_test = torf(X_train), torf(X_test)
        y_train, y_test = torf(y_train), torf(y_test)
    elif format_out == "numpy":
        pass
    else:
        raise NotImplementedError(
            "{} output format is not implemented".format(format_out) +
            " Choose between 'torch_float', 'torch_float_long' and 'numpy'")

    return X_train, y_train, X_test, y_test
