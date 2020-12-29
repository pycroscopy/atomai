import sys

import numpy as np
import torch
import pytest
from numpy.testing import assert_equal, assert_

sys.path.append("../../../")

from atomai.utils.preproc import (array2list, check_image_dims,
                                  check_signal_dims, get_array_memsize,
                                  init_fcnn_dataloaders,
                                  init_imspec_dataloaders,
                                  num_classes_from_labels,
                                  preprocess_training_image_data)


def gen_data(dims):
    """
    Dummy variables
    """
    if dims == 2:
        X = np.random.random(size=(5, 8))
        X_ = np.random.random(size=(5, 8))
    if dims == 3:
        X = np.random.random(size=(5, 8, 8))
        X_ = np.random.random(size=(5, 8, 8))
    elif dims == 4:
        X = np.random.random(size=(5, 1, 8, 8))
        X_ = np.random.random(size=(5, 1, 8, 8))
    return X, X_


def gen_labels(dims, binary=False):
    if dims == 2:
        y = np.random.random(size=(5, 8))
        y_ = np.random.random(size=(5, 8))
    if dims == 3:
        if binary:
            y = np.random.randint(0, 2, size=(5, 8, 8))
            y_ = np.random.randint(0, 2, size=(5, 8, 8))
        else:
            y = np.random.randint(0, 3, size=(5, 8, 8))
            y_ = np.random.randint(0, 3, size=(5, 8, 8))
    if dims == 4:
        if binary:
            y = np.random.randint(0, 2, size=(5, 1, 8, 8))
            y_ = np.random.randint(0, 2, size=(5, 1, 8, 8))
        else:
            y = np.random.randint(0, 3, size=(5, 1, 8, 8))
            y_ = np.random.randint(0, 3, size=(5, 1, 8, 8))
    return y, y_


@pytest.mark.parametrize(
    "dims, binary, num_classes",
    [(3, True, 1), (4, False, 3), (3, False, 3), (4, True, 1)])
def test_num_classes_from_labels(dims, binary, num_classes):
    y, _ = gen_labels(dims, binary)
    print(binary, num_classes, num_classes_from_labels(y))
    assert_equal(num_classes, num_classes_from_labels(y))


@pytest.mark.parametrize(
    "img_dims_in, lbl_dims_in, n_classes, img_dims_out, lbl_dims_out",
    [(3, 3, 1, 4, 4), (3, 3, 2, 4, 3), (4, 3, 1, 4, 4), (4, 3, 2, 4, 3)])
def test_trainer_input_dims(img_dims_in, lbl_dims_in, n_classes,
                            img_dims_out, lbl_dims_out):
    X, X_ = gen_data(img_dims_in)
    y, y_ = gen_labels(lbl_dims_in)
    X, y, X_, y_ = check_image_dims(X, y, X_, y_, n_classes)
    assert_equal(X.ndim, img_dims_out)
    assert_equal(X_.ndim, img_dims_out)
    assert_equal(y.ndim, lbl_dims_out)
    assert_equal(y_.ndim, lbl_dims_out)


@pytest.mark.parametrize(
    "img_dims_in, lbl_dims_in, img_dims_out, lbl_dims_out",
    [(3, 2, 4, 3), (4, 3, 4, 3), (2, 3, 3, 4), (3, 4, 3, 4)])
def test_check_signal_dims(img_dims_in, lbl_dims_in,
                           img_dims_out, lbl_dims_out):
    X, X_ = gen_data(img_dims_in)
    y, y_ = gen_labels(lbl_dims_in)
    X, y, X_, y_ = check_signal_dims(X, y, X_, y_)
    assert_equal(X.ndim, img_dims_out)
    assert_equal(X_.ndim, img_dims_out)
    assert_equal(y.ndim, lbl_dims_out)
    assert_equal(y_.ndim, lbl_dims_out)


def test_array2list():
    batch_size = 2
    X, X_ = gen_data(dims=4)
    y, y_ = gen_labels(dims=3)
    X, y, X_, y_ = array2list(X, y, X_, y_, batch_size)
    assert_(isinstance(X, list))
    assert_(isinstance(y, list))
    assert_(isinstance(X_, list))
    assert_(isinstance(y_, list))
    assert_(isinstance(X[0], np.ndarray))
    assert_(isinstance(y[0], np.ndarray))
    assert_(isinstance(X_[0], np.ndarray))
    assert_(isinstance(y_[0], np.ndarray))
    assert_(len(X) == len(X_) == len(y) == len(y_) == 2)
    assert_(X[0].shape[0] == y[0].shape[0] == X_[0].shape[0] == y_[0].shape[0])
    assert_equal(X[0].shape[0], batch_size)


def test_preprocess_training_image_data():
    X, _ = gen_data(4)
    X_ = [1, 2, 3]
    y, y_ = gen_labels(3)
    with pytest.raises(TypeError):
        _ = preprocess_training_image_data(X, X_, y, y_)


def test_init_fcnn_dataloaders():
    X, X_ = gen_data(dims=4)
    y, y_ = gen_labels(dims=3)
    train_loader, test_loader, _ = init_fcnn_dataloaders(X, y, X_, y_, 2)
    assert_(isinstance(train_loader, torch.utils.data.dataloader.DataLoader))
    assert_(isinstance(test_loader, torch.utils.data.dataloader.DataLoader))


def test_init_imspec_dataloaders():
    X, X_ = gen_data(dims=3)
    y, y_ = gen_labels(dims=2)
    train_loader, test_loader, _ = init_imspec_dataloaders(X, y, X_, y_, 2)
    assert_(isinstance(train_loader, torch.utils.data.dataloader.DataLoader))
    assert_(isinstance(test_loader, torch.utils.data.dataloader.DataLoader))


@pytest.mark.parametrize(
    "precision, arr_dtype, arrsize",
    [("single", torch.float32, 0.32768), ("single", torch.float64, 0.32768),
     ("double", torch.float32, 0.65536), ("double", torch.float64, 0.65536)])
def test_tensor_memsize(precision, arr_dtype, arrsize):
    arr = torch.randn(20000, 64, 64, 1, dtype=arr_dtype)
    arrsize_ = get_array_memsize(arr, precision) / 1e9
    assert_equal(arrsize_, arrsize)


@pytest.mark.parametrize(
    "precision, arr_dtype, arrsize",
    [("single", np.float32, 0.32768), ("single", np.float64, 0.32768),
     ("double", np.float32, 0.65536), ("double", np.float64, 0.65536)])
def test_array_memsize(precision, arr_dtype, arrsize):
    arr = np.random.randn(20000, 64, 64, 1).astype(arr_dtype)
    arrsize_ = get_array_memsize(arr, precision) / 1e9
    assert_equal(arrsize_, arrsize)


def test_noarray_memsize():
    arr = None
    arrsize_ = get_array_memsize(arr)
    assert_equal(arrsize_, 0)
