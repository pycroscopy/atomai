import sys

import numpy as np
import pytest
import torch
from numpy.testing import assert_, assert_equal

sys.path.append("../../../")

from atomai.utils.img import (cv_resize, cv_resize_stack, extract_patches,
                              extract_random_subimages, img_pad)


def test_cv_resize():
    rs = (16, 16)
    img = np.random.randn(32, 32)
    img_r = cv_resize(img, rs)
    assert_equal(img_r.shape, rs)


def test_cv_resize_stack():
    rs = (16, 16)
    img = np.random.randn(3, 32, 32)
    img_r = cv_resize_stack(img, rs)
    assert_equal(len(img_r), len(img))
    assert_equal(img_r.shape[1:], rs)


def test_extract_patches_from_stack():
    patch_size = 32
    num_patches = 64
    images = np.random.randn(2, 128, 128)
    masks = np.random.randint(0, 3, size=(2, 128, 128))
    patches = extract_patches(images, masks, patch_size, num_patches)
    assert_equal(patches[0].shape, patches[1].shape)
    assert_equal(patches[0].shape, (num_patches * 2, patch_size, patch_size))


def test_extract_patches_from_single_im():
    patch_size = 32
    num_patches = 64
    images = np.random.randn(128, 128)
    masks = np.random.randint(0, 3, size=(128, 128))
    patches = extract_patches(images, masks, patch_size, num_patches)
    assert_equal(patches[0].shape, patches[1].shape)
    assert_equal(patches[0].shape, (num_patches, patch_size, patch_size))


@pytest.mark.parametrize("shape", [(31, 32), (32, 32), (32, 31), (30, 29)])
def test_img_pad(shape):
    images = np.random.randn(3, *shape)
    images_padded = img_pad(images, 8)
    assert_equal(images_padded.shape, (3, 32, 32))


def test_extract_random_subimages_px():
    window_size = 32
    num_images = 20
    images = np.random.randn(3, 128, 128)
    patches = extract_random_subimages(images, window_size, num_images)
    assert_equal(
        patches[0].squeeze().shape, (num_images * 3, window_size, window_size))


def test_extract_random_subimages_coord():
    window_size = 32
    num_images = 20
    coord = np.random.randint(10, 100, (num_images * 2, 2))
    classes = np.zeros(num_images * 2)
    coord = np.concatenate((coord, classes[:, None]), -1)
    coord = {i: coord for i in range(3)}
    images = np.random.randn(3, 128, 128)
    patches = extract_random_subimages(
        images, window_size, num_images, coordinates=coord)
    assert_equal(
        patches[0].squeeze().shape, (num_images * 3, window_size, window_size))
