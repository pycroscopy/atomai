import sys
import os
import numpy as np
from numpy.testing import assert_, assert_allclose

sys.path.append("../../../")

from atomai.predictors import Locator


test_nnout = np.load(os.path.join(
    os.path.dirname(__file__), 'test_data/test_nnoutput.npy'))
test_inimg = np.load(os.path.join(
    os.path.dirname(__file__), 'test_data/test_inputimg.npy'))
test_coord = np.load(os.path.join(
    os.path.dirname(__file__), 'test_data/test_coordinates.npy'),
    allow_pickle=True)[()]


def test_locator():
    loc = Locator()
    coord = loc.run(test_nnout)
    assert_(np.array_equal(test_coord[0], coord[0]))


def test_locator_edge():
    nnout = test_nnout[:, 0:256, 0:256]
    loc = Locator()
    coord1 = loc.run(nnout)
    loc = Locator(dist_edge=30)
    coord2 = loc.run(nnout)
    assert_(not np.equal(len(coord1[0]), len(coord2[0])))


def test_locator_dimorder():
    nnout = test_nnout.transpose(0, -1, 1, 2)
    loc = Locator(dim_order='channel_first')
    coord = loc.run(nnout)
    assert_(np.array_equal(test_coord[0], coord[0]))


def test_locator_refine():
    nnout = test_nnout[:, 0:256, 0:256]
    inimg = test_inimg[0:256, 0:256][None, ..., None]
    loc = Locator()
    coord1 = loc.run(nnout)
    loc = Locator(refine=True)
    coord2 = loc.run(nnout, inimg)
    assert(not np.array_equal(coord1[0], coord2[0]))
