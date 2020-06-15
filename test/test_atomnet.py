import os
import numpy as np
import torch
from atomai.core import atomnet, models
from atomai.utils import load_weights
import pytest
from numpy.testing import assert_allclose, assert_equal
import matplotlib
matplotlib.use('Agg')

test_img = os.path.join(
    os.path.dirname(__file__), 'test_data/test_img.npy')
test_coord_s = os.path.join(
    os.path.dirname(__file__), 'test_data/test_coord_s.npy')
test_coord_m = os.path.join(
    os.path.dirname(__file__), 'test_data/test_coord_m.npy')
test_model_s = os.path.join(
    os.path.dirname(__file__), 'test_data/model_s_weights_final.pt')
test_model_m = os.path.join(
    os.path.dirname(__file__), 'test_data/model_m_weights_final.pt')


def gen_dummy_data(n_atoms):
    """
    Generate dummy train and test variables
    """
    X_train = np.random.random(size=(25, 32, 32))
    X_test = np.random.random(size=(10, 32, 32))
    y_train = np.random.randint(0, n_atoms+1, size=(25, 32, 32))
    y_test = np.random.randint(0, n_atoms+1, size=(10, 32, 32))
    return X_train, y_train, X_test, y_test


@pytest.mark.parametrize(
    "img_dims_in, lbl_dims_in, n_atoms, img_dims_out, lbl_dims_out",
    [(3, 3, 1, 4, 4), (3, 3, 2, 4, 3), (4, 3, 1, 4, 4), (4, 3, 2, 4, 3)])
def test_trainer_input_dims(img_dims_in, lbl_dims_in, n_atoms,
                            img_dims_out, lbl_dims_out):

    def gen_data(dims):
        """
        Dummy variables
        """
        if dims == 3:
            X = np.random.random(size=(25, 32, 32))
            X_ = np.random.random(size=(10, 32, 32))
        elif dims == 4:
            X = np.random.random(size=(25, 1, 32, 32))
            X_ = np.random.random(size=(10, 1, 32, 32))
        return X, X_

    def gen_labels(dims, n_atoms):
        """
        Dummy variables
        """
        if dims == 3:
            y = np.random.randint(0, n_atoms+1, size=(25, 32, 32))
            y_ = np.random.randint(0, n_atoms+1, size=(10, 32, 32))
        elif dims == 4:
            y = np.random.randint(0, n_atoms+1, size=(25, 1, 32, 32))
            y_ = np.random.randint(0, n_atoms+1, size=(10, 1, 32, 32))
        return y, y_

    X_train, X_test = gen_data(img_dims_in)
    y_train, y_test = gen_labels(lbl_dims_in, n_atoms)
    m = atomnet.trainer(
        X_train, y_train, X_test, y_test, training_cycles=1, batch_size=4)
    assert_equal(m.images_all[0].ndim, img_dims_out)
    assert_equal(m.images_test_all[0].ndim, img_dims_out)
    assert_equal(m.labels_all[0].ndim, lbl_dims_out)
    assert_equal(m.labels_test_all[0].ndim, lbl_dims_out)


@pytest.mark.parametrize(
    "loss_user, criterion_, n_atoms",
     [("dice", "dice_loss()", 1),
     ("focal", "focal_loss()", 1),
     ("ce", "BCEWithLogitsLoss()", 1),
     ("ce", "CrossEntropyLoss()", 2)])
def test_trainer_loss_selection(loss_user, n_atoms, criterion_):
    X_train, y_train, X_test, y_test = gen_dummy_data(n_atoms)
    m = atomnet.trainer(
        X_train, y_train, X_test, y_test,
        training_cycles=1, batch_size=4, loss=loss_user)
    assert_equal(str(m.criterion), criterion_)


@pytest.mark.parametrize(
    "n_atoms, tensor_type",
    [(1, torch.float32), (2, torch.int64)])
def test_trainer_dataloader(n_atoms, tensor_type):
    X_train, y_train, X_test, y_test = gen_dummy_data(n_atoms)
    m = atomnet.trainer(
        X_train, y_train, X_test, y_test, training_cycles=1, batch_size=4)
    X_train_, y_train_ = m.dataloader(0)
    assert_equal(y_train_.dtype, tensor_type)
    assert_equal(len(X_train), len(y_train))
    assert_equal(X_train_.is_cuda, torch.cuda.is_available())

@pytest.mark.parametrize("model_type", ['dilUnet', 'dilnet'])
def test_trainer_determinism(model_type):
    X_train, y_train, X_test, y_test = gen_dummy_data(1)
    m1 = atomnet.trainer(
        X_train, y_train, X_test, y_test, model_type=model_type,
        training_cycles=5, batch_size=4, upsampling="nearest")
    m1.run()
    loss1 = m1.train_loss[-1]
    m2 = atomnet.trainer(
        X_train, y_train, X_test, y_test, model_type=model_type,
        training_cycles=5, batch_size=4, upsampling="nearest")
    m2.run()
    loss2 = m2.train_loss[-1]
    assert_allclose(loss1, loss2)
    for p1, p2 in zip(m1.net.parameters(), m2.net.parameters()):
        assert_allclose(p1.detach().cpu().numpy(), p2.detach().cpu().numpy())


@pytest.mark.parametrize(
    "weights_, nb_classes, coord_expected",
    [(test_model_s, 1, test_coord_s),
    (test_model_m, 3, test_coord_m)])
def test_atomfind(weights_, nb_classes, coord_expected):
    test_img_ = np.load(test_img)
    coordinates_expected = np.load(coord_expected)
    model_ = load_weights(models.dilUnet(nb_classes), weights_)
    _, (nn_output, coordinates_) = atomnet.predictor(test_img_, model_).run()
    assert_allclose(coordinates_[0], coordinates_expected)