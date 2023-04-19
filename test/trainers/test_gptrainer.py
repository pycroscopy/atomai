import sys
from copy import deepcopy as dc

import numpy as np
import pytest
import torch
from numpy.testing import assert_, assert_equal

sys.path.append("../../../")

from atomai.trainers import dklGPTrainer, GPTrainer


def weights_equal(m1, m2):
    eq_w = []
    for p1, p2 in zip(m1.values(), m2.values()):
        eq_w.append(np.allclose(
            p1.detach().cpu().numpy(),
            p2.detach().cpu().numpy()))
    return all(eq_w)




def test_gptrainer_compiler():
    X = np.random.randn(10, 2)
    y = np.random.randn(10)
    t = GPTrainer(precision="single")
    t.compile_trainer(X, y, 1)
    assert_(t.gp_model is not None)
    assert_(t.likelihood is not None)


def test_gptrainer_train():
    h = w = 10
    X = np.array([(i, j) for i in range(h) for j in range(w)])
    y = np.random.randn(len(X))
    t = GPTrainer()
    t.compile_trainer(X, y)
    params_init = dc(t.gp_model.base_covar_module.base_kernel.lengthscale.detach().cpu().numpy())
    t.train_step()
    params_final = t.gp_model.base_covar_module.base_kernel.lengthscale.detach().cpu().numpy()
    assert_(not np.array_equal(params_init, params_final))


def test_gptrainer_compile_and_run():
    h = w = 10
    X = np.array([(i, j) for i in range(h) for j in range(w)])
    y = np.random.randn(len(X))
    t = GPTrainer()
    t.compile_trainer(X, y, training_cycles=3)
    _ = t.run()
    assert_equal(len(t.train_loss), 3)


def test_gptrainer_run():
    h = w = 10
    X = np.array([(i, j) for i in range(h) for j in range(w)])
    y = np.random.randn(len(X))
    t = GPTrainer()
    _ = t.run(X, y, 3)
    assert_equal(len(t.train_loss), 3)


@pytest.mark.parametrize(
    "precision, dtype",
    [("single", torch.float32), ("double", torch.float64)])
def test_dkltrainer_precision(precision, dtype):
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPTrainer(indim, precision=precision)
    X_, y_ = t.set_data(X, y)
    assert_equal(X_.dtype, dtype)
    assert_equal(y_.dtype, dtype)


def test_dkltrainer_compiler():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPTrainer(indim, precision="single")
    t.compile_trainer(X, y, 2)
    assert_(t.gp_model is not None)
    assert_(t.likelihood is not None)


def test_multi_model_dkltrainer_compiler():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(2, 50)
    t = dklGPTrainer(indim, shared_embedding_space=False, precision="single")
    t.compile_multi_model_trainer(X, y, 2)
    assert_(t.gp_model is not None)
    assert_(t.likelihood is not None)


def test_dkltrainer_train():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPTrainer(indim, precision="single")
    t.compile_trainer(X, y)
    w_init = dc(t.gp_model.feature_extractor.state_dict())
    t.train_step()
    w_final = t.gp_model.feature_extractor.state_dict()
    assert_(not weights_equal(w_init, w_final))


def test_multi_model_dkltrainer_train():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(2, 50)
    t = dklGPTrainer(indim, shared_embedding_space=False, precision="single")
    t.compile_multi_model_trainer(X, y)
    w_init1 = dc(t.gp_model.models[0].feature_extractor.state_dict())
    w_init2 = dc(t.gp_model.models[1].feature_extractor.state_dict())
    t.train_step()
    w_final1 = t.gp_model.models[0].feature_extractor.state_dict()
    w_final2 = t.gp_model.models[1].feature_extractor.state_dict()
    assert_(not weights_equal(w_final1, w_final2))
    assert_(not weights_equal(w_init1, w_final1))
    assert_(not weights_equal(w_init2, w_final2))


def test_dkltrainer_train_freeze_w():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPTrainer(indim, precision="single")
    t.compile_trainer(X, y, freeze_weights=True)
    w_init = dc(t.gp_model.feature_extractor.state_dict())
    t.train_step()
    w_final = t.gp_model.feature_extractor.state_dict()
    assert_(weights_equal(w_init, w_final))


def test_dkltrainer_compile_and_run():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPTrainer(indim, precision="single")
    t.compile_trainer(X, y, training_cycles=3)
    _ = t.run()
    assert_equal(len(t.train_loss), 3)


def test_ensemble_dkltrainer_compile_and_run():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(1, 50).repeat(3, axis=0)
    t = dklGPTrainer(indim, precision="single", shared_embedding_space=False)
    t.ensemble = True
    t.compile_multi_model_trainer(X, y)
    w1 = t.gp_model.models[0].state_dict()
    w2 = t.gp_model.models[2].state_dict()
    assert_(not weights_equal(w1, w2))


def test_dkltrainer_run():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPTrainer(indim, precision="single")
    _ = t.run(X, y, 3)
    assert_equal(len(t.train_loss), 3)


def test_dkltrainer_save_weights():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPTrainer(indim, precision="single")
    _ = t.run(X, y, 1)
    w1 = dc(t.gp_model.feature_extractor.state_dict())
    t.save_weights("m.pt")
    loaded_weights = torch.load("m.pt")
    t.gp_model.feature_extractor.load_state_dict(loaded_weights)
    w2 = t.gp_model.feature_extractor.state_dict()
    assert_(weights_equal(w1, w2))

