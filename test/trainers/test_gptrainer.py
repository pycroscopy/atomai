import sys
from copy import deepcopy as dc

import numpy as np
import pytest
import torch
from numpy.testing import assert_, assert_equal

sys.path.append("../../../")

from atomai.trainers import dklGPTrainer


def weights_equal(m1, m2):
    eq_w = []
    for p1, p2 in zip(m1.values(), m2.values()):
        eq_w.append(np.allclose(
            p1.detach().cpu().numpy(),
            p2.detach().cpu().numpy()))
    return all(eq_w)


@pytest.mark.parametrize(
    "precision, dtype",
    [("single", torch.float32), ("double", torch.float64)])
def test_trainer_precision(precision, dtype):
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPTrainer(indim, precision=precision)
    X_, y_ = t.set_data(X, y)
    assert_equal(X_.dtype, dtype)
    assert_equal(y_.dtype, dtype)


def test_trainer_compiler():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPTrainer(indim, precision="single")
    t.compile_trainer(X, y, 2)
    assert_(t.gp_model is not None)
    assert_(t.likelihood is not None)


def test_multi_model_trainer_compiler():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(2, 50)
    t = dklGPTrainer(indim, shared_embedding_space=False, precision="single")
    t.compile_multi_model_trainer(X, y, 2)
    assert_(t.gp_model is not None)
    assert_(t.likelihood is not None)


def test_trainer_train():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPTrainer(indim, precision="single")
    t.compile_trainer(X, y)
    w_init = dc(t.gp_model.feature_extractor.state_dict())
    t.train_step()
    w_final = t.gp_model.feature_extractor.state_dict()
    assert_(not weights_equal(w_init, w_final))


def test_multi_model_trainer_train():
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


def test_trainer_train_freeze_w():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPTrainer(indim, precision="single")
    t.compile_trainer(X, y, freeze_weights=True)
    w_init = dc(t.gp_model.feature_extractor.state_dict())
    t.train_step()
    w_final = t.gp_model.feature_extractor.state_dict()
    assert_(weights_equal(w_init, w_final))


def test_trainer_compile_and_run():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPTrainer(indim, precision="single")
    t.compile_trainer(X, y, training_cycles=3)
    _ = t.run()
    assert_equal(len(t.train_loss), 3)


def test_ensemble_trainer_compile_and_run():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(1, 50).repeat(3, axis=0)
    t = dklGPTrainer(indim, precision="single", shared_embedding_space=False)
    t.ensemble = True
    t.compile_multi_model_trainer(X, y)
    w1 = t.gp_model.models[0].state_dict()
    w2 = t.gp_model.models[2].state_dict()
    assert_(not weights_equal(w1, w2))


def test_trainer_run():
    indim = 32
    X = np.random.randn(50, indim)
    y = np.random.randn(50)
    t = dklGPTrainer(indim, precision="single")
    _ = t.run(X, y, 3)
    assert_equal(len(t.train_loss), 3)


def test_trainer_save_weights():
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
