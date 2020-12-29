import sys

import numpy as np
import torch
import pytest
from numpy.testing import assert_equal, assert_, assert_allclose

sys.path.append("../../../")

from atomai.trainers import SegTrainer, ImSpecTrainer


def gen_image_data():
    """
    Dummy images with random pixels
    """
    X = np.random.random(size=(5, 1, 8, 8))
    X_ = np.random.random(size=(5, 1, 8, 8))
    return X, X_


def gen_image_labels(binary=False):
    """
    Dummy labels for dummy images
    """
    if binary:
        y = np.random.randint(0, 2, size=(5, 1, 8, 8))
        y_ = np.random.randint(0, 2, size=(5, 1, 8, 8))
    else:
        y = np.random.randint(0, 3, size=(5, 8, 8))
        y_ = np.random.randint(0, 3, size=(5, 8, 8))
    return y, y_


def gen_spectra():
    """
    Dummy 1D signal with random points
    """
    X = np.random.random(size=(5, 1, 16))
    X_ = np.random.random(size=(5, 1, 16))
    return X, X_


@pytest.mark.parametrize(
    "loss_user, criterion_, binary",
    [("dice", "dice_loss()", True),
     ("focal", "focal_loss()", True),
     ("ce", "BCEWithLogitsLoss()", True),
     ("ce", "CrossEntropyLoss()", False)])
def test_trainer_loss_selection(loss_user, binary, criterion_):
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_image_labels(binary)
    nb_cls = 1 if binary else 3
    t = SegTrainer(nb_classes=nb_cls)
    t.compile_trainer(
        (X_train, y_train, X_test, y_test),
        training_cycles=1, batch_size=4,
        loss=loss_user)
    assert_equal(str(t.criterion), criterion_)


@pytest.mark.parametrize("model_type", ['Unet', 'dilnet'])
def test_segtrainer_determinism(model_type):
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_image_labels(binary=True)
    t1 = SegTrainer(model_type, upsampling="nearest", seed=1)
    t1.compile_trainer(
        (X_train, y_train, X_test, y_test),
        training_cycles=5, batch_size=4)
    _ = t1.run()
    loss1 = t1.loss_acc["train_loss"][-1]
    t2 = SegTrainer(model_type, upsampling="nearest", seed=1)
    t2.compile_trainer(
        (X_train, y_train, X_test, y_test),
        training_cycles=5, batch_size=4)
    _ = t2.run()
    loss2 = t2.loss_acc["train_loss"][-1]
    assert_allclose(loss1, loss2)
    for p1, p2 in zip(t1.net.parameters(), t2.net.parameters()):
        assert_allclose(p1.detach().cpu().numpy(), p2.detach().cpu().numpy())


@pytest.mark.parametrize("latent_dim", [2, 10])
def test_im2spec_trainer_determinism(latent_dim):
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_spectra()
    in_dim = (8, 8)
    out_dim = (16,)
    t1 = ImSpecTrainer(in_dim, out_dim, latent_dim, seed=1)
    t1.compile_trainer((X_train, y_train, X_test, y_test),
                       loss="mse", training_cycles=5, batch_size=4)
    _ = t1.run()
    loss1 = t1.loss_acc["train_loss"][-1]
    t2 = ImSpecTrainer(in_dim, out_dim, latent_dim, seed=1)
    t2.compile_trainer((X_train, y_train, X_test, y_test),
                       loss="mse", training_cycles=5, batch_size=4)
    _ = t2.run()
    loss2 = t2.loss_acc["train_loss"][-1]
    assert_allclose(loss1, loss2)
    for p1, p2 in zip(t1.net.parameters(), t2.net.parameters()):
        assert_allclose(p1.detach().cpu().numpy(), p2.detach().cpu().numpy())


@pytest.mark.parametrize("latent_dim", [2, 10])
def test_spec2im_trainer_determinism(latent_dim):
    X_train, X_test = gen_spectra()
    y_train, y_test = gen_image_data()
    in_dim = (16,)
    out_dim = (8, 8)
    t1 = ImSpecTrainer(
        in_dim, out_dim, latent_dim, seed=1)
    t1.compile_trainer(
        (X_train, y_train, X_test, y_test),
        loss="mse", training_cycles=5, batch_size=4)
    _ = t1.run()
    loss1 = t1.loss_acc["train_loss"][-1]
    t2 = ImSpecTrainer(
        in_dim, out_dim, latent_dim, seed=1)
    t2.compile_trainer(
        (X_train, y_train, X_test, y_test),
        loss="mse", training_cycles=5, batch_size=4)
    t2.run()
    loss2 = t2.loss_acc["train_loss"][-1]
    assert_allclose(loss1, loss2)
    for p1, p2 in zip(t1.net.parameters(), t2.net.parameters()):
        assert_allclose(p1.detach().cpu().numpy(), p2.detach().cpu().numpy())


@pytest.mark.parametrize(
    "binary, tensor_type",
    [(True, torch.float32), (False, torch.int64)])
def test_segtrainer_dataloader(binary, tensor_type):
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_image_labels(binary)
    nb_cls = 1 if binary else 3
    t = SegTrainer(nb_classes=nb_cls)
    t.compile_trainer(
        (X_train, y_train, X_test, y_test),
        training_cycles=1, batch_size=4)
    X_train_, y_train_ = t.dataloader(0)
    assert_equal(X_train_.dtype, torch.float32)
    assert_equal(y_train_.dtype, tensor_type)
    assert_equal(len(X_train), len(y_train))
    assert_equal(X_train_.is_cuda, torch.cuda.is_available())


def test_im2spectrainer_dataloader():
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_spectra()
    in_dim, out_dim = (8, 8), (16,)
    t = ImSpecTrainer(in_dim, out_dim, seed=1)
    t.compile_trainer((X_train, y_train, X_test, y_test),
                      loss="mse", training_cycles=5, batch_size=4)
    X_train_, y_train_ = t.dataloader(0)
    assert_equal(X_train_.dtype, torch.float32)
    assert_equal(y_train_.dtype, torch.float32)
    assert_equal(len(X_train), len(y_train))
    assert_equal(X_train_.is_cuda, torch.cuda.is_available())


def test_spec2imtrainer_dataloader():
    X_train, X_test = gen_spectra()
    y_train, y_test = gen_image_data()
    in_dim, out_dim = (16,), (8, 8)
    t = ImSpecTrainer(in_dim, out_dim, seed=1)
    t.compile_trainer((X_train, y_train, X_test, y_test),
                      loss="mse", training_cycles=5, batch_size=4)
    X_train_, y_train_ = t.dataloader(0)
    assert_equal(X_train_.dtype, torch.float32)
    assert_equal(y_train_.dtype, torch.float32)
    assert_equal(len(X_train), len(y_train))
    assert_equal(X_train_.is_cuda, torch.cuda.is_available())


@pytest.mark.parametrize(
    "bn, n_bn, layers",
    [(True, 16, [1, 2, 3, 4]),
     (True, 20, [2, 3, 3, 4]),
     (False, 0, [2, 3, 3, 4])])
def test_init_unet_bn(bn, n_bn, layers):
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_image_labels(binary=False)
    t = SegTrainer("Unet", nb_classes=3, batch_norm=bn, layers=layers)
    t.compile_trainer(
        (X_train, y_train, X_test, y_test),
        training_cycles=1, batch_size=4)
    n_bn_ = len([k for k in t.net.state_dict().keys() if 'running_mean' in k])
    assert_equal(n_bn_, n_bn)


@pytest.mark.parametrize(
    "dropouts, n_dropouts",
    [(False, 0), (True, 3)])
def test_init_unet_dropouts(dropouts, n_dropouts):
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_image_labels(binary=False)
    t = SegTrainer("Unet", nb_classes=3, dropout=dropouts)
    t.compile_trainer(
        (X_train, y_train, X_test, y_test),
        training_cycles=1, batch_size=4)
    n_dropouts_ = 0
    for c in t.net.children():
        if 'Dropout' in str([m for m in c.named_modules()]):
            n_dropouts_ += 1
    assert_equal(n_dropouts_, n_dropouts)


@pytest.mark.parametrize(
    "layers", [([1, 2, 3, 4]), ([2, 3, 3, 4])])
def test_init_unet_layers(layers):
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_image_labels(binary=False)
    t = SegTrainer("Unet", nb_classes=3, layers=layers)
    t.compile_trainer(
        (X_train, y_train, X_test, y_test),
        training_cycles=1, batch_size=4)
    n_bn_ = len([k for k in t.net.state_dict().keys() if 'running_mean' in k])
    layers_ = len([k for k in t.net.state_dict().keys() if "weight" in k])
    layers_ = layers_ - 4 - n_bn_
    assert_equal(layers_, 2 * sum(layers[:-1]) + layers[-1])


@pytest.mark.parametrize(
    "bn, n_bn, layers",
    [(True, 6, [1, 2, 2, 1]),
     (True, 10, [2, 3, 3, 2]),
     (False, 0, [3, 4, 4, 1])])
def test_init_dilnet_bn(bn, n_bn, layers):
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_image_labels(binary=False)
    t = SegTrainer(
        "dilnet", nb_classes=3, batch_norm=bn, layers=layers)
    t.compile_trainer(
        (X_train, y_train, X_test, y_test),
        training_cycles=1, batch_size=4)
    n_bn_ = len([k for k in t.net.state_dict().keys() if 'running_mean' in k])
    assert_equal(n_bn_, n_bn)


@pytest.mark.parametrize(
    "dropouts, n_dropouts",
    [(False, 0, ), (True, 2), (True, 2)])
def test_init_dilnet_dropouts(dropouts, n_dropouts):
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_image_labels(binary=False)
    t = SegTrainer(
        "dilnet", nb_classes=3, dropout=dropouts)
    t.compile_trainer(
        (X_train, y_train, X_test, y_test),
        training_cycles=1, batch_size=4)
    n_dropouts_ = 0
    for c in t.net.children():
        if 'Dropout' in str([m for m in c.named_modules()]):
            n_dropouts_ += 1
    assert_equal(n_dropouts_, n_dropouts)


@pytest.mark.parametrize(
    "layers", [([1, 2, 2, 1]), ([2, 3, 3, 2]), ([3, 4, 4, 1])])
def test_init_dilnet_layers(layers):
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_image_labels(binary=False)
    t = SegTrainer("dilnet", nb_classes=3, layers=layers)
    t.compile_trainer(
        (X_train, y_train, X_test, y_test),
        training_cycles=1, batch_size=4)
    n_bn_ = len([k for k in t.net.state_dict().keys() if 'running_mean' in k])
    layers_ = len([k for k in t.net.state_dict().keys() if "weight" in k])
    layers_ = layers_ - 2 - n_bn_
    assert_equal(layers_, sum(layers))


@pytest.mark.parametrize(
    "model, nb_filters_in, nb_filters_layers",
    [("Unet", 25, [25, 50, 100, 200, 100, 50, 25]),
     ("dilnet", 25, [25, 50, 50, 25])])
def test_init_segmodel_filters(model, nb_filters_in, nb_filters_layers):
    from atomai.nets.blocks import UpsampleBlock
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_image_labels(binary=False)
    t = SegTrainer(
        model, nb_classes=3,
        batch_norm=False, nb_filters=nb_filters_in)
    t.compile_trainer(
        (X_train, y_train, X_test, y_test),
        training_cycles=1, batch_size=4)
    filters_ = []
    for child in t.net.children():
        if isinstance(child, UpsampleBlock):
            continue
        filt_ = []
        for p in child.state_dict().values():
            filt_.append(p.shape[0])
        filters_.append(np.unique(filt_)[0])
    assert_equal(filters_[:-1], nb_filters_layers)


@pytest.mark.parametrize(
    "bn, encoder_layers, decoder_layers, n_bn",
    [(True, 2, 2, 5), (True, 3, 4, 8), (False, 3, 4, 0)])
def test_init_imspec_bn(bn, encoder_layers, decoder_layers, n_bn):
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_spectra()
    in_dim, out_dim = (8, 8), (16,)
    t = ImSpecTrainer(in_dim, out_dim, nblayers_encoder=encoder_layers,
                      nblayers_decoder=decoder_layers, batch_norm=bn)
    t.compile_trainer((X_train, y_train, X_test, y_test),
                      batch_size=4, loss="mse", training_cycles=1)
    n_bn_ = len([k for k in t.net.state_dict().keys() if 'running_mean' in k])
    assert_equal(n_bn_, n_bn)


@pytest.mark.parametrize(
    "downsample_factor, output_dim",
    [(2, (4, 4)), (4, (2, 2)), (0, (8, 8)), (1, (8, 8))])
def test_init_im2spec_edownsample(downsample_factor, output_dim):
    from atomai.utils import Hook, mock_forward
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_spectra()
    in_dim, out_dim = (8, 8), (16,)
    t = ImSpecTrainer(in_dim, out_dim, encoder_downsampling=downsample_factor)
    t.compile_trainer((X_train, y_train, X_test, y_test),
                      batch_size=4, loss="mse", training_cycles=1)
    hookF = [Hook(layer[1]) for layer in list(t.net.encoder._modules.items())]
    mock_forward(t.net, dims=(1, 8, 8))
    assert_equal(tuple([h.output.shape[-2:] for h in hookF][0]), output_dim)


@pytest.mark.parametrize(
    "downsample_factor, output_dim",
    [(2, 8), (4, 4), (0, 16), (1, 16)])
def test_init_spec2im_edownsample(downsample_factor, output_dim):
    from atomai.utils import Hook, mock_forward
    X_train, X_test = gen_spectra()
    y_train, y_test = gen_image_data()
    in_dim, out_dim = (16,), (8, 8)
    t = ImSpecTrainer(in_dim, out_dim, encoder_downsampling=downsample_factor)
    t.compile_trainer((X_train, y_train, X_test, y_test),
                      batch_size=4, loss="mse", training_cycles=1)
    hookF = [Hook(layer[1]) for layer in list(t.net.encoder._modules.items())]
    mock_forward(t.net, dims=(1, 16))
    assert_equal([h.output.shape[-1] for h in hookF][0], output_dim)


@pytest.mark.parametrize(
    "upsample, dim_", [(True, [4, 8, 16]), (False, [16, 16, 16])])
def test_init_im2spec_dupsample(upsample, dim_):
    from atomai.utils import Hook, mock_forward
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_spectra()
    in_dim, out_dim = (8, 8), (16,)
    t = ImSpecTrainer(in_dim, out_dim, decoder_upsampling=upsample)
    t.compile_trainer((X_train, y_train, X_test, y_test),
                      batch_size=4, loss="mse", training_cycles=1)
    hookF = [Hook(layer[1]) for layer in list(t.net.decoder._modules.items())]
    mock_forward(t.net, dims=(1, 8, 8))
    assert_equal([h.output.shape[-1] for h in hookF][1:4], dim_)


@pytest.mark.parametrize(
    "upsample, dim_", [(True, [2, 4, 8]), (False, [8, 8, 8])])
def test_init_spec2im_dupsample(upsample, dim_):
    from atomai.utils import Hook, mock_forward
    X_train, X_test = gen_spectra()
    y_train, y_test = gen_image_data()
    in_dim, out_dim = (16,), (8, 8)
    t = ImSpecTrainer(in_dim, out_dim, decoder_upsampling=upsample)
    t.compile_trainer((X_train, y_train, X_test, y_test),
                      batch_size=4, loss="mse", training_cycles=1)
    hookF = [Hook(layer[1]) for layer in list(t.net.decoder._modules.items())]
    mock_forward(t.net, dims=(1, 16))
    assert_equal([h.output.shape[-1] for h in hookF][1:4], dim_)


@pytest.mark.parametrize(
    "efilt, dfilt, filters_e_sorted, filters_d_sorted",
    [(25, 50, [25], [1, 50]), (32, 16, [32], [1, 16])])
def test_im2spec_filters(efilt, dfilt, filters_e_sorted, filters_d_sorted):
    from atomai.nets import SignalDecoder, SignalEncoder
    X_train, X_test = gen_image_data()
    y_train, y_test = gen_spectra()
    in_dim, out_dim = (8, 8), (16,)
    t = ImSpecTrainer(
        in_dim, out_dim, nbfilters_encoder=efilt, nbfilters_decoder=dfilt)
    t.compile_trainer((X_train, y_train, X_test, y_test),
                      batch_size=4, loss="mse", training_cycles=1)
    filters_e, filters_d = [], []
    for child in t.net.children():
        if isinstance(child, SignalEncoder):
            filt_ = []
            for p in child.state_dict().values():
                if len(p.shape) == 4:
                    filt_.append(p.shape[0])
            filters_e.extend(filt_)
        elif isinstance(child, SignalDecoder):
            filt_ = []
            for p in child.state_dict().values():
                if len(p.shape) == 3:
                    filt_.append(p.shape[0])
            filters_d.extend(filt_)
    assert_equal(sorted(np.unique(filters_e).tolist()), filters_e_sorted)
    assert_equal(sorted(np.unique(filters_d).tolist()), filters_d_sorted)


@pytest.mark.parametrize(
    "efilt, dfilt, filters_e_sorted, filters_d_sorted",
    [(25, 50, [25], [1, 50]), (32, 16, [32], [1, 16])])
def test_spec2im_filters(efilt, dfilt, filters_e_sorted, filters_d_sorted):
    from atomai.nets import SignalDecoder, SignalEncoder
    X_train, X_test = gen_spectra()
    y_train, y_test = gen_image_data()
    in_dim, out_dim = (16,), (8, 8)
    t = ImSpecTrainer(
        in_dim, out_dim, nbfilters_encoder=efilt, nbfilters_decoder=dfilt)
    t.compile_trainer((X_train, y_train, X_test, y_test),
                      batch_size=4, loss="mse", training_cycles=1)
    filters_e, filters_d = [], []
    for child in t.net.children():
        if isinstance(child, SignalEncoder):
            filt_ = []
            for p in child.state_dict().values():
                if len(p.shape) == 3:
                    filt_.append(p.shape[0])
            filters_e.extend(filt_)
        elif isinstance(child, SignalDecoder):
            filt_ = []
            for p in child.state_dict().values():
                if len(p.shape) == 4:
                    filt_.append(p.shape[0])
            filters_d.extend(filt_)
    assert_equal(sorted(np.unique(filters_e).tolist()), filters_e_sorted)
    assert_equal(sorted(np.unique(filters_d).tolist()), filters_d_sorted)