"""
trainer.py
==========

Module for training fully convolutional neural networs
for atom/defect/particle finding and encoder-decoder neural networks
for prediction of spectra/images from images/spectra. It can also be
used for training custom PyTorch neural networks

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""


import copy
import warnings
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from atomai import losses_metrics
from atomai.nets import init_fcnn_model, init_imspec_model
from atomai.utils import (array2list, average_weights, gpu_usage_map,
                          init_dataloaders, init_fcnn_dataloaders,
                          init_imspec_dataloaders, plot_losses, reset_bnorm,
                          preprocess_training_image_data, weights_init,
                          preprocess_training_imspec_data, set_train_rng)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

warnings.filterwarnings("ignore", module="torch.nn.functional")

augfn_type = Callable[[torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]


class BaseTrainer:
    """
    Base trainer class for training semantic segmentation
    and image-to-spectrum/spectrum-to-image deep learning models
    as well as custom PyTorch neural networks

    Example:

    >>> # Load 4 numpy arrays with training and test data
    >>> dataset = np.load('training_data.npz')
    >>> images = dataset['X_train']
    >>> labels = dataset['y_train']
    >>> images_test = dataset['X_test']
    >>> labels_test = dataset['y_test']
    >>> # Initialize a trainer
    >>> t = BaseTrainer()
    >>> # Set a model
    >>> t.set_model(atomai.nets.Unet(), nb_classes=1)
    >>> # Compile trainer
    >>> t.compile_trainer(
    >>>     (images, labels, images_test_1, labels_test_1),
    >>>     loss="ce", full_epoch=True, training_cycles=25, swa=True)
    >>> # Train and save model's weights
    >>> t.fit()
    >>> t.save_model("my_model")
    """
    def __init__(self):
        set_train_rng(1)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = None
        self.criterion = None
        self.optimizer = None
        self.compute_accuracy = False
        self.full_epoch = True
        self.swa = False
        self.perturb_weights = False
        self.running_weights = {}
        self.training_cycles = 0
        self.batch_idx_train, self.batch_idx_test = [], []
        self.batch_size = 1
        self.nb_classes = None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.train_loader = torch.utils.data.TensorDataset()
        self.test_loader = torch.utils.data.TensorDataset()
        self.data_is_set = False
        self.augdict = {}
        self.augment_fn = None
        self.filename = "model"
        self.print_loss = 1
        self.meta_state_dict = dict()
        self.loss_acc = {"train_loss": [], "test_loss": [],
                         "train_accuracy": [], "test_accuracy": []}

    def _reset_rng(self, seed: int) -> None:
        """
        (re)sets seeds for pytorch and numpy random number generators
        """
        set_train_rng(seed)

    def _reset_weights(self) -> None:
        """
        Resets weights of convolutional and linear NN layers
        using Xavier initialization
        """
        self.net.apply(weights_init)
        self.net.apply(reset_bnorm)

    def _reset_training_history(self) -> None:
        """
        Empties training/test losses and accuracies
        (can be useful for ensemble training)
        """
        self.loss_acc = {"train_loss": [], "test_loss": [],
                         "train_accuracy": [], "test_accuracy": []}

    def set_data(self,
                 X_train: Union[torch.Tensor, np.ndarray],
                 y_train: Union[torch.Tensor, np.ndarray],
                 X_test: Union[torch.Tensor, np.ndarray],
                 y_test: Union[torch.Tensor, np.ndarray],
                 **kwargs: float) -> None:
        """
        Sets training and test data by initializing PyTorch dataloaders
        or creating a list of PyTorch tensors from which it will randomly
        choose an element at each training iteration.

        Args:
            X_train: Training data
            y_train: Training data labels/ground-truth
            X_test: Test data
            y_test: Test data labels/ground-truth
            memory_alloc: threshold (in GB) for holding all training data on GPU
        """
        memory_alloc = kwargs.get("memory_alloc", 4)
        tor = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        X_train, y_train = tor(X_train), tor(y_train)
        X_test, y_test = tor(X_test), tor(y_test)

        if self.full_epoch:
            self.train_loader, self.test_loader = init_dataloaders(
                X_train, y_train, X_test, y_test,
                self.batch_size, memory_alloc)
        else:
            (self.X_train, self.y_train,
             self.X_test, self.y_test) = array2list(
                X_train, y_train, X_test, y_test,
                self.batch_size, memory_alloc)

        self.data_is_set = True

    def set_model(self,
                  model: Type[torch.nn.Module],
                  nb_classes: int = None) -> None:
        """
        Sets a neural network model and a number of classes (if any)

        Args:
            model: initialized PyTorch model
            nb_classes: number of classes in classification scheme (if any)
        """
        self.net = model
        self.net.to(self.device)
        if self.nb_classes is None and nb_classes:
            self.nb_classes = nb_classes

    def get_loss_fn(self,
                    loss: Union[str, Callable] = 'mse',
                    nb_classes: int = None) -> None:
        """
        Returns a loss function. Available loss functions are: 'mse' (MSE),
        'ce' (cross-entropy), 'focal' (focal loss; single class only),
        and 'dice' (dice loss; for semantic segmentation problems)
        """
        return losses_metrics.select_loss(loss, nb_classes)

    def train_step(self,
                   feat: torch.Tensor,
                   tar: torch.Tensor) -> Tuple[float]:
        """
        Propagates image(s) through a network to get model's prediction
        and compares predicted value with ground truth; then performs
        backpropagation to compute gradients and optimizes weights.

        Args:
            feat: input features
            tar: targets
        """
        self.net.train()
        self.optimizer.zero_grad()
        feat, tar = feat.to(self.device), tar.to(self.device)
        prob = self.net(feat)
        loss = self.criterion(prob, tar)
        loss.backward()
        self.optimizer.step()
        if self.compute_accuracy:
            acc_score = self.accuracy_fn(tar, prob)
            return (loss.item(), acc_score)
        return (loss.item(),)

    def test_step(self,
                  feat: torch.Tensor,
                  tar: torch.Tensor) -> float:
        """
        Forward pass for test data with deactivated autograd engine

        Args:
            feat: input features
            tar: targets
        """
        feat, tar = feat.to(self.device), tar.to(self.device)
        self.net.eval()
        with torch.no_grad():
            prob = self.net(feat)
            loss = self.criterion(prob, tar)
        if self.compute_accuracy:
            acc_score = self.accuracy_fn(tar, prob)
            return (loss.item(), acc_score)
        return (loss.item(),)

    def step(self, e: int) -> None:
        """
        Single train-test step which passes a single
        mini-batch (for both training and testing), i.e.
        1 "epoch" = 1 mini-batch
        """
        features, targets = self.dataloader(
            self.batch_idx_train[e], mode='train')
        # Training step
        loss = self.train_step(features, targets)
        self.loss_acc["train_loss"].append(loss[0])
        features_, targets_ = self.dataloader(
            self.batch_idx_test[e], mode='test')
        # Test step
        loss_ = self.test_step(features_, targets_)
        self.loss_acc["test_loss"].append(loss_[0])
        if self.compute_accuracy:
            self.loss_acc["train_accuracy"].append(loss[1])
            self.loss_acc["test_accuracy"].append(loss_[1])

    def step_full(self) -> None:
        """
        A standard PyTorch training loop where
        all available mini-batches are passed at
        a single step/epoch
        """
        c, c_test = 0, 0
        losses, losses_test = 0, 0
        if self.compute_accuracy:
            acc, acc_test = 0, 0
        # Training step
        for features, targets in self.train_loader:
            if self.augment_fn is not None:
                features, targets = self.augment_fn(
                    features, targets, seed=c)
            loss = self.train_step(features, targets)
            losses += loss[0]
            if self.compute_accuracy:
                acc += loss[1]
            c += 1
        else:  # Test step
            for features_, targets_ in self.test_loader:
                if self.augment_fn is not None:
                    features_, targets_ = self.augment_fn(
                        features_, targets_, seed=c_test)
                loss_ = self.test_step(features_, targets_)
                losses_test += loss_[0]
                if self.compute_accuracy:
                    acc_test += loss_[1]
                c_test += 1
        self.loss_acc["train_loss"].append(losses / c)
        self.loss_acc["test_loss"].append(losses_test / c_test)
        if self.compute_accuracy:
            self.loss_acc["train_accuracy"].append(acc / c)
            self.loss_acc["test_accuracy"].append(acc_test / c_test)

    def eval_model(self) -> None:
        """
        Evaluates model on the entire dataset
        """
        self.net.eval()
        running_loss_test, c = 0, 0
        if self.compute_accuracy:
            running_acc_test = 0
        if self.full_epoch:
            for features_, targets_ in self.test_loader:
                if self.augment_fn is not None:
                    features, targets = self.augment_fn(
                        features_, targets_, seed=c)
                loss_ = self.test_step(features_, targets_)
                running_loss_test += loss_[0]
                if self.compute_accuracy:
                    running_acc_test += loss_[1]
                c += 1
            print('Model (final state) evaluation loss:',
                  np.around(running_loss_test / c, 4))
            if self.compute_accuracy:
                print('Model (final state) IoU:',
                      np.around(running_acc_test / c, 4))
        else:
            running_loss_test, running_acc_test = 0, 0
            for idx in range(len(self.X_test)):
                features_, targets_ = self.dataloader(idx, mode='test')
                loss_ = self.test_step(features_, targets_)
                running_loss_test += loss_[0]
                if self.compute_accuracy:
                    running_acc_test += loss_[1]
            print('Model (final state) evaluation loss:',
                  np.around(running_loss_test / len(self.X_test), 4))
            if self.compute_accuracy:
                print('Model (final state) IoU:',
                      np.around(running_acc_test / len(self.X_test), 4))

    def dataloader(self,
                   batch_num: int,
                   mode: str = 'train') -> Tuple[torch.Tensor]:
        """
        Generates input training data with images/spectra
        and the associated labels (spectra/images)
        """
        if mode == 'test':
            features = self.X_test[batch_num][:self.batch_size]
            targets = self.y_test[batch_num][:self.batch_size]
        else:
            features = self.X_train[batch_num][:self.batch_size]
            targets = self.y_train[batch_num][:self.batch_size]
        if self.augment_fn is not None:
            features, targets = self.augment_fn(
                features, targets, seed=len(self.loss_acc["train_loss"]))
        return features, targets

    def save_model(self, *args: str) -> None:
        """
        Saves trained weights, optimizer and key information about model's
        architecture (the latter works only for built-in AtomAI models)
        """
        try:
            filename = args[0]
        except IndexError:
            filename = self.filename
        self.meta_state_dict["weights"] = self.meta_state_dict.get(
            "weights", self.net.state_dict())
        self.meta_state_dict["optimizer"] = self.meta_state_dict.get(
            "optimizer", self.optimizer)
        torch.save(self.meta_state_dict,
                   filename + '.tar')

    def print_statistics(self, e: int, **kwargs) -> None:
        """
        Print loss and (optionally) IoU score on train
        and test data, as well as GPU memory usage.
        """
        accuracy_metrics = self.accuracy_metrics
        if accuracy_metrics is None:
            accuracy_metrics = "Accuracy"
        if torch.cuda.is_available():
            gpu_usage = gpu_usage_map(torch.cuda.current_device())
        else:
            gpu_usage = ['N/A ', ' N/A']
        if self.compute_accuracy:
            print('Epoch {}/{} ...'.format(e+1, self.training_cycles),
                  'Training loss: {} ...'.format(
                      np.around(self.loss_acc["train_loss"][-1], 4)),
                  'Test loss: {} ...'.format(
                      np.around(self.loss_acc["test_loss"][-1], 4)),
                  'Train {}: {} ...'.format(
                      accuracy_metrics,
                      np.around(self.loss_acc["train_accuracy"][-1], 4)),
                  'Test {}: {} ...'.format(
                      accuracy_metrics,
                      np.around(self.loss_acc["test_accuracy"][-1], 4)),
                  'GPU memory usage: {}/{}'.format(
                      gpu_usage[0], gpu_usage[1]))
        else:
            print('Epoch {}/{} ...'.format(e+1, self.training_cycles),
                  'Training loss: {} ...'.format(
                      np.around(self.loss_acc["train_loss"][-1], 4)),
                  'Test loss: {} ...'.format(
                      np.around(self.loss_acc["test_loss"][-1], 4)),
                  'GPU memory usage: {}/{}'.format(
                      gpu_usage[0], gpu_usage[1]))

    def accuracy_fn(self, *args) -> None:
        """
        Computes accuracy score
        """
        raise NotImplementedError

    def weight_perturbation(self, e: int) -> None:
        """
        Time-dependent weights perturbation
        (role of time is played by "epoch" number)
        """
        a = self.perturb_weights["a"]
        gamma = self.perturb_weights["gamma"]
        e_p = self.perturb_weights["e_p"]
        if self.perturb_weights and (e + 1) % e_p == 0:
            var = torch.tensor(a / (1 + e)**gamma)
            for k, v in self.net.state_dict().items():
                v_prime = v + v.new(v.shape).normal_(0, torch.sqrt(var))
                self.net.state_dict()[k].copy_(v_prime)
        return

    def save_running_weights(self, e: int) -> None:
        """
        Saves running weights (for stochastic weights averaging)
        """
        swa_epochs = 5 if self.full_epoch else 30
        if self.training_cycles - e <= swa_epochs:
            i_ = swa_epochs - (self.training_cycles - e)
            state_dict_ = OrderedDict()
            for k, v in self.net.state_dict().items():
                state_dict_[k] = copy.deepcopy(v).cpu()
            self.running_weights[i_] = state_dict_
        return

    def data_augmentation(self,
                          augment_fn: augfn_type) -> None:
        """
        Set up data augmentation. To use it, pass a function that takes
        two torch tensors (features and targets), peforms some transforms,
        and returns the transformed tensors. The dimensions of the transformed
        tensors must be the same as the dimensions of the original ones.
        """
        self.augment_fn = augment_fn

    def compile_trainer(self,
                        train_data: Union[Tuple[torch.Tensor], Tuple[np.ndarray]] = None,
                        loss: Union[str, Callable] = 'ce',
                        optimizer: Optional[Type[torch.optim.Optimizer]] = None,
                        training_cycles: int = 1000,
                        batch_size: int = 32,
                        compute_accuracy: bool = False,
                        full_epoch: bool = False,
                        swa: bool = False,
                        perturb_weights: bool = False,
                        **kwargs):
        """
        Compile a trainer

        Args:
            train_data:
                4-element tuple of ndarrays or torch tensors
                (train_data, train_labels, test_data, test_labels)
            loss:
                loss function. Available loss functions are: 'mse' (MSE),
                'ce' (cross-entropy), 'focal' (focal loss; single class only),
                and 'dice' (dice loss; for semantic segmentation problems).
                One can also pass a custom loss function that takes prediction
                and ground truth and computes a loss score.
            optimizer:
                weights optimizer (defaults to Adam optimizer with lr=1e-3)
            training_cycles:
                Number of training 'epochs'.
                If full_epoch argument is set to False, 1 epoch == 1 batch.
                Otherwise, each cycle corresponds to all mini-batches of data
                passing through a NN.
            batch_size:
                Size of training and test batches
            compute_accuracy:
                Computes accuracy function at each training cycle
            full_epoch:
                If True, passes all mini-batches of training/test data
                at each training cycle and computes the average loss. If False,
                passes a single (randomly chosen) mini-batch at each cycle.
            swa:
                Saves the recent stochastic weights and averages
                them at the end of training
            perturb_weights:
                Time-dependent weight perturbation, :math:`w\\leftarrow w + a / (1 + e)^\\gamma`,
                where parameters *a* and *gamma* can be passed as a dictionary,
                together with parameter *e_p* determining every n-th epoch at
                which a perturbation is applied
            **batch_seed (int):
                Random state for generating sequence of training and test batches
            **overwrite_train_data (bool):
                Overwrites the exising training data using self.set_data()
                (Default: True)
            **memory_alloc (float):
                threshold (in GB) for holding all training data on GPU
            **print_loss (int):
                Prints loss every *n*-th epoch
            **accuracy_metrics (str):
                Accuracy metrics (used only for printing training statistics)
            **filename (str):
                Filename for model weights
                (appended with "_test_weights_best.pt" and "_weights_final.pt")
            **plot_training_history (bool):
                Plots training and test curves vs epochs at the end of training
        """
        self.full_epoch = full_epoch
        self.training_cycles = training_cycles
        self.batch_size = batch_size
        self.compute_accuracy = compute_accuracy
        self.swa = swa
        alloc = kwargs.get("memory_alloc", 4)

        if self.data_is_set:
            if kwargs.get("overwrite_train_data", True):
                self.set_data(*train_data, memory_alloc=alloc)
            else:
                pass
        else:
            self.set_data(*train_data, memory_alloc=alloc)

        self.perturb_weights = perturb_weights
        if self.perturb_weights:
            if self.meta_state_dict["batchnorm"]:
                raise AssertionError(
                    "To use time-dependent weights perturbation, " +
                    "turn off the batch normalization layes")
            if isinstance(self.perturb_weights, bool):
                e_p = 1 if self.full_epoch else 50
                self.perturb_weights = {"a": .01, "gamma": 1.5, "e_p": e_p}

        params = self.net.parameters()
        if optimizer is None:
            self.optimizer = torch.optim.Adam(params, lr=1e-3)
        else:
            self.optimizer = optimizer(params)
        self.criterion = self.get_loss_fn(loss, self.nb_classes)

        if not self.full_epoch:
            r = self.training_cycles // len(self.X_train)
            batch_idx_train = np.arange(
                len(self.X_train)).repeat(r+1)[:self.training_cycles]
            r_ = self.training_cycles // len(self.X_test)
            batch_idx_test = np.arange(
                len(self.X_test)).repeat(r_+1)[:self.training_cycles]
            self.batch_idx_train = shuffle(
                batch_idx_train, random_state=kwargs.get("batch_seed", 1))
            self.batch_idx_test = shuffle(
                batch_idx_test, random_state=kwargs.get("batch_seed", 1))
            #self.batch_idx_train = np.random.randint(
            #    0, len(self.X_train), self.training_cycles)
            #self.batch_idx_test = np.random.randint(
            #    0, len(self.X_test), self.training_cycles)

        self.print_loss = kwargs.get("print_loss")
        if self.print_loss is None:
            if not self.full_epoch:
                self.print_loss = 100
            else:
                self.print_loss = 1
        self.accuracy_metrics = kwargs.get("accuracy_metrics")
        self.filename = kwargs.get("filename", "./model")
        self.plot_training_history = kwargs.get("plot_training_history", True)

    def run(self) -> Type[torch.nn.Module]:
        """
        Trains a neural network, prints the statistics,
        saves the final model weights. One can also pass
        kwargs for utils.datatransform class to perform
        the data augmentation "on-the-fly"
        """
        for e in range(self.training_cycles):
            if self.full_epoch:
                self.step_full()
            else:
                self.step(e)
            if self.swa:
                self.save_running_weights(e)
            if self.perturb_weights:
                self.weight_perturbation(e)
            if any([e == 0, (e+1) % self.print_loss == 0,
                    e == self.training_cycles-1]):
                self.print_statistics(e)
        self.save_model(self.filename + "_metadict_final")
        if not self.full_epoch:
            self.eval_model()
        if self.swa:
            print("Performing stochastic weights averaging...")
            self.net.load_state_dict(average_weights(self.running_weights))
            self.eval_model()
        if self.plot_training_history:
            plot_losses(self.loss_acc["train_loss"],
                        self.loss_acc["test_loss"])
        return self.net

    def fit(self) -> None:
        _ = self.run()


class SegTrainer(BaseTrainer):
    """
    Class for training a fully convolutional neural network
    for semantic segmentation of noisy experimental data

    Args:
        model:
            Type of model to train: 'Unet', 'Uplusnet' or 'dilnet' (Default: 'Unet').
            See atomai.nets for more details. One can also pass a custom fully
            convolutional neural network model.
        nb_classes:
            Number of classes in the classification scheme adopted
            (must match the number of classes in training data)
        **seed (int):
            Deterministic mode for model training (Default: 1)
        **batch_seed (int):
            Separate seed for generating a sequence of batches
            for training/testing. Equal to 'seed' if set to None (default)
        **batch_norm (bool):
            Apply batch normalization after each convolutional layer
            (Default: True)
        **dropout (bool):
            Apply dropouts in the three inner blocks in the middle of a network
            (Default: False)
        **upsampling (str):
            "bilinear" or "nearest" upsampling method (Default: "bilinear")
        **nb_filters (int):
            Number of convolutional filters in the first convolutional block
            (this number doubles in the consequtive block(s),
            see definition of Unet and dilnet models for details)
        **with_dilation (bool):
            Use dilated convolutions in the bottleneck of Unet
            (Default: False)
        **layers (list):
            List with a number of layers in each block.
            For U-Net the first 4 elements in the list
            are used to determine the number of layers
            in each block of the encoder (including bottleneck layer),
            and the number of layers in the decoder  is chosen accordingly
            (to maintain symmetry between encoder and decoder)
    """
    def __init__(self,
                 model: Union[Type[torch.nn.Module], str] = "Unet",
                 nb_classes: int = 1,
                 **kwargs: Union[int, List, str, bool]) -> None:
        """
        Initialize a single FCNN model trainer
        """
        super(SegTrainer, self).__init__()
        seed = kwargs.get("seed", 1)
        kwargs["batch_seed"] = kwargs.get("batch_seed", seed)
        set_train_rng(seed)
        self.nb_classes = nb_classes
        self.net, self.meta_state_dict = init_fcnn_model(
                                model, self.nb_classes, **kwargs)
        self.net.to(self.device)
        if self.device == 'cpu':
            warnings.warn(
                "No GPU found. The training can be EXTREMELY slow",
                UserWarning)
        self.meta_state_dict["weights"] = self.net.state_dict()
        self.meta_state_dict["optimizer"] = self.optimizer

    def set_data(self,
                 X_train: Tuple[np.ndarray, torch.Tensor],
                 y_train: Tuple[np.ndarray, torch.Tensor],
                 X_test: Optional[Tuple[np.ndarray, torch.Tensor]] = None,
                 y_test: Optional[Tuple[np.ndarray, torch.Tensor]] = None,
                 **kwargs: Union[float, int]) -> None:
        """
        Sets training and test data.

        Args:
            X_train:
                4D numpy array or pytorch tensor of training images
                (n_samples, 1, height, width). One can also pass a regular
                3D image stack without a channel dimension of 1 which will
                be added automatically
            y_train:
                4D (binary) / 3D (multiclass) numpy array or pytorch tensor
                of training masks (aka ground truth) stacked along
                the first dimension. The reason why in the multiclass case
                the X_train is 4-dimensional and the y_train is 3-dimensional
                is because of how the cross-entropy loss is calculated in PyTorch
                (see https://pytorch.org/docs/stable/nn.html#nllloss).
            X_test:
                4D numpy array or pytorch tensor of test images
                (stacked along the first dimension)
            y_test:
                4D (binary) / 3D (multiclass) numpy array or pytorch tensor
                of training masks (aka ground truth) stacked along
                the first dimension.
            kwargs:
                Parameters for train_test_split ('test_size' and 'seed') when
                separate test set is not provided and 'memory_alloc', which
                sets a threshold (in GBs) for holding entire training data on GPU
        """

        if X_test is None or y_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=kwargs.get("test_size", .15),
                shuffle=True, random_state=kwargs.get("seed", 1))

        if self.full_epoch:
            loaders = init_fcnn_dataloaders(
                X_train, y_train, X_test, y_test,
                self.batch_size, memory_alloc=kwargs.get("memory_alloc", 4))
            self.train_loader, self.test_loader, nb_classes = loaders
        else:
            (self.X_train, self.y_train,
             self.X_test, self.y_test,
             nb_classes) = preprocess_training_image_data(
                                    X_train, y_train, X_test, y_test,
                                    self.batch_size,
                                    kwargs.get("memory_alloc", 4))

        if self.nb_classes != nb_classes:
            raise AssertionError("Number of classes in initialized model" +
                                 " is different from the number of classes" +
                                 " contained in training data")

    def accuracy_fn(self,
                    y: torch.Tensor,
                    y_prob: torch.Tensor,
                    *args):
        iou_score = losses_metrics.IoU(
                y, y_prob, self.nb_classes).evaluate()
        return iou_score


class ImSpecTrainer(BaseTrainer):
    """
    Trainer of neural network for image-to-spectrum
    and spectrum-to-image transformations

    Args:
        in_dim:
            Input data dimensions.
            (height, width) for images or (length,) for spectra
        out_dim:
            output dimensions.
            (length,) for spectra or (height, width) for images
        latent_dim:
            dimensionality of the latent space
            (number of neurons in a fully connected bottleneck layer)
        **seed (int):
            Deterministic mode for model training (Default: 1)
        **batch_seed (int):
            Separate seed for generating a sequence of batches
            for training/testing. Equal to 'seed' if set to None (default)
        **nblayers_encoder (int):
            number of convolutional layers in the encoder
        **nblayers_decoder (int):
            number of convolutional layers in the decoder
        **nbfilters_encoder (int):
            number of convolutional filters in each layer of the encoder
        **nbfilters_decoder (int):
            number of convolutional filters in each layer of the decoder
        **batch_norm (bool):
            Apply batch normalization after each convolutional layer
            (Default: True)
        **encoder_downsampling (int):
            downsamples input data by this factor before passing
            to convolutional layers (Default: no downsampling)
        **decoder_upsampling (bool):
            performs upsampling+convolution operation twice on the reshaped latent
            vector (starting from image/spectra dims 4x smaller than the target dims)
            before passing  to the decoder
    """
    def __init__(self,
                 in_dim: Tuple[int],
                 out_dim: Tuple[int],
                 latent_dim: int = 2,
                 **kwargs: Union[int, bool, str]) -> None:
        super(ImSpecTrainer, self).__init__()
        """
        Initialize trainer's parameters
        """
        seed = kwargs.get("seed", 1)
        kwargs["batch_seed"] = kwargs.get("batch_seed", seed)
        set_train_rng(seed)
        
        self.in_dim, self.out_dim = in_dim, out_dim
        (self.net,
         self.meta_state_dict) = init_imspec_model(in_dim, out_dim, latent_dim,
                                                   **kwargs)

        self.net.to(self.device)
        self.meta_state_dict["weights"] = self.net.state_dict()
        self.meta_state_dict["optimizer"] = self.optimizer

    def set_data(self,
                 X_train: Union[np.ndarray, torch.Tensor],
                 y_train: Union[np.ndarray, torch.Tensor],
                 X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 **kwargs: Union[float, int]) -> None:
        """
        Sets training and test data.

        Args:
            X_train:
                4D numpy array or torch tensor with image data
                (n_samples x 1 x height x width) or 3D array/tensor
                with spectral data (n_samples x 1 x signal_length).
                It is also possible to pass 3D and 2D arrays by ignoring
                the channel dim of 1, which will be added automatically.
                The X_train is typically referred to as 'features'
            y_train:
                3D numpy array or torch tensor with spectral data
                (n_samples x 1 x signal_length) or 4D array/tensor with
                image data (n_samples x 1 x height x width).
                It is also possible to pass 2D and 3D arrays by ignoring
                the channel dim of 1, which will be added automatically.
                Note that if your X_train data are images,
                then your y_train must be spectra and vice versa.
                The y_train is typicaly referred to as "targets"
            X_test:
                Test data (features) of the same dimesnionality
                (except for the number of samples) as X_train
            y_test:
                Test data (targets) of the same dimesnionality
                (except for the number of samples) as y_train
            kwargs:
                Parameters for train_test_split ('test_size' and 'seed') when
                separate test set is not provided and 'memory_alloc', which
                sets a threshold (in GBs) for holding entire training data on GPU
        """

        if X_test is None or y_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=kwargs.get("test_size", .15),
                shuffle=True, random_state=kwargs.get("seed", 1))

        if self.full_epoch:
            self.train_loader, self.test_loader, dims = init_imspec_dataloaders(
                X_train, y_train, X_test, y_test,
                self.batch_size, kwargs.get("memory_alloc", 4))
        else:
            (self.X_train, self.y_train,
             self.X_test, self.y_test, dims) = preprocess_training_imspec_data(
                X_train, y_train, X_test, y_test,
                self.batch_size, kwargs.get("memory_alloc", 4))

        if dims[0] != self.in_dim or dims[1] != self.out_dim:
            raise AssertionError(
                "The input/output dimensions of the model must match" +
                " the height, width and length (for spectra) of training")
