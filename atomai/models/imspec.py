from typing import Type, Union, Tuple, Optional, Dict, Callable
import torch
import numpy as np
from ..trainers import ImSpecTrainer
from ..predictors import ImSpecPredictor
from ..transforms import datatransform


class ImSpec(ImSpecTrainer):
    """
    Model for converting images to spectra and vice versa

    Args:
        in_dim (tuple):
            Input data dimensions.
            (height, width) for images or (length,) for spectra
        out_dim (tuple):
            output dimensions.
            (length,) for spectra or (height, width) for images
        latent_dim (int):
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
                 **kwargs) -> None:
        super(ImSpec, self).__init__(in_dim, out_dim, latent_dim, **kwargs)
        self.latent_dim = latent_dim

    def fit(self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            X_test: Optional[torch.Tensor] = None,
            y_test: Optional[torch.Tensor] = None,
            loss: str = 'mse',
            optimizer: Optional[Type[torch.optim.Optimizer]] = None,
            training_cycles: int = 1000,
            batch_size: int = 64,
            compute_accuracy: bool = False,
            full_epoch: bool = False,
            swa: bool = False,
            perturb_weights: bool = False,
            **kwargs):
        """
        Compiles a trainer and performs model training

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
            loss (str):
                loss function. Available loss functions are: 'mse' (MSE),
                'ce' (cross-entropy), 'focal' (focal loss; single class only),
                and 'dice' (dice loss; for semantic segmentation problems)
            optimizer:
                weights optimizer (defaults to Adam optimizer with lr=1e-3)
            training_cycles (int): Number of training 'epochs'.
                If full_epoch argument is set to False, 1 epoch == 1 batch.
                Otherwise, each cycle corresponds to all mini-batches of data
                passing through a NN.
            batch_size (int):
                Size of training and test batches
            compute_accuracy (bool):
                Computes accuracy function at each training cycle
            full_epoch (bool):
                If True, passes all mini-batches of training/test data
                at each training cycle and computes the average loss. If False,
                passes a single (randomly chosen) mini-batch at each cycle.
            swa (bool):
                Saves the recent stochastic weights and averages
                them at the end of training
            perturb_weights (bool or dict):
                Time-dependent weight perturbation, :math:`w\\leftarrow w + a / (1 + e)^\\gamma`,
                where parameters *a* and *gamma* can be passed as a dictionary,
                together with parameter *e_p* determining every n-th epoch at
                which a perturbation is applied
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
        self.compile_trainer(
            (X_train, y_train, X_test, y_test),
            loss, optimizer, training_cycles, batch_size,
            compute_accuracy, full_epoch, swa, perturb_weights,
            **kwargs)
        
        self.augment_fn = imspec_augmentor(self.in_dim, self.out_dim, **kwargs)
        _ = self.run()

    def predict(self,
                data: np.ndarray,
                **kwargs) -> np.ndarray:
        """
        Apply trained model to new data

        Args:
            signal (numpy array): Input image/spectrum or batch of images/spectra
            **num_batches (int): number of batches (Default: 10)
            **verbose (bool): verbosity (Default: True)            
        """
        use_gpu = self.device == 'cuda'
        nn_output = ImSpecPredictor(
            self.net, self.out_dim, use_gpu,
            **kwargs).run(data, **kwargs)
        return nn_output

    def load_weights(self, filepath: str) -> None:
        """
        Loads saved weights dictionary
        """
        weight_dict = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(weight_dict)


def imspec_augmentor(in_dim: Tuple[int],
                     out_dim: Tuple[int],
                     **kwargs
                     ) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    auglist = ["custom_transform", "gauss_noise", "jitter",
               "poisson_noise", "contrast", "salt_and_pepper", "blur",
               "background"]
    augdict = {k: kwargs[k] for k in auglist if k in kwargs.keys()}
    if len(augdict) == 0:
        return
    if len(in_dim) < len(out_dim):
        raise NotImplementedError("The built-in data augmentor works only" +
                                  " for img->spec models (i.e. input is image)")

    def augmentor(features, targets, seed):
        features = features.cpu().numpy().astype(np.float64)
        targets = targets.cpu().numpy().astype(np.float64)
        dt = datatransform(seed, **augdict)
        features, targets = dt.run(features[:, 0, ...], targets)
        features = torch.from_numpy(features).float()
        targets = torch.from_numpy(targets).float()
        return features, targets

    return augmentor