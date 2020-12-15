from typing import Type, Union, Tuple, Optional
import torch
import numpy as np
from ..trainers import ImSpecTrainer
from ..predictors import ImSpecPredictor
from ..transforms import imspec_augmentor


class ImSpec(ImSpecTrainer):
    """
    Model for converting images to spectra and vice versa

    Args:
        in_dim:
            Input data dimensions.
            (height, width) for images or (length,) for spectra
        out_dim:
            Output dimensions.
            (length,) for spectra or (height, width) for images
        latent_dim:
            Dimensionality of the latent space
            (number of neurons in a fully connected "bottleneck" layer)
        **seed (int):
            Seed used when initializng model weights (Default: 1)
        **nblayers_encoder (int):
            Number of convolutional layers in the encoder
        **nblayers_decoder (int):
            Number of convolutional layers in the decoder
        **nbfilters_encoder (int):
            number of convolutional filters in each layer of the encoder
        **nbfilters_decoder (int):
            Number of convolutional filters in each layer of the decoder
        **batch_norm (bool):
            Apply batch normalization after each convolutional layer
            (Default: True)
        **encoder_downsampling (int):
            Downsamples input data by this factor before passing
            to convolutional layers (Default: no downsampling)
        **decoder_upsampling (bool):
            Performs upsampling+convolution operation twice on the reshaped latent
            vector (starting from image/spectra dims 4x smaller than the target dims)
            before passing  to the decoder

    Example:

    >>> in_dim = (16, 16)  # Input dimensions
    >>> out_dim = (64,)  # Output dimensions
    >>> # Initialize and train model
    >>> model = aoi.models.ImSpec(in_dim, out_dim, latent_dim=10)
    >>> model.fit(imgs_train, spectra_train, imgs_test, spectra_test,
    >>>        full_epoch=True, training_cycles=120, swa=True)
    >>> # Make a prediction with the trained model
    >>> prediction = model.predict(imgs_test, norm=False)
    """
    def __init__(self,
                 in_dim: Tuple[int],
                 out_dim: Tuple[int],
                 latent_dim: int = 2,
                 **kwargs) -> None:
        super(ImSpec, self).__init__(in_dim, out_dim, latent_dim, **kwargs)
        self.latent_dim = latent_dim

    def fit(self,
            X_train: Union[np.ndarray, torch.Tensor],
            y_train: Union[np.ndarray, torch.Tensor],
            X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
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
            loss:
                Loss function (currently works only with 'mse')
            optimizer:
                weights optimizer (defaults to Adam optimizer with lr=1e-3)
            training_cycles: Number of training 'epochs'.
                If full_epoch argument is set to False, 1 epoch == 1 mini-batch.
                Otherwise, each cycle corresponds to all mini-batches of data
                passing through a NN.
            batch_size:
                Size of training and test mini-batches
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
                together with parameter *e_p* determining every *n*-th epoch at
                which a perturbation is applied
            **print_loss (int):
                Prints loss every *n*-th epoch
            **filename (str):
                Filename for model weights
                (appended with "_test_weights_best.pt" and "_weights_final.pt")
            **plot_training_history (bool):
                Plots training and test curves vs. training cycles
                at the end of training
            **kwargs:
                One can also pass kwargs for utils.datatransform class
                to perform the augmentation "on-the-fly" (e.g. rotation=True,
                gauss_nois=[20, 60], etc.)
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
        Apply (trained model) to new data

        Args:
            data: Input image/spectrum or batch of images/spectra
            **num_batches (int): number of batches (Default: 10)
            **norm (bool): Normalize data to (0, 1) during pre-processing
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
