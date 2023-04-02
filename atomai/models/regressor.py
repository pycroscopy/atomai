from typing import Type, Union, Optional
import torch
import numpy as np
from ..trainers import RegTrainer
from ..predictors import RegPredictor
from ..transforms import reg_augmentor


class Regressor(RegTrainer):
    """
    Model for image->vector regression tasks

    Args:
        model:
            The backbone regressor model (defaults to 'mobilenet')
        out_dim:
            Output dimensions (Defaults to 1)
       
    Example:

    >>> # Initialize and train a regression model
    >>> model = aoi.models.Regressor(out_dim=1)  # single-output regression
    >>> model.fit(train_images, train_targets, test_images, test_targets,
    >>>           full_epoch=True, training_cycles=30, swa=True)
    >>> # Make a prediction with the trained model
    >>> prediction = model.predict(imgs_new, norm=True)
    """
    def __init__(self,
                 model: str = 'mobilenet',
                 out_dim: int = 1,
                 **kwargs) -> None:
        super(Regressor, self).__init__(out_dim, model, **kwargs)

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
                4D numpy array with image data (n_samples x 1 x height x width).
                It is also possible to pass 3D by ignoring the channel dim,
                which will be added automatically.
            y_train:
                2D numpy array with target values (n_samples x out_dim).
                For single-outut regression tasks, one can simply pass an (n_samples,) array
            X_test:
                4D numpy array with image data (n_samples x 1 x height x width).
                It is also possible to pass 3D by ignoring the channel dim,
                which will be added automatically.
            y_test:
                2D numpy array with target values (n_samples x out_dim).
                For single-outut regression tasks, one can simply pass an (n_samples,) array
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
                to perform the augmentation "on-the-fly"
                (e.g. gauss_noise=[20, 60], etc.)
        """
        self.compile_trainer(
            (X_train, y_train, X_test, y_test),
            loss, optimizer, training_cycles, batch_size,
            compute_accuracy, full_epoch, swa, perturb_weights,
            **kwargs)

        self.augment_fn = reg_augmentor(**kwargs)
        _ = self.run()

    def predict(self,
                data: np.ndarray,
                **kwargs) -> np.ndarray:
        """
        Apply (trained model) to new data

        Args:
            data: Input image or batch of images
            **num_batches (int): number of batches (Default: 10)
            **norm (bool): Normalize data to (0, 1) during pre-processing
            **verbose (bool): verbosity (Default: True)
        """
        use_gpu = self.device == 'cuda'
        nn_output = RegPredictor(
            self.net, self.output_size, use_gpu,
            **kwargs).run(data, **kwargs)
        return nn_output

    def load_weights(self, filepath: str) -> None:
        """
        Loads saved weights dictionary
        """
        weight_dict = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(weight_dict)
