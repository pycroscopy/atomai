from typing import Tuple, Optional, List

import numpy as np
import gpytorch
import torch

from ...trainers import GPTrainer
from ...utils import prepare_gp_input, create_batches, get_lengthscale_constraints


class Reconstructor(GPTrainer):

    """
    Sparse image reconstructor based on the structured kernel interpolation framework.

    Keyword Args:
        device:
            Sets device to which model and data will be moved.
            Defaults to 'cuda:0' if a GPU is available and to CPU otherwise.
        precision:
            Sets tensor types for 'single' (torch.float32)
            or 'double' (torch.float64) precision
        seed:
            Seed for enforcing reproducibility
    """

    def __init__(self, **kwargs):
        super(Reconstructor, self).__init__(**kwargs)

    def fit(self, X: torch.Tensor, y: torch.Tensor,
            training_cycles: int, **kwargs):
        """
        Performs model training

        Args:
            X: Input training data. Usually, these are indices of pixels where the sparse measurements were performed.
            The dimensions of X should be (N, num_features). For 2D images, it will be (N, 2).
            y: Output targets of (N,) dimensions (usually, these are pixel values)
            training_cycles: Number of training epochs

        Keyword Args:
            grid_points_ratio: Determines a grid size for the KISS-GP kernel. Defaults to 1.0 (recommended)
            lr: learning rate (Default: 0.01)
            kernel_type: Type of kernel to use, either 'sparse' or 'kissgp'.
            base_kernel: Name of the base kernel as a string, either 'rbf' or 'matern', or a custom base kernel object.
            inducing_points: Inducing points for the sparse kernel.
            lengthscale_contraints: Optional lengthscale constraints for the base kernel.
            print_loss: print loss at every n-th training cycle (epoch)
        """
        _ = self.run(X, y, training_cycles, **kwargs)

    def predict(self, X_new: torch.Tensor, **kwargs):
        """
        Prediction on new data

        Args:
            X_new: new inputs (usually, a full set of image indices)

        Keyword Args:
            batch_size: batch size for a batch-by-batch prediction (to avoid memory overflow)
            device: Sets device to which model and data will be moved.
            Defaults to 'cuda:0' if a GPU is available and to CPU otherwise.

        Returns:
            Predictive mean
        """
        batch_size = kwargs.get("batch_size", len(X_new))
        device = kwargs.get("device")
        X_new_batches = create_batches(X_new, batch_size)
        self.gp_model.eval()
        self.likelihood.eval()
        reconstruction = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for x in X_new_batches:
                x = self._set_data(x, device)
                y_pred = self.likelihood(self.gp_model(x))
                reconstruction.append(y_pred.mean)
        return torch.cat(reconstruction)

    def reconstruct(self, sparse_image: np.ndarray,
                    training_cycles: int = 100,
                    lengthscale_constraints: Optional[Tuple[List[float]]] = None,
                    grid_points_ratio: float = 1.0, **kwargs):
        """
        Trains a reconstructor on sparse image pixels
        and uses the trained model to reconstruct the entire image.

        Args:
            sparse_image: Input sparse image. The non-measured pixels must be zeros.
            training_cycles: Number of training epochs
            lengthscale_contraints: Optional lengthscale constraints for the base kernel.
            grid_points_ratio: Determines a grid size for the KISS-GP kernel. Defaults to 1.0 (recommended)

        Keyword Args:
            lr: learning rate (Default: 0.01)
            kernel_type: Type of kernel to use, either 'sparse' or 'kissgp'.
            base_kernel: Name of the base kernel as a string, either 'rbf' or 'matern', or a custom base kernel object.
            inducing_points: Inducing points for the sparse kernel.
            print_loss: print loss at every n-th training cycle (epoch)

        Returns:
            Reconstructed image
        """
        X_train, y_train, X_full = prepare_gp_input(sparse_image)
        if not lengthscale_constraints:
            lengthscale_constraints = get_lengthscale_constraints(X_full)
        print("Model training ...\n")
        self.fit(X_train, y_train, training_cycles,
                 lengthscale_constraints=lengthscale_constraints,
                 grid_points_ratio=grid_points_ratio, **kwargs)
        print('\n\rPerforming reconstruction... ', end="")
        reconstruction = self.predict(X_full, **kwargs)
        print("Done")
        return reconstruction.view(sparse_image.shape).cpu().numpy()
