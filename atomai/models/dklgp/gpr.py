from typing import Tuple

import numpy as np
import gpytorch
import torch

from ...trainers import GPTrainer
from ...utils import prepare_gp_input, create_batches, get_lengthscale_constraints


class Reconstructor(GPTrainer):

    def __init__(self, **kwargs):
        super(Reconstructor, self).__init__(**kwargs)

    def fit(self, X: torch.Tensor, y: torch.Tensor,
            training_cycles: int, **kwargs):
        _ = self.run(X, y, training_cycles, **kwargs)

    def predict(self, X_new: torch.Tensor, **kwargs):
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
                    training_cycles: int = 100, lengthscale_constraints=None,
                    grid_points_ratio: float = 1.0, **kwargs):
        X_train, y_train, X_full = prepare_gp_input(sparse_image)
        if not lengthscale_constraints:
            lengthscale_constraints = get_lengthscale_constraints(X_full)
        self.fit(X_train, y_train, training_cycles,
                 lengthscale_constraints=lengthscale_constraints,
                 grid_points_ratio=grid_points_ratio, **kwargs)
        reconstruction = self.predict(X_full, **kwargs)
        return reconstruction.view(sparse_image.shape).cpu().numpy()
