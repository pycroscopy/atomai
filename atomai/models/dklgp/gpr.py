from typing import Tuple

import numpy as np
import gpytorch
import torch

from ...trainers import GPTrainer
from ...utils import prepare_gp_input, create_batches


class Reconstructor(GPTrainer):

    def __init__(self, input_dim: int, **kwargs):
        super(Reconstructor, self).__init__(input_dim, **kwargs)

    def fit(self, X: torch.Tensor, y: torch.Tensor,
            training_cycles: int, **kwargs):
        self.run(X, y, training_cycles, **kwargs)

    def predict(self, X_new: torch.Tensor, img_shape: Tuple[int], **kwargs):
        batch_size = kwargs.get("batch_size", len(X_new))
        X_new_batches = create_batches(X_new, batch_size)
        self.gp_model.eval()
        self.likelihood.eval()
        reconstruction = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for x in X_new_batches:
                self._set_data(x, **kwargs)
                y_pred = self.likelihood(self.gp_model(x))
                reconstruction.append(y_pred)
        reconstruction = torch.cat(reconstruction)
        return reconstruction.mean.view(img_shape).cpu().numpy()

    def reconstruct(self, sparse_image: np.ndarray,
                    training_cycles: int = 100, **kwargs):
        X_train, y_train, X_full = prepare_gp_input(sparse_image)
        self.fit(X_train, y_train, training_cycles, **kwargs)
        return self.predict(X_full, sparse_image.shape, **kwargs)
