from typing import Type, Union, Tuple, Optional, Dict
import torch
import numpy as np
from ..trainers import SegTrainer
from ..predictors import SegPredictor
from ..transforms import seg_augmentor
from ..utils import get_downsample_factor


class Segmentor(SegTrainer):
    """
    Model for semantic segmentation-based analysis of images

    Args:
        model:
            Type of model to train: 'Unet', 'Uplusnet' or 'dilnet' (Default: 'Unet').
            See atomai.nets for more details. One can also pass here a custom
            fully convolutional neural network model.
        nb_classes:
            Number of classes in classification scheme (last NN layer)
        **batch_norm (bool):
            Apply batch normalization after each convolutional layer
            (Default: True)
        **dropout (bool):
            Apply dropouts to the three inner blocks in the middle of a network
            (Default: False)
        **upsampling_mode (str):
            "bilinear" or "nearest" upsampling method (Default: "bilinear")
        **nb_filters (int):
            Number of convolutional filters (aka "kernels") in the first
            convolutional block (this number doubles in the consequtive block(s),
            see definition of *Unet* and *dilnet* models for details)
        **with_dilation (bool):
            Use dilated convolutions in the bottleneck of *Unet*
            (Default: False)
        **layers (list):
            List with a number of layers in each block.
            For *UNet* the first 4 elements in the list
            are used to determine the number of layers
            in each block of the encoder (including bottleneck layer),
            and the number of layers in the decoder  is chosen accordingly
            (to maintain symmetry between encoder and decoder)

    Example:

    >>> # Initialize model
    >>> model = aoi.models.Segmentor(nb_classes=3)
    >>> # Train
    >>> model.fit(images, masks, images_test, masks_test,
    >>>        training_cycles=300, compute_accuracy=True, swa=True)
    >>> # Predict with trained model
    >>> nn_output, coordinates = model.predict(expdata)
    """
    def __init__(self,
                 model: Type[Union[str, torch.nn.Module]] = "Unet",
                 nb_classes: int = 1,
                 **kwargs) -> None:
        super(Segmentor, self).__init__(model, nb_classes, **kwargs)
        self.downsample_factor = None

    def fit(self,
            X_train: Union[np.ndarray, torch.Tensor],
            y_train: Union[np.ndarray, torch.Tensor],
            X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            loss: str = 'ce',
            optimizer: Optional[Type[torch.optim.Optimizer]] = None,
            training_cycles: int = 1000,
            batch_size: int = 32,
            compute_accuracy: bool = False,
            full_epoch: bool = False,
            swa: bool = False,
            perturb_weights: bool = False,
            **kwargs):
        """
        Compiles a trainer and performs model training

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
            loss:
                loss function. Available loss functions are: 'mse' (MSE),
                'ce' (cross-entropy), 'focal' (focal loss; single class only),
                and 'dice' (dice loss)
            optimizer:
                weights optimizer (defaults to Adam optimizer with lr=1e-3)
            training_cycles: Number of training 'epochs'.
                If full_epoch argument is set to False, 1 epoch == 1 mini-batch.
                Otherwise, each cycle corresponds to all mini-batches of data
                passing through a NN.
            batch_size: Size of training and test mini-batches
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
                together with parameter *e_p* determining every *n*-th epoch at
                which a perturbation is applied
            **print_loss (int):
                Prints loss every *n*-th epoch
            **accuracy_metrics (str):
                Accuracy metrics (used only for printing training statistics)
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

        self.augment_fn = seg_augmentor(self.nb_classes, **kwargs)
        _ = self.run()

    def predict(self,
                imgdata: Union[np.ndarray, torch.Tensor],
                refine: bool = False,
                logits: bool = True,
                resize: Tuple[int, int] = None,
                compute_coords: bool = True,
                **kwargs) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Apply (trained) model to new data

        Args:
            image_data:
                3D image stack or a single 2D image (all greyscale)
            refine:
                Atomic positions refinement with 2d Gaussian peak fitting
                (may take some time)
            logits:
                Indicates whether the features are passed through
                a softmax/sigmoid layer at the end of a neural network
                (logits=True for AtomAI models)
            resize:
                Resizes input data to (new_height, new_width) before passing
                to a neural network
            compute_coords (bool):
                Computes centers of the mass of individual blobs
                in the segmented images (Default: True)
            **thresh (float):
                Value between 0 and 1 for thresholding the NN output
                (Default: 0.5)
            **d (int):
                half-side of a square around each atomic position used
                for refinement with 2d Gaussian peak fitting. Defaults to 1/4
                of average nearest neighbor atomic distance
            **num_batches (int): number of batches (Default: 10)
            **norm (bool): Normalize data to (0, 1) during pre-processing
            **verbose (bool): verbosity
        
        Returns:
            Semantically segmented image and coordinates of (atomic) objects

        """
        if self.downsample_factor is None:
            self.downsample_factor = get_downsample_factor(self.net)
        use_gpu = self.device == 'cuda'
        prediction = SegPredictor(
            self.net, refine, resize, use_gpu, logits,
            nb_classes=self.nb_classes, downsampling=self.downsample_factor,
            **kwargs).run(imgdata, compute_coords, **kwargs)

        return prediction

    def load_weights(self, filepath: str) -> None:
        """
        Loads saved weights dictionary
        """
        weight_dict = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(weight_dict)
