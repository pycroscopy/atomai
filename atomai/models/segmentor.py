from typing import Type, Union, Tuple, Optional, Dict, Callable
import torch
import numpy as np
from ..trainers import SegTrainer
from ..predictors import SegPredictor
from ..transforms import datatransform, unsqueeze_channels
from ..utils import get_downsample_factor


class Segmentor(SegTrainer):
    """
    Model for semantic segmentation-based analysis of images

    Args:
        model (str):
            Type of model to train: 'Unet' or 'dilnet' (Default: 'Unet').
            See atomai.nets for more details. One can also pass a custom fully
            convolutional neural network model.
        **batch_norm (bool):
            Apply batch normalization after each convolutional layer
            (Default: True)
        **dropout (bool):
            Apply dropouts in the three inner blocks in the middle of a network
            (Default: False)
        **upsampling_mode (str):
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
                 model: Type[Union[str, torch.nn.Module]] = "Unet",
                 nb_classes: int = 1,
                 **kwargs) -> None:
        super(Segmentor, self).__init__(model, nb_classes, **kwargs)
        self.downsample_factor = None

    def fit(self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            X_test: Optional[torch.Tensor] = None,
            y_test: Optional[torch.Tensor] = None,
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
            X_train (numpy array):
                4D numpy array (3D image tensors stacked along the first dim)
                representing training images
            y_train (numpy array):
                4D (binary) / 3D (multiclass) numpy array
                where 3D / 2D images stacked along the first array dimension
                represent training labels (aka masks aka ground truth).
                The reason why in the multiclass case the images are 4-dimensional
                tensors and the labels are 3-dimensional tensors is because of how
                the cross-entropy loss is calculated in PyTorch
                (see https://pytorch.org/docs/stable/nn.html#nllloss).
            X_test (numpy array):
                4D numpy array (3D image tensors stacked along the first dim)
                representing test images
            y_test (numpy array):
                4D (binary) / 3D (multiclass) numpy array
                where 3D / 2D images stacked along the first array dimension
                represent test labels (aka masks aka ground truth)
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
            batch_size (int): Size of training and test batches
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
        
        self.augment_fn = seg_augmentor(self.nb_classes, **kwargs)
        _ = self.run()

    def predict(self,
                imgdata: np.ndarray,
                refine: bool = False,
                logits: bool = True,
                **kwargs) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Apply trained model to new data

        Args:
            image_data (2D or 3D numpy array):
                Image stack or a single image (all greyscale)
            refine (bool):
                Atomic positions refinement with 2d Gaussian peak fitting
            logits (bool):
                Indicates that the image data is passed through
                a softmax/sigmoid layer when set to False
                (logits=True for AtomAI models)
            **thresh (float):
                value between 0 and 1 for thresholding the NN output
            **d (int):
                half-side of a square around each atomic position used
                for refinement with 2d Gaussian peak fitting. Defaults to 1/4
                of average nearest neighbor atomic distance
            **num_batches (int): number of batches (Default: 10)
            **norm (bool): Normalize data to (0, 1) during pre-processing
            **verbose (bool): verbosity

        """
        if self.downsample_factor is None:
            self.downsample_factor = get_downsample_factor(self.net)
        use_gpu = self.device == 'cuda'
        nn_output, coords = SegPredictor(
            self.net, refine, None, use_gpu, logits,
            nb_classes=self.nb_classes, downsampling=self.downsample_factor,
            **kwargs).run(imgdata, **kwargs)

        return nn_output, coords

    def load_weights(self, filepath: str) -> None:
        """
        Loads saved weights dictionary
        """
        weight_dict = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(weight_dict)


def seg_augmentor(nb_classes: int,
                  **kwargs
                  ) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

    auglist = ["custom_transform", "zoom", "gauss_noise", "jitter",
               "poisson_noise", "contrast", "salt_and_pepper", "blur",
               "resize", "rotation", "background"]
    augdict = {k: kwargs[k] for k in auglist if k in kwargs.keys()}
    if len(augdict) == 0:
        return

    def augmentor(images, labels, seed):
        images = images.cpu().numpy().astype(np.float64)
        labels = labels.cpu().numpy().astype(np.float64)
        dt = datatransform(
                nb_classes, "channel_first", 'channel_first',
                True, seed, **augdict)
        images, labels = dt.run(
            images[:, 0, ...], unsqueeze_channels(labels, nb_classes))
        images = torch.from_numpy(images).float()
        if nb_classes == 1:
            labels = torch.from_numpy(labels).float()
        else:
            labels = torch.from_numpy(labels).long()
        return images, labels

    return augmentor
