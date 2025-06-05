"""
denoiser.py
===========

Denoising autoencoder model for image cleaning

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified with conventional batch normalization approach
"""

from typing import Type, Union, Optional, Tuple
import torch
import numpy as np
from ..trainers import BaseTrainer
from ..predictors import BasePredictor
from ..nets import ConvBlock, UpsampleBlock
from ..utils import set_train_rng, preprocess_denoiser_data


class DenoisingAutoencoder(BaseTrainer):
    """
    Denoising autoencoder model for image cleaning and noise reduction
    
    Args:
        encoder_filters: List of filter sizes for encoder layers (Default: [8, 16, 32, 64])
        decoder_filters: List of filter sizes for decoder layers (Default: [64, 32, 16, 8])
        encoder_layers: Number of convolutional layers per encoder block (Default: [1, 2, 2, 2])
        decoder_layers: Number of convolutional layers per decoder block (Default: [2, 2, 2, 1])
        use_batch_norm: Whether to use batch normalization in both encoder and decoder (Default: True)
        upsampling_mode: Upsampling method ('nearest' or 'bilinear') (Default: 'nearest')
        **seed: Random seed for reproducibility (Default: 1)
        
    Example:
        >>> # Initialize model
        >>> model = aoi.models.DenoisingAutoencoder()
        >>> # Train on noisy/clean image pairs
        >>> model.fit(noisy_images, clean_images, noisy_test, clean_test,
        >>>           training_cycles=500, swa=True)
        >>> # Denoise new images
        >>> cleaned = model.predict(new_noisy_images)
    """
    
    def __init__(self,
                 encoder_filters: list = [8, 16, 32, 64],
                 decoder_filters: list = [64, 32, 16, 8],
                 encoder_layers: list = [1, 2, 2, 2],
                 decoder_layers: list = [2, 2, 2, 1],
                 use_batch_norm: bool = False,
                 upsampling_mode: str = 'nearest',
                 **kwargs) -> None:
        """
        Initialize denoising autoencoder
        """
        super(DenoisingAutoencoder, self).__init__()
        
        seed = kwargs.get("seed", 1)
        set_train_rng(seed)
        
        # Store architecture parameters
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.use_batch_norm = use_batch_norm
        self.upsampling_mode = upsampling_mode
        
        # Build the autoencoder
        self.net = self._build_autoencoder()
        self.net.to(self.device)
        
        # Initialize meta state dict for saving/loading
        self.meta_state_dict = {
            "model_type": "denoising_autoencoder",
            "encoder_filters": encoder_filters,
            "decoder_filters": decoder_filters,
            "encoder_layers": encoder_layers,
            "decoder_layers": decoder_layers,
            "use_batch_norm": use_batch_norm,
            "upsampling_mode": upsampling_mode,
            "weights": self.net.state_dict()
        }
    
    def _build_autoencoder(self) -> torch.nn.Module:
        """
        Build the encoder-decoder architecture with consistent batch norm placement
        """
        # Build encoder
        encoder_modules = []
        in_channels = 1  # Assuming grayscale images
        
        for i, (filters, layers) in enumerate(zip(self.encoder_filters, self.encoder_layers)):
            # Add convolutional block with consistent batch norm usage
            encoder_modules.append(
                ConvBlock(ndim=2, nb_layers=layers, input_channels=in_channels,
                         output_channels=filters, batch_norm=self.use_batch_norm)
            )
            # Add max pooling (except for the last layer)
            if i < len(self.encoder_filters) - 1:
                encoder_modules.append(torch.nn.MaxPool2d(2, 2))
            in_channels = filters
        
        encoder = torch.nn.Sequential(*encoder_modules)
        
        # Build decoder
        decoder_modules = []
        
        for i, (filters, layers) in enumerate(zip(self.decoder_filters, self.decoder_layers)):
            # Add upsampling (except for the first layer)
            if i > 0:
                decoder_modules.append(
                    UpsampleBlock(ndim=2, input_channels=in_channels,
                                output_channels=in_channels, mode=self.upsampling_mode)
                )
            
            # Add convolutional block with same batch norm setting as encoder
            decoder_modules.append(
                ConvBlock(ndim=2, nb_layers=layers, input_channels=in_channels,
                         output_channels=filters, batch_norm=self.use_batch_norm)
            )
            in_channels = filters
        
        # Final output layer (no batch norm for final reconstruction)
        decoder_modules.append(torch.nn.Conv2d(in_channels, 1, 1))
        
        decoder = torch.nn.Sequential(*decoder_modules)
        
        # Combine encoder and decoder
        autoencoder = torch.nn.Sequential(encoder, decoder)
        
        return autoencoder
    
    def fit(self,
            X_train: Union[np.ndarray, torch.Tensor],
            y_train: Union[np.ndarray, torch.Tensor],
            X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            loss: str = 'mse',
            optimizer: Optional[Type[torch.optim.Optimizer]] = None,
            training_cycles: int = 500,
            batch_size: int = 32,
            compute_accuracy: bool = False,
            full_epoch: bool = False,
            swa: bool = True,
            perturb_weights: bool = False,
            **kwargs):
        """
        Train the denoising autoencoder
        
        Args:
            X_train: Noisy input images for training
            y_train: Clean target images for training
            X_test: Noisy input images for testing
            y_test: Clean target images for testing
            loss: Loss function (Default: 'mse')
            optimizer: Optimizer (Default: Adam with lr=1e-3)
            training_cycles: Number of training epochs
            batch_size: Batch size for training
            compute_accuracy: Whether to compute accuracy metrics
            full_epoch: Whether to use full epochs
            swa: Whether to use stochastic weight averaging
            perturb_weights: Whether to use weight perturbation
            **kwargs: Additional arguments for training
        """
        if X_test is None or y_test is None:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=kwargs.get("test_size", .15),
                shuffle=True, random_state=kwargs.get("seed", 1))
        
        # Preprocess data
        X_train, y_train, X_test, y_test = preprocess_denoiser_data(
            X_train, y_train, X_test, y_test)
            
        # Compile and run training
        self.compile_trainer(
            (X_train, y_train, X_test, y_test),
            loss=loss, optimizer=optimizer, training_cycles=training_cycles,
            batch_size=batch_size, compute_accuracy=compute_accuracy,
            full_epoch=full_epoch, swa=swa, perturb_weights=perturb_weights,
            **kwargs
        )
        
        self.run()
        
        # Update meta state dict
        self.meta_state_dict["weights"] = self.net.state_dict()
    
    def predict(self,
                data: Union[np.ndarray, torch.Tensor],
                **kwargs) -> np.ndarray:
        """
        Denoise input images
        
        Args:
            data: Input noisy images
            **num_batches: Number of batches for prediction (Default: 10)
            
        Returns:
            Denoised images
        """
        use_gpu = self.device == 'cuda'
        predictor = BasePredictor(self.net, use_gpu, **kwargs)
        
        # Ensure proper format for prediction
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                data = data[None, None, ...]  # Add batch and channel dims
            elif data.ndim == 3:
                data = data[:, None, ...]     # Add channel dim
        
        prediction = predictor.predict(data, **kwargs)

        return prediction.detach().cpu().numpy().squeeze()
    
    def load_weights(self, filepath: str) -> None:
        """
        Load saved model weights
        """
        weight_dict = torch.load(filepath, map_location=self.device)
        if "weights" in weight_dict:
            self.net.load_state_dict(weight_dict["weights"])
        else:
            self.net.load_state_dict(weight_dict)


def init_denoising_autoencoder(**kwargs) -> Tuple[Type[torch.nn.Module], dict]:
    """
    Initialize a denoising autoencoder model
    
    Returns:
        Tuple of (model, meta_state_dict)
    """
    model = DenoisingAutoencoder(**kwargs)
    return model.net, model.meta_state_dict


# Convenience function for quick denoising
def denoise_images(noisy_images: np.ndarray,
                   clean_images: np.ndarray,
                   test_noisy: Optional[np.ndarray] = None,
                   test_clean: Optional[np.ndarray] = None,
                   training_cycles: int = 500,
                   **kwargs) -> Tuple[DenoisingAutoencoder, np.ndarray]:
    """
    Convenience function for training a denoising autoencoder and making predictions
    
    Args:
        noisy_images: Training noisy images
        clean_images: Training clean images
        test_noisy: Test noisy images (optional)
        test_clean: Test clean images (optional)
        training_cycles: Number of training cycles
        **kwargs: Additional arguments for model and training
        
    Returns:
        Tuple of (trained_model, predictions_on_test_data)
    """
    # Initialize model
    model = DenoisingAutoencoder(**kwargs)
    
    # Train model
    model.fit(noisy_images, clean_images, test_noisy, test_clean,
              training_cycles=training_cycles, **kwargs)
    
    # Make predictions if test data provided
    predictions = None
    if test_noisy is not None:
        predictions = model.predict(test_noisy)
    
    return model, predictions
