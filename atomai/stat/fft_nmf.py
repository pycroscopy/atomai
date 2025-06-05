import numpy as np
from scipy import fftpack
from scipy import ndimage
from sklearn.decomposition import NMF
from skimage.util import view_as_windows
from skimage import io, color
import os

from ..utils import load_image


class SlidingFFTNMF:
    
    def __init__(self, window_size_x=None, window_size_y=None, 
                 window_step_x=None, window_step_y=None,
                 interpolation_factor=2, zoom_factor=2, 
                 hamming_filter=True, components=4):
        '''Sliding Window FFT with NMF unmixing.
        This class calculates the FFT window transform
        and unmixes the output using NMF
        
        Parameters:
        -----------
        window_size_x, window_size_y : int, optional
            Window dimensions. If None, will be auto-calculated based on image size
        window_step_x, window_step_y : int, optional
            Step size for sliding windows. If None, will be auto-calculated as window_size // 4
        '''

        # Store user-provided values (None means auto-calculate)
        self._user_window_size_x = window_size_x
        self._user_window_size_y = window_size_y
        self._user_window_step_x = window_step_x
        self._user_window_step_y = window_step_y
        
        # These will be set in _calculate_window_params or use defaults
        self.window_size_x = window_size_x or 64  # Default fallback
        self.window_size_y = window_size_y or 64
        self.window_step_x = window_step_x or 16
        self.window_step_y = window_step_y or 16
        
        self.interpol_factor = interpolation_factor
        self.zoom_factor = zoom_factor
        self.hamming_filter = hamming_filter
        self.components = components
        
        # Will be initialized when window sizes are determined
        self.hamming_window = None
        
    def _calculate_window_params(self, image_shape):
        """Calculate optimal window and step sizes based on image dimensions"""
        
        height, width = image_shape[:2]
        
        # Auto-calculate window sizes if not provided
        if self._user_window_size_x is None:
            # Use a fraction of image height, with reasonable bounds
            self.window_size_x = max(32, min(128, height // 8))
            # Ensure it's a power of 2 for efficient FFT (optional but recommended)
            self.window_size_x = 2 ** int(np.log2(self.window_size_x))
            print(f"Auto-calculated window_size_x: {self.window_size_x}")
        else:
            self.window_size_x = self._user_window_size_x
            
        if self._user_window_size_y is None:
            # Use a fraction of image width, with reasonable bounds
            self.window_size_y = max(32, min(128, width // 8))
            # Ensure it's a power of 2 for efficient FFT
            self.window_size_y = 2 ** int(np.log2(self.window_size_y))
            print(f"Auto-calculated window_size_y: {self.window_size_y}")
        else:
            self.window_size_y = self._user_window_size_y
            
        # Auto-calculate step sizes if not provided (typically 1/4 of window size for good overlap)
        if self._user_window_step_x is None:
            self.window_step_x = max(1, self.window_size_x // 4)
            print(f"Auto-calculated window_step_x: {self.window_step_x}")
        else:
            self.window_step_x = self._user_window_step_x
            
        if self._user_window_step_y is None:
            self.window_step_y = max(1, self.window_size_y // 4)
            print(f"Auto-calculated window_step_y: {self.window_step_y}")
        else:
            self.window_step_y = self._user_window_step_y
            
        # Validate that windows will fit in the image
        if self.window_size_x > height:
            print(f"Warning: window_size_x ({self.window_size_x}) > image height ({height}). Adjusting...")
            self.window_size_x = min(64, height)
            self.window_step_x = max(1, self.window_size_x // 4)
            
        if self.window_size_y > width:
            print(f"Warning: window_size_y ({self.window_size_y}) > image width ({width}). Adjusting...")
            self.window_size_y = min(64, width)
            self.window_step_y = max(1, self.window_size_y // 4)
            
        # Calculate expected number of windows
        n_windows_x = max(1, (height - self.window_size_x) // self.window_step_x + 1)
        n_windows_y = max(1, (width - self.window_size_y) // self.window_step_y + 1)
        total_windows = n_windows_x * n_windows_y
        
        print(f"Window configuration: {self.window_size_x}×{self.window_size_y}, step: {self.window_step_x}×{self.window_step_y}")
        print(f"Expected {n_windows_x}×{n_windows_y} = {total_windows} windows")
        
        # Initialize hamming window now that we know the sizes
        bw2d = np.outer(np.hamming(self.window_size_x), np.ones(self.window_size_y))
        self.hamming_window = np.sqrt(bw2d * bw2d.T)
        
    def make_windows(self, image):
        """Generate windows from an image using efficient striding operations"""
        
        # Handle color images by converting to grayscale
        if len(image.shape) > 2:
            # Convert RGB to grayscale 
            if image.shape[2] >= 3:
                image = color.rgb2gray(image[:,:,:3])  # Handle RGBA images
            else:
                image = np.mean(image, axis=2)  # Simple average for other formats
        
        # Calculate window parameters based on image size
        self._calculate_window_params(image.shape)
        
        # Ensure image is float type and normalize to 0-1
        image = image.astype(float)
        if np.max(image) > 0:  # Avoid division by zero
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Check if image is big enough for windowing
        if image.shape[0] < self.window_size_x or image.shape[1] < self.window_size_y:
            raise ValueError(f"Image dimensions {image.shape} are smaller than window size ({self.window_size_x}, {self.window_size_y})")
        
        # Pad image if necessary to ensure we can extract at least one window
        pad_x = max(0, self.window_size_x - image.shape[0])
        pad_y = max(0, self.window_size_y - image.shape[1])
        if pad_x > 0 or pad_y > 0:
            image = np.pad(image, ((0, pad_x), (0, pad_y)), mode='constant')
            print(f"Image padded to size {image.shape}")
        
        # Define window parameters
        window_size = (self.window_size_x, self.window_size_y)
        window_step = (self.window_step_x, self.window_step_y)
        
        # Use view_as_windows to efficiently create sliding windows
        windows = view_as_windows(image, window_size, step=window_step)
        
        # Store window shape information for later visualization
        self.windows_shape = (windows.shape[0], windows.shape[1])
        print(f"Created {self.windows_shape[0]}×{self.windows_shape[1]} = {windows.shape[0] * windows.shape[1]} windows")
        
        # Create position vectors for visualization
        x_positions = np.arange(0, windows.shape[1] * window_step[1], window_step[1])
        y_positions = np.arange(0, windows.shape[0] * window_step[0], window_step[0])
        xx, yy = np.meshgrid(x_positions, y_positions)
        self.pos_vec = np.column_stack((yy.flatten(), xx.flatten()))
        
        # Reshape to the expected output format
        return windows.reshape(-1, window_size[0], window_size[1])

    def process_fft(self, windows):
        """Perform FFT on each window with optional hamming filter and zooming"""
        
        num_windows = windows.shape[0]
        fft_results = []
        
        for i in range(num_windows):
            img_window = windows[i].copy()  # Make a copy to avoid modifying original
            
            # Apply Hamming filter if requested
            if self.hamming_filter:
                img_window = img_window * self.hamming_window
                
            # Compute 2D FFT and shift for visualization
            fft_result = fftpack.fftshift(fftpack.fft2(img_window))
            
            # Take the magnitude of the complex FFT result (ensures non-negative values)
            fft_mag = np.abs(fft_result)
            
            # Apply log transform to enhance visibility of lower amplitude frequencies
            fft_mag = np.log1p(fft_mag)  # log(1+x) avoids log(0) issues
            
            # Zoom in on center region
            center_x, center_y = self.window_size_x // 2, self.window_size_y // 2
            zoom_size = max(1, self.window_size_x // (2 * self.zoom_factor))  # Ensure minimum size of 1
            
            # Extract center region, with boundary checking
            x_min = max(0, center_x - zoom_size)
            x_max = min(fft_mag.shape[0], center_x + zoom_size)
            y_min = max(0, center_y - zoom_size)
            y_max = min(fft_mag.shape[1], center_y + zoom_size)
            
            zoomed = fft_mag[x_min:x_max, y_min:y_max]
            
            # Apply interpolation if the interpol factor is greater than 1
            if self.interpol_factor > 1:
                try:
                    final_fft = ndimage.zoom(zoomed, self.interpol_factor, order=1)
                except:
                    print(f"Warning: Interpolation failed for window {i}, using original")
                    final_fft = zoomed
            else:
                final_fft = zoomed
            
            fft_results.append(final_fft)
        
        # Ensure all results have the same shape by padding if necessary
        shapes = [result.shape for result in fft_results]
        max_shape = tuple(max(s[i] for s in shapes) for i in range(2))
        
        for i, result in enumerate(fft_results):
            if result.shape != max_shape:
                padded = np.zeros(max_shape)
                padded[:result.shape[0], :result.shape[1]] = result
                fft_results[i] = padded
        
        self.fft_size = max_shape
        result_array = np.array(fft_results)
        
        # Final check for NaN or Inf values
        result_array = np.nan_to_num(result_array)
        
        return result_array
    
    def run_nmf(self, fft_results):
        """Run NMF on FFT results to extract components"""
        
        # Reshape for NMF
        fft_flat = fft_results.reshape(fft_results.shape[0], -1)
        
        # Ensure all values are non-negative
        fft_flat = np.maximum(0, fft_flat)  # Hard clip any negatives to zero
        
        # Check if we have valid data
        if np.all(fft_flat == 0) or np.isnan(fft_flat).any() or np.isinf(fft_flat).any():
            raise ValueError("Invalid data for NMF: contains zeros, NaNs or Infs")
        
        # Check if we have enough windows
        if fft_flat.shape[0] < self.components:
            print(f"Warning: Number of windows ({fft_flat.shape[0]}) is less than components ({self.components})")
            self.components = min(fft_flat.shape[0], 3)  # Reduce components to avoid error
            print(f"Reducing components to {self.components}")
            
        nmf = NMF(
            n_components=self.components, 
            init='random', 
            random_state=42, 
            max_iter=1000,
            tol=1e-4,
            solver='cd'  # Coordinate descent is typically more robust
        )
        abundances = nmf.fit_transform(fft_flat)
        components = nmf.components_
       
        # Reshape components and abundances for visualization
        try:
            components = components.reshape(self.components, self.fft_size[0], self.fft_size[1])
            abundances = abundances.reshape(self.windows_shape[0], self.windows_shape[1], self.components)
        except Exception as e:
            print(f"Error reshaping results: {e}")
            # Try to reshape in a more flexible way
            components_flat = components.copy()
            components = np.zeros((self.components, self.fft_size[0], self.fft_size[1]))
            for i in range(self.components):
                flat_size = min(components_flat[i].size, self.fft_size[0] * self.fft_size[1])
                components[i].flat[:flat_size] = components_flat[i][:flat_size]
            
            abundances = np.zeros((self.windows_shape[0], self.windows_shape[1], self.components))
            for i in range(min(abundances.shape[2], self.components)):
                abundances[:,:,i] = abundances.reshape(-1, self.components)[:,i].reshape(self.windows_shape)
        
        return components, abundances
    

    def analyze_image(self, image_input, output_path=None):
        """Full analysis pipeline for an image
        
        Parameters:
        -----------
        image_input : str or numpy.ndarray
            Either a file path to an image or a numpy array containing image data
        output_path : str, optional
            Path for saving output files. If None, will be auto-generated for file inputs
            or use current directory for array inputs
        """
        
        # Handle different input types
        if isinstance(image_input, str):
            # File path provided
            self.image_path = image_input
            print(f"Reading image: {image_input}")
            image = load_image(image_input)
            
            # Auto-generate output path if not provided
            if output_path is None:
                base_dir = os.path.dirname(image_input)
                base_name = os.path.splitext(os.path.basename(image_input))[0]
                output_path = os.path.join(base_dir, f"{base_name}_analysis")
                
        elif isinstance(image_input, np.ndarray):
            # Numpy array provided
            self.image_path = "numpy_array_input"
            print("Processing numpy array input")
            image = image_input.copy()  # Make a copy to avoid modifying original
            
            # Auto-generate output path if not provided
            if output_path is None:
                output_path = "array_analysis"
                
        else:
            raise TypeError("image_input must be either a file path (string) or numpy array")
        
        print("Creating windows...")
        windows = self.make_windows(image)
        
        print("Computing FFTs...")
        fft_results = self.process_fft(windows)
        
        print("Running NMF analysis...")
        components, abundances = self.run_nmf(fft_results)

        print("Saving NumPy arrays...")
        np.save(f"{output_path}_components.npy", components)
        np.save(f"{output_path}_abundances.npy", abundances.transpose(-1, 0, 1))

        abundances = abundances.transpose(-1, 0, 1) # (n_components, h, w)
        
        return components, abundances