import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import urllib.request


class ParticleAnalyzer:
    """
    A class to encapsulate an end-to-end particle segmentation and analysis
    workflow using the Segment Anything Model (SAM).

    This class handles:
    - Automatic downloading of SAM model checkpoints.
    - Image pre-processing, including normalization and optional contrast enhancement.
    - Running SAM with preset or custom parameters.
    - Advanced post-processing to filter masks by area and shape, and to remove duplicates.
    - Extraction of detailed properties for each detected particle.
    - Conversion of results to a pandas DataFrame and visualization of results.

    Example:
        >>> # 1. Initialize the analyzer (downloads model if needed)
        >>> analyzer = ParticleAnalyzer(model_type="vit_h")
        >>>
        >>> # 2. Load image and run the analysis
        >>> image = np.load(IMAGE_PATH)
        >>> result = analyzer.analyze(image)
        >>>
        >>> # 3. Print summary and visualize results
        >>> print(f"Found {result['total_count']} particles.")
        >>> df = ParticleAnalyzer.particles_to_dataframe(result)
        >>> print(df.head())
        >>>
        >>> # This will generate and show a side-by-side plot
        >>> ParticleAnalyzer.visualize_particles(
        ...     result,
        ...     original_image_for_plot=image,
        ...     show_plot=True
        ... )
    """
    def __init__(self, checkpoint_path=None, model_type="vit_h", device="auto"):
        """
        Initializes the ParticleAnalyzer by loading the SAM model.
        If the model checkpoint is not found, it will be downloaded automatically.
        
        Args:
            checkpoint_path (str, optional): Path to the SAM model checkpoint file. 
                                             If None, a default path will be used.
            model_type (str): The type of SAM model (e.g., "vit_h", "vit_l", "vit_b").
            device (str): The device to run the model on ("auto", "cuda", "cpu").
        """
        print("Initializing Particle Analyzer...")
        self.device = self._get_device(device)
        
        # Determine the final checkpoint path and download if necessary
        final_checkpoint_path = self._download_model_if_needed(checkpoint_path, model_type)
        
        self.sam_model = self._load_model(final_checkpoint_path, model_type)
        print(f"SAM model loaded successfully on device: {self.device}")

    def _get_device(self, device):
        """Determines the appropriate device for PyTorch."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _download_model_if_needed(self, checkpoint_path, model_type):
        """Checks for the model checkpoint and downloads it if it doesn't exist."""
        model_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        
        if checkpoint_path is None:
            # Create a default path if none is provided
            checkpoint_dir = "./checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"sam_{model_type}.pth")
            
        if not os.path.exists(checkpoint_path):
            url = model_urls.get(model_type)
            if url is None:
                raise ValueError(f"Unknown model type: '{model_type}'. Cannot download.")
            
            print(f"SAM model checkpoint not found at '{checkpoint_path}'.")
            print(f"Downloading model for '{model_type}' from {url}...")
            
            urllib.request.urlretrieve(url, checkpoint_path)
            print(f"Download complete. Model saved to '{checkpoint_path}'.")
            
        return checkpoint_path

    def _load_model(self, checkpoint_path, model_type):
        """Loads the SAM model from a checkpoint and moves it to the device."""
        try:
            from segment_anything import sam_model_registry
        except ImportError:
            raise ImportError(
                "The 'segment-anything' package is required to use this feature.\n"
                "Please install it directly from the official repository:\n\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
            
        try:
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            return sam
        except Exception as e:
            print(f"Error loading SAM model from '{checkpoint_path}': {e}")
            raise

    def analyze(self, image_array, params=None):
        """
        Runs the full analysis pipeline on a given image using a set of parameters.

        Args:
            image_array (np.array): The input 2D grayscale image.
            params (dict, optional): A dictionary of parameters controlling the analysis.
                                     If None, a set of default parameters will be used.
        """
        # If no parameters are provided, use a default set for baseline analysis.
        if params is None:
            print("No parameters provided. Using default analysis settings.")
            params = {
                "use_clahe": False,
                "sam_parameters": "default",
                "min_area": 500,
                "max_area": 50000,
                "use_pruning": False,
                "pruning_iou_threshold": 0.5
            }
            
        # 1. Pre-process the image
        processed_image = self._preprocess_image(image_array, params.get("use_clahe", False))
        image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)

        # 2. Generate masks with SAM using specified parameters
        all_masks = self._run_sam(image_rgb, params.get("sam_parameters", "default"))
        print(f"Generated {len(all_masks)} raw masks.")

        # 3. Filter and prune masks
        final_masks_info = self._filter_and_prune(all_masks, params)
        print(f"Kept {len(final_masks_info)} masks after filtering and pruning.")

        # 4. Extract properties from final masks
        particles = []
        for i, mask in enumerate(final_masks_info):
            particle_info = self._extract_particle_properties(mask, processed_image, i + 1)
            particles.append(particle_info)
        
        # Sort by area for consistent ordering
        particles = sorted(particles, key=lambda x: x['area'], reverse=True)
        # Reassign IDs after sorting
        for i, particle in enumerate(particles):
            particle['id'] = i + 1

        return {
            'particles': particles,
            'original_image': processed_image,
            'rgb_image': image_rgb,
            'total_count': len(particles)
        }

    def _preprocess_image(self, image_array, use_clahe):
        """Normalizes image to uint8 and optionally applies CLAHE."""
        # Normalize to uint8
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0 and image_array.min() >= 0.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                min_val, max_val = image_array.min(), image_array.max()
                if max_val > min_val:
                    image_array = ((image_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    image_array = np.zeros_like(image_array, dtype=np.uint8)
        
        # Apply CLAHE if requested
        if use_clahe:
            print("Applying CLAHE...")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_array = clahe.apply(image_array)
        
        return image_array

    def _run_sam(self, image_rgb, preset_name):
        """Initializes and runs the SAM mask generator based on a preset."""
        try:
            from segment_anything import SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "The 'segment-anything' package is required to use this feature.\n"
                "Please install it directly from the official repository:\n\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        sam_param_presets = {
            "default": {},
            "sensitive": {
                "points_per_side": 96,
                "pred_iou_thresh": 0.80,
                "stability_score_thresh": 0.85,
            },
            "ultra-permissive": {
                "points_per_side": 96,
                "pred_iou_thresh": 0.60,
                "stability_score_thresh": 0.70,
            }
        }
        
        sam_params = sam_param_presets.get(preset_name, {})
        print(f"Running SAM with preset: '{preset_name}'")
        
        mask_generator = SamAutomaticMaskGenerator(self.sam_model, **sam_params)
        return mask_generator.generate(image_rgb)

    def _filter_and_prune(self, masks, params):
        """Applies area filtering and optional shape-based pruning."""
        min_area = params.get("min_area", 0)
        max_area = params.get("max_area", float('inf'))
        
        # Area filtering
        area_filtered_masks = [m for m in masks if min_area <= m['area'] <= max_area]
        
        if params.get("use_pruning", False):
            print("Applying shape-based pruning...")
            iou_threshold = params.get("pruning_iou_threshold", 0.5)
            return self._prune_by_shape_and_iou(area_filtered_masks, iou_threshold)
        else:
            return area_filtered_masks

    def _extract_particle_properties(self, mask, image, particle_id):
        """Extracts detailed properties for a single particle mask."""
        binary_mask = mask['segmentation']
        area = mask['area']
        
        y_coords, x_coords = np.where(binary_mask)
        centroid = (np.mean(x_coords), np.mean(y_coords))
        
        particle_pixels = image[binary_mask]
        
        perimeter = self._calculate_perimeter(binary_mask)

        return {
            'id': particle_id,
            'area': area,
            'centroid': centroid,
            'bbox': mask['bbox'],
            'mean_intensity': np.mean(particle_pixels),
            'std_intensity': np.std(particle_pixels),
            'min_intensity': np.min(particle_pixels),
            'max_intensity': np.max(particle_pixels),
            'perimeter': perimeter,
            'circularity': 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0,
            'equiv_diameter': 2 * np.sqrt(area / np.pi),
            'aspect_ratio': mask['bbox'][3] / mask['bbox'][2] if mask['bbox'][2] > 0 else 1,
            'solidity': mask.get('solidity', self._calculate_solidity(mask)), # Use pre-calculated solidity if available
            'mask': binary_mask
        }

    def _calculate_perimeter(self, binary_mask):
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cv2.arcLength(contours[0], True) if contours else 0

    def _calculate_solidity(self, mask):
        binary_mask = mask['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return 0
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        return area / hull_area if hull_area > 0 else 0

    def _calculate_iou(self, mask1, mask2):
        bbox1, bbox2 = mask1['bbox'], mask2['bbox']
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y_bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        if x_right < x_left or y_bottom < y_top: return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1, area2 = bbox1[2] * bbox1[3], bbox2[2] * bbox2[3]
        union_area = area1 + area2 - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    def _prune_by_shape_and_iou(self, masks, iou_threshold):
        """Prunes masks based on a goodness score and IoU."""
        if not masks: return []

        for m in masks:
            m['solidity'] = self._calculate_solidity(m)
            m['score'] = m['area'] * (m['solidity'] ** 2)

        sorted_masks = sorted(masks, key=lambda x: x['score'], reverse=True)
        
        pruned_masks = []
        for mask in sorted_masks:
            is_duplicate = any(self._calculate_iou(mask, kept_mask) > iou_threshold for kept_mask in pruned_masks)
            if not is_duplicate:
                pruned_masks.append(mask)
        return pruned_masks

    @staticmethod
    def particles_to_dataframe(result):
        """Converts the 'particles' list from the result into a pandas DataFrame."""
        particles = result.get('particles', [])
        if not particles: return pd.DataFrame()
        
        data = []
        for p in particles:
            row = {k: v for k, v in p.items() if k != 'mask'}
            row['centroid_x'], row['centroid_y'] = p['centroid']
            row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height'] = p['bbox']
            del row['centroid'], row['bbox']
            data.append(row)
        return pd.DataFrame(data)

    @staticmethod
    def visualize_particles(result, original_image_for_plot=None, show_plot=False, show_labels=True, show_centroids=True):
        """
        Creates an RGB image visualizing the detected particles and optionally displays a plot.
        
        Args:
            result (dict): The output dictionary from the analyze method.
            original_image_for_plot (np.array, optional): The raw, unprocessed image for side-by-side comparison.
                                                          If None, the processed image from the result is used.
            show_plot (bool): If True, displays a matplotlib plot comparing original and segmented images.
            show_labels (bool): If True, shows particle ID labels on the overlay.
            show_centroids (bool): If True, shows particle centroids on the overlay.
            
        Returns:
            np.array: The RGB overlay image with particles drawn on it.
        """
        overlay = result['rgb_image'].copy()
        for particle in result.get('particles', []):
            contours, _ = cv2.findContours(particle['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
            
            cx, cy = int(particle['centroid'][0]), int(particle['centroid'][1])
            if show_centroids:
                cv2.circle(overlay, (cx, cy), 5, (0, 255, 0), -1)
            if show_labels:
                cv2.putText(overlay, str(particle['id']), (cx + 5, cy + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if show_plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Use the provided original image for the 'before' plot, otherwise use the processed one from results
            display_image = original_image_for_plot if original_image_for_plot is not None else result['original_image']
            
            axes[0].imshow(display_image, cmap='gray')
            axes[0].set_title('Original Input')
            axes[1].imshow(overlay)
            axes[1].set_title(f"Detected Particles (n={result['total_count']})")
            for ax in axes:
                ax.set_axis_off()
            plt.tight_layout()
            plt.show()
            
        return overlay
