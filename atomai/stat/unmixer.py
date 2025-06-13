import warnings
from sklearn.decomposition import NMF, PCA, FastICA
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt


class SpectralUnmixer:
    """
    Applies various decomposition algorithms to hyperspectral data for 
    spectral unmixing and component analysis.
    
    Supported methods: 'nmf', 'pca', 'ica', 'gmm'.
    """
    def __init__(self, method: str = 'nmf', n_components: int = 4, normalize: bool = False, **kwargs):
        """
        Initializes the unmixer.

        Args:
            method (str): The decomposition method to use. 
                          Options: 'nmf', 'pca', 'ica', 'gmm'.
            n_components (int): The number of components to find.
            normalize (bool): If True, each spectrum is L1-normalized (sums to 1)
                              before decomposition. This is highly recommended for NMF.
            **kwargs: Additional keyword arguments to pass to the
                      underlying sklearn model (e.g., max_iter, pca_dims).
        """
        self.method = method
        self.n_components = n_components
        self.normalize = normalize
        self.kwargs = kwargs
        
        if self.method == 'nmf':
            self.model = NMF(n_components=n_components, **self.kwargs)
        elif self.method == 'pca':
            self.model = PCA(n_components=n_components, **self.kwargs)
        elif self.method == 'ica':
            self.model = FastICA(n_components=n_components, whiten='unit-variance', max_iter=self.kwargs.get("max_iter", 200))
        elif self.method == 'gmm':
            self.model = GaussianMixture(n_components=n_components, **self.kwargs)
        else:
            raise ValueError("Method not recognized. Choose from 'nmf', 'pca', 'ica', 'gmm'.")
        self.components_ = None
        self.abundance_maps_ = None
        self.image_shape_ = None


    def fit(self, hspy_data: np.ndarray):
        """
        Fits the selected model to a hyperspectral data cube.
        """
        if hspy_data.ndim != 3:
            raise ValueError("Input data must be a 3D hyperspectral cube (h, w, e).")
        
        self.image_shape_ = hspy_data.shape[:2]
        h, w, e = hspy_data.shape
        spectra_matrix = hspy_data.reshape((h * w, e))
        
        # Data to be passed to the fitting algorithm
        spectra_to_fit = spectra_matrix.copy()

        # Optional per-spectrum L1 normalization
        if self.normalize:
            print("Normalizing each spectrum to sum to 1 (L1 norm)...")
            # Store norms for later rescaling of abundances
            l1_norms = np.sum(spectra_matrix, axis=1, keepdims=True)
            # Avoid division by zero for empty spectra (e.g., from outside scan region)
            l1_norms[l1_norms == 0] = 1
            spectra_to_fit = spectra_matrix / l1_norms

        print(f"Fitting data with {self.method.upper()}...")

        # NMF non-negativity check
        if self.method == 'nmf':
            min_val = np.min(spectra_to_fit)
            if min_val < 0:
                warnings.warn(f"NMF requires non-negative data. Shifting data by {-min_val:.2f}.")
                spectra_to_fit = spectra_to_fit - min_val

        # GMM's PCA+GMM robust workflow
        if self.method == 'gmm':
            # PCA dimension selection
            pca_param = self.kwargs.get('pca_dims', 0.99) # Default to 99% variance
            
            print("Applying PCA for dimensionality reduction before GMM...")
            # First, fit PCA on all components to check variance
            pca_full = PCA()
            pca_full.fit(spectra_to_fit)
            
            if isinstance(pca_param, int):
                n_components_pca = pca_param
                print(f"Using a fixed number of {n_components_pca} principal components.")
            elif isinstance(pca_param, float) and 0 < pca_param < 1:
                cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
                # Find the number of components needed to reach the threshold
                n_components_pca = np.searchsorted(cumulative_variance, pca_param) + 1
                explained_var_actual = cumulative_variance[n_components_pca - 1]
                print(
                    f"Found {n_components_pca} components that explain {explained_var_actual:.1%} "
                    f"of the variance (threshold was {pca_param:.1%})."
                )
            else:
                raise ValueError("pca_dims' must be an int or a float between 0 and 1.")

            # Perform the final PCA transformation with the optimal number of components
            pca_final = PCA(n_components=n_components_pca)
            projected_data = pca_final.fit_transform(spectra_to_fit)
            
            # Fit GMM on the low-dimensional data
            self.model.fit(projected_data)
            labels = self.model.predict(projected_data)
            abundances_unscaled = self.model.predict_proba(projected_data)
            self.components_ = np.array([
                spectra_matrix[labels == i].mean(axis=0) 
                for i in range(self.n_components)
            ])
        else: # For NMF, PCA, ICA
            abundances_unscaled = self.model.fit_transform(spectra_to_fit)
            self.components_ = self.model.components_

        # Rescale abundances if data was normalized
        if self.normalize:
            abundances = abundances_unscaled * l1_norms
        else:
            abundances = abundances_unscaled
            
        # Reshape abundance maps back to image dimensions
        self.abundance_maps_ = abundances.reshape((h, w, self.n_components))
        
        print("Fit complete.")
        return self.components_, self.abundance_maps_

    def plot_results(self, x_axis_vals=None, x_axis_units=None, **kwargs):
        if self.components_ is None:
            print("You must run .fit() first.")
            return

        cmap = 'seismic'
        cmap = kwargs.get("cmap", cmap)
        
        n_cols = self.n_components
        fig, axes = plt.subplots(2, n_cols, figsize=kwargs.get("figsize", (n_cols * 3.5, 6)))

        for i in range(self.n_components):
            # Plot component spectrum
            xaxis = x_axis_vals if x_axis_vals is not None else np.arange(0, self.components_.shape[-1])
            axes[0, i].plot(xaxis, self.components_[i, :])
            axes[0, i].set_title(f'{self.method.upper()} Component {i+1}')
            axes[0, i].set_xlabel(x_axis_units if x_axis_units is not None else 'Energy Bin')
            if i == 0:
                axes[0, i].set_ylabel('Intensity')

            # Plot abundance map
            ax_map = axes[1, i]
            im = ax_map.imshow(self.abundance_maps_[..., i], cmap=cmap)
            ax_map.set_title(f'Abundance Map {i+1}')
            ax_map.axis('off')
            fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
            
        plt.tight_layout()
        plt.show()