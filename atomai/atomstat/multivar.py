"""
multivar.py
===========

Module for statistical analysis of local image descriptors

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Tuple, List, Dict, Union

import copy
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from sklearn import cluster, decomposition, mixture

from atomai.utils import get_intensities, plot_transitions, extract_subimages


class imlocal:
    """
    Class for extraction and statistical analysis of local image descriptors.
    It assumes that input image data is an output of a neural network, but
    it can also work with regular experimental images
    (make sure you have extra dimensions for channel and batch size).

    Args:
        network_output (4D numpy array):
            Output of a fully convolutional neural network where
            a class is assigned to every pixel in the input image(s).
            The dimensions are
            :math:`images \\times height \\times width \\times channels`
        coord_class_dict_all (dict):
            Prediction from atomnet.locator
            (can be from other source but must be in the same format)
            Each element is a :math:`N \\times 3` numpy array,
            where *N* is a number of detected atoms/defects,
            the first 2 columns are *xy* coordinates
            and the third columns is class (starts with 0)
        crop_size (int):
            Side of the square for subimage cropping
        coord_class (int):
            Class of atoms/defects around around which
            the subimages will be cropped; in the atomnet.locator output
            the class is the 3rd column (the first two are *xy* positions)

    Examples:

    Identification of distortion domains in a single atomic image:

        >>> # First obtain a "cleaned" image and atomic coordinates using a trained model
        >>> nn_input, (nn_output, coordinates) = atomnet.predictor(expdata, model, use_gpu=False).run()
        >>> # Now get local image descriptors using atomstat.imlocal
        >>> imstack = atomstat.imlocal(nn_output, coordinates, crop_size=32, coord_class=1)
        >>> # Compute PCA scree plot to estimate the number of components/sources
        >>> imstack.pca_scree_plot(plot_results=True);
        >>> # Do PCA analysis and plot results
        >>> pca_results = imstack.imblock_pca(n_components=4, plot_results=True)
        >>> # Do NMF analysis and plot results
        >>> pca_results = imstack.imblock_nmf(n_components=4, plot_results=True)

    Analysis of atomic/defect trajectories from movies (3D image stack):

        >>> # Get local descriptors (such as subimages centered around impurities)
        >>> imstack = atomstat.imlocal(nn_output, coordinates, crop_size=32, coord_class=1)
        >>> # Calculate Gaussian mixture model (GMM) components
        >>> components_img, classes_list = imstack.gmm(n_components=10, plot_results=True)
        >>> # Calculate GMM components and transition probabilities for different trajectories
        >>> traj_all, trans_all, fram_all = imstack.transition_matrix(n_components=10, rmax=10)
    """
    def __init__(self,
                 network_output: np.ndarray,
                 coord_class_dict_all: Dict[int, np.ndarray],
                 crop_size: int = None,
                 coord_class: int = 0,
                 window_size: int = None) -> None:
        """
        Initializes parameters and collects a stack of subimages
        for the statistical analysis of local descriptors
        """
        self.network_output = network_output
        self.nb_classes = network_output.shape[-1]
        self.coord_all = coord_class_dict_all
        self.coord_class = np.float(coord_class)
        self.r = crop_size
        if window_size is not None:
            self.r = window_size
        else:
            warnings.warn("The crop_size argument is deprecated. Use window_size to specify size of subimages",
                          UserWarning)
        (self.imgstack,
         self.imgstack_com,
         self.imgstack_frames) = self.extract_subimages_()
        self.d0, self.d1, self.d2, self.d3 = self.imgstack.shape

    def extract_subimages_(self) -> Tuple[np.ndarray]:
        """
        Extracts subimages centered at certain atom class/type
        in the neural network output

        Returns:
            3-element tuple containing

            - stack of subimages
            - (x, y) coordinates of their centers
            - frame number associated with each subimage
        """
        imgstack, imgstack_com, imgstack_frames = extract_subimages(
            self.network_output, self.coord_all, self.r, self.coord_class)
        return imgstack, imgstack_com, imgstack_frames

    def gmm(self,
            n_components: int,
            covariance: str = 'diag',
            random_state: int = 1,
            plot_results: bool = False) -> Tuple[np.ndarray, List]:
        """
        Applies Gaussian mixture model to image stack.

        Args:
            n_components (int):
                Number of components
            covariance (str):
                Type of covariance ('full', 'diag', 'tied', 'spherical')
            random_state (int):
                Random state instance
            plot_results (bool):
                Plotting gmm components

        Returns:
            3-element tuple containing

            - 4D numpy array with GMM "centroids" (averaged images for each class)
            - List where each element contains 4D images belonging to each GMM class
            - 2D numpy array with *xy* coordinates, label and corresponding frame number for each subimage
        """
        clf = mixture.GaussianMixture(
            n_components=n_components,
            covariance_type=covariance,
            random_state=random_state)
        X_vec = self.imgstack.reshape(self.d0, self.d1*self.d2*self.d3)
        classes = clf.fit_predict(X_vec) + 1
        cla = np.ndarray(shape=(
            np.amax(classes), int(self.r), int(self.r), self.nb_classes))
        if plot_results:
            rows = int(np.ceil(float(n_components)/5))
            cols = int(np.ceil(float(np.amax(classes))/rows))
            fig = plt.figure(figsize=(4*cols, 4*(1+rows//2)))
            gs = gridspec.GridSpec(rows, cols)
            print('\nGMM components')
        cl_all = []
        for i in range(np.amax(classes)):
            cl = self.imgstack[classes == i + 1]
            cl_all.append(cl)
            cla[i] = np.mean(cl, axis=0)
            if plot_results:
                ax = fig.add_subplot(gs[i])
                if self.nb_classes == 3:
                    ax.imshow(cla[i], Interpolation='Gaussian')
                elif self.nb_classes == 1:
                    ax.imshow(cla[i, :, :, 0], cmap='seismic',
                              Interpolation='Gaussian')
                else:
                    raise NotImplementedError(
                        "Can plot only images with 3 and 1 channles")
                ax.axis('off')
                ax.set_title('Class '+str(i+1)+'\nCount: '+str(len(cl)))
        if plot_results:
            plt.subplots_adjust(hspace=0.6, wspace=0.4)
            plt.show()
        com_frames = np.concatenate(
            (self.imgstack_com, classes[:, None],
             self.imgstack_frames[:, None]), axis=-1)
        return cla, cl_all, com_frames

    def pca(self,
            n_components: int,
            random_state: int = 1,
            plot_results: bool = False) -> Tuple[np.ndarray]:
        """
        Computes PCA eigenvectors for a stack of subimages.

        Args:
            n_components (int):
                Number of PCA components
            random_state (int):
                Random state instance
            plot_results (bool):
                Plots computed eigenvectors

        Returns:
            3-element tuple containing

            - 4D numpy array with computed and reshaped principal axes
            - 2D numpy with projection of X_vec (vector with flattened subimages) on the first principal components
            - 2D numpy array with center-of-mass coordinates and corresponding frame number for each subimage
        """
        pca = decomposition.PCA(
            n_components=n_components,
            random_state=random_state)
        X_vec = self.imgstack.reshape(self.d0, self.d1*self.d2*self.d3)
        X_vec_t = pca.fit_transform(X_vec)
        components = pca.components_
        components = components.reshape(
            n_components, self.d1, self.d2, self.d3)
        com_frames = np.concatenate(
            (self.imgstack_com, self.imgstack_frames[:, None]), axis=-1)
        if plot_results:
            self.plot_decomposition_results(
                components, X_vec_t, plot_loading_maps=False)
        return components, X_vec_t, com_frames

    def ica(self,
            n_components: int,
            random_state: int = 1,
            plot_results: bool = False) -> Tuple[np.ndarray]:
        """
        Computes ICA independent souces for a stack of subimages.

        Args:
            n_components (int):
                Number of ICA components
            random_state (int):
                Random state instance
            plot_results (bool):
                Plots computed sources

        Returns:
            3-element tuple containing

            - 4D numpy array with computed and reshaped independent sources
            - 2D numpy array with recovered sources from X_vec (vector with flattned subimages)
            - 2D numpy aray with center-of-mass coordinates and corresponding frame number for each subimage
        """
        ica = decomposition.FastICA(
            n_components=n_components,
            random_state=random_state)
        X_vec = self.imgstack.reshape(self.d0, self.d1*self.d2*self.d3)
        X_vec_t = ica.fit_transform(X_vec)
        components = ica.components_
        components = components.reshape(
            n_components, self.d1, self.d2, self.d3)
        com_frames = np.concatenate(
            (self.imgstack_com, self.imgstack_frames[:, None]), axis=-1)
        if plot_results:
            self.plot_decomposition_results(
                components, X_vec_t, plot_loading_maps=False)
        return components, X_vec_t, com_frames

    def nmf(self,
            n_components: int,
            random_state: int = 1,
            plot_results: bool = False,
            **kwargs: int) -> Tuple[np.ndarray]:
        """
        Applies NMF to source separation from a stack of subimages

        Args:
            n_components (int):
                Number of NMF components
            random_state (int):
                Random state instance
            plot_results (bool):
                Plots computed sources
            **max_iterations (int):
                Maximum number of iterations before timing out

        Returns:
            3-element tuple containing

            - 4D numpy array with computed and reshaped sources
            - 2D numpy array with transformed data according to the trained NMF model,
            - 2D numpy aray with center-of-mass coordinates and corresponding frame number for each subimage
        """

        max_iter = kwargs.get('max_iterations', 1000)
        nmf = decomposition.NMF(
            n_components=n_components,
            random_state=random_state,
            max_iter=max_iter)
        X_vec = self.imgstack.reshape(self.d0, self.d1*self.d2*self.d3)
        X_vec_t = nmf.fit_transform(X_vec)
        components = nmf.components_
        components = components.reshape(
            n_components, self.d1, self.d2, self.d3)
        com_frames = np.concatenate(
            (self.imgstack_com, self.imgstack_frames[:, None]), axis=-1)
        if plot_results:
            self.plot_decomposition_results(
                components, X_vec_t, plot_loading_maps=False)
        return components, X_vec_t, com_frames

    def pca_gmm(self,
                n_components_gmm: int,
                n_components_pca: int,
                plot_results: bool = False,
                covariance_type: str = 'diag',
                random_state: int = 1) -> Tuple[np.ndarray, List]:
        """
        Performs PCA decomposition on GMM-unmixed classes. Can be used when
        GMM allows separating different symmetries
        (e.g. different sublattices in graphene)

        Args:
            n_components_gmm (int):
                Number of components for GMM
            n_components_pca (int or list of int):
                Number of PCA components. Pass a list of integers in order
                to have different number PCA of components for each GMM class
            covariance (str):
                Type of covariance ('full', 'diag', 'tied', 'spherical')
            random_state (int):
                Random state instance
            plot_results (bool):
                Plotting GMM components

        Returns:
            4-element tuple containing

            - 4D numpy array with GMM "centroids" (averaged images for each GMM class)
            - List of 4D numpy arrays with PCA components
            - List with PCA-transformed data
            - 2D numpy array with *xy* coordinates, GMM-assigned labels, and corresponding frame numbers
        """
        gmm_components, gmm_imgs, com_class_frames = self.gmm(
            n_components_gmm, covariance_type, random_state, plot_results)
        if isinstance(n_components_pca, np.int):
            n_components_pca = [n_components_pca for _ in range(n_components_gmm)]
        pca_components_all, X_vec_t_all = [], []
        for j, (imgs, ncomp) in enumerate(zip(gmm_imgs, n_components_pca)):
            pca = decomposition.PCA(
                n_components=ncomp, random_state=random_state)
            X_vec_t = pca.fit_transform(
                imgs.reshape(imgs.shape[0], self.d1*self.d2*self.d3))
            pca_components = pca.components_
            pca_components = pca_components.reshape(
                ncomp, self.d1, self.d2, self.d3)
            pca_components_all.append(pca_components)
            X_vec_t_all.append(X_vec_t)
            if plot_results:
                print("\nPCA components for GMM class {}".format(j+1))
                self.plot_decomposition_results(
                    pca_components, X_vec_t, plot_loading_maps=False)
        return gmm_components, pca_components_all, X_vec_t_all, com_class_frames

    def pca_scree_plot(self, plot_results: bool = True) -> np.ndarray:
        """
        Computes and plots PCA 'scree plot'
        (explained variance ratio vs number of components)
        """
        # PCA decomposition
        pca = decomposition.PCA()
        X_vec = self.imgstack.reshape(self.d0, self.d1*self.d2*self.d3)
        pca.fit(X_vec)
        explained_var = pca.explained_variance_ratio_
        if plot_results:
            # Plotting
            _, ax = plt.subplots(1, 1, figsize=(6,6))
            ax.plot(explained_var, '-o')
            ax.set_xlim(-0.5, 50)
            ax.set_xlabel('Number of components')
            ax.set_ylabel('Explained variance')
            plt.show()
        return explained_var

    def pca_gmm_scree_plot(self,
                           n_components_gmm: int,
                           covariance_type: str = 'diag',
                           random_state: int = 1,
                           plot_results: bool = True) -> List[np.ndarray]:
        """
        Computes PCA scree plot for each GMM class

        Args:
            n_components_gmm (int):
                Number of components for GMM
            covariance (str):
                Type of covariance ('full', 'diag', 'tied', 'spherical')
            random_state (int):
                Random state instance
            plot_results (bool):
                Plotting GMM components and PCA scree plot

        Returns:
            List with PCA explained variances for each GMM component
        """
        _, gmm_imgs, _ = self.gmm(
            n_components_gmm, covariance_type, random_state, plot_results)
        explained_var_all = []
        for j, imgs in enumerate(gmm_imgs):
            pca = decomposition.PCA()
            pca.fit(imgs.reshape(imgs.shape[0], self.d1*self.d2*self.d3))
            explained_var = pca.explained_variance_ratio_
            if plot_results:
                print('\nPCA scree plot for GMM component {}'.format(j+1))
                _, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.plot(explained_var, '-o')
                xlim_ = imgs.shape[0] if imgs.shape[0] < 50 else 50
                ax.set_xlim(-0.5, xlim_)
                ax.set_xlabel('Number of components')
                ax.set_ylabel('Explained variance')
                plt.show()
            explained_var_all.append(explained_var)
        return explained_var_all

    def imblock_pca(self,
                    n_components: int,
                    random_state: int = 1,
                    plot_results: bool = False,
                    **kwargs: int) -> Tuple[np.ndarray]:
        """
        Computes PCA eigenvectors and their loading maps
        for a stack of subimages. Intended to be used for
        finding domains ("blocks") (e.g. ferroic domains)
        in a single image.

        Args:
            n_components (int):
                Number of PCA components
            random_state (int):
                Random state instance
            plot_results (bool):
                Plots computed eigenvectors and loading maps
            **marker_size (int):
                Controls marker size for loading maps plot

        Returns:
            3-element tuple containing

            - 4D numpy array with computed (and reshaped) principal axes
            - 2D numpy array with projection of X_vec (vector with flattened subimages) on the first principal components
            - 2D numpy array with coordinates of each subimage
        """

        m_s = kwargs.get('marker_size')
        components, X_vec_t, com_frames = self.pca(n_components, random_state)
        if plot_results:
            if self.network_output.shape[0] != 1:
                raise AssertionError(
                    "The 'mother image' dimensions must be (1 x h x w x c)")
            self.plot_decomposition_results(
                components, X_vec_t,
                self.network_output.shape[1:3],
                com_frames[:, :2], marker_size=m_s)
        return components, X_vec_t, com_frames[:, :2]

    def imblock_ica(self,
                    n_components: int,
                    random_state: int = 1,
                    plot_results: bool = False,
                    **kwargs: int) -> Tuple[np.ndarray]:
        """
        Computes ICA independent souces and their loading maps
        for a stack of subimages. Intended to be used for
        finding domains ("blocks") (e.g. ferroic domains)
        in a single image.

        Args:
            n_components (int):
                Number of ICA components
            random_state (int):
                Random state instance
            plot_results (bool):
                Plots computed eigenvectors and loading maps
            **marker_size (int):
                controls marker size for loading maps plot

        Returns:
            3-element tuple containing

            - 4D numpy array with computed (and reshaped) independent sources
            - 2D numpy array with recovered sources from X_vec (vector with flattened subimages)
            - 2D numpy array with coordinates of each subimage
        """

        m_s = kwargs.get('marker_size')
        components, X_vec_t, com_frames = self.ica(n_components, random_state)
        if plot_results:
            if self.network_output.shape[0] != 1:
                raise AssertionError(
                    "The 'mother image' dimensions must be (1 x h x w x c)")
            self.plot_decomposition_results(
                components, X_vec_t,
                self.network_output.shape[1:3],
                com_frames[:, :2], marker_size=m_s)
        return components, X_vec_t, com_frames[:, :2]

    def imblock_nmf(self,
                    n_components: int,
                    random_state: int = 1,
                    plot_results: bool = False,
                    **kwargs: int) -> Tuple[np.ndarray]:
        """
        Applies NMF to source separation.
        Computes sources and their loading maps
        for a stack of subimages. Intended to be used for
        finding domains ("blocks") (e.g. ferroic domains)
        in a single image.

        Args:
            n_components (int):
                Number of NMF components
            random_state (int):
                Random state instance
            plot_results (bool):
                Plots computed eigenvectors and loading maps
            **max_iterations (int):
                Maximum number of iterations before timing out
            **marker_size (int):
                Controls marker's size for loading maps plots

        Returns:
            3-element tuple containing

            - 4D numpy array with computed (and reshaped) sources
            - 2D numpy array with transformed X_vec (vector with flattened subimages) according to the trained NMF model
            - 2D numpy array with coordinates of each subimage
        """

        m_s = kwargs.get('marker_size')
        components, X_vec_t, com_frames = self.nmf(n_components, random_state)
        if plot_results:
            if self.network_output.shape[0] != 1:
                raise AssertionError(
                    "The 'mother image' dimensions must be (1 x h x w x c)")
            self.plot_decomposition_results(
                components, X_vec_t,
                self.network_output.shape[1:3],
                com_frames[:, :2], marker_size=m_s)
        return components, X_vec_t, com_frames[:, :2]

    @classmethod
    def plot_decomposition_results(cls,
                                   components: np.ndarray,
                                   X_vec_t: np.ndarray,
                                   image_hw: Tuple = None,
                                   xy_centers: np.ndarray = None,
                                   plot_loading_maps: bool = True,
                                   **kwargs: int) -> None:
        """
        Plots decomposition "eigenvectors". Plots loading maps

        Args:
            components (4D numpy array):
                Computed (and reshaped)
                principal axes / independent sources / factorization matrix
                for stack of subimages
            X_vec_t (2D numpy array):
                Projection of X_vec on the first principal components /
                Recovered sources from X_vec /
                transformed X_vec according to the learned NMF model
                (is used to create "loading maps")
            img_hw (tuple):
                Height and width of the "mother image"
            xy_centers (n x 2 numpy array):
                (x, y) coordinates of the extracted subimages
            plot_loading_maps (bool):
                Plots loading maps for each "eigenvector"
            **marker_size (int):
                Controls marker's size for loading maps plots
        """
        nc = components.shape[0]
        rows = int(np.ceil(float(nc)/5))
        cols = int(np.ceil(float(nc)/rows))
        # plot eigenvectors
        gs1 = gridspec.GridSpec(rows, cols)
        fig1 = plt.figure(figsize=(4*cols, 4*(1+rows//2)))
        comp_ = components[..., :-1] if components.shape[-1] > 1 else components
        for i in range(nc):
            ax1 = fig1.add_subplot(gs1[i])
            ax1.imshow(
                np.sum(comp_[i], axis=-1),
                cmap='seismic', Interpolation='Gaussian')
            ax1.set_aspect('equal')
            ax1.axis('off')
            ax1.set_title('Component '+str(i + 1)+'\nComponent')
        plt.show()
        if plot_loading_maps:
            # plot loading maps
            m_s = kwargs.get("marker_size", 32)
            y, x = xy_centers.T
            img_h, img_w = image_hw
            gs2 = gridspec.GridSpec(rows, cols)
            fig2 = plt.figure(figsize=(4*cols, 4*(1+rows//2)))
            for i in range(nc):
                ax2 = fig2.add_subplot(gs2[i])
                ax2.scatter(
                    x, y, c=X_vec_t[:, i],
                    cmap='seismic', marker='s', s=m_s)
                ax2.set_xlim(0, img_w)
                ax2.set_ylim(img_h, 0)
                ax2.set_aspect('equal')
                ax2.axis('off')
                ax2.set_title('Component '+str(i+1)+'\nLoading map')
            plt.show()

    @classmethod
    def get_trajectory(cls,
                       coord_class_dict: Dict[int, np.ndarray],
                       start_coord: np.ndarray,
                       rmax: int) -> Tuple[np.ndarray]:
        """
        Extracts a trajectory of a single defect/atom from image stack

        Args:
            coord_class_dict (dict):
                Dictionary of atomic coordinates
                (same format as produced by atomnet.locator)
            start_coord (N x 2 numpy array):
                Coordinate of defect/atom in the first frame
                whose trajectory we are going to track
            rmax (int):
                Max allowed distance (projected on xy plane) between defect
                in one frame and the position of its nearest neigbor
                in the next one

        Returns:
            2-element tuple containing

            - Numpy array of defect/atom coordinates form a single trajectory
            - Frames corresponding to this trajectory
        """
        flow = np.empty((0, 3))
        frames = []
        c0 = start_coord
        for k, c in coord_class_dict.items():
            d, index = spatial.cKDTree(
                c[:,:2]).query(c0, distance_upper_bound=rmax)
            if d != np.inf:
                flow = np.append(flow, [c[index]], axis=0)
                frames.append(k)
                c0 = c[index][:2]
        return flow, np.array(frames)

    def get_all_trajectories(self,
                             min_length: int = 0,
                             run_gmm: bool = False,
                             rmax: int = 10,
                             **kwargs: Union[int, str]) -> Dict:
        """
        Extracts trajectories for the detected defects
        starting from the first frame. Applies (optionally)
        Gaussian mixture model to a stack of local descriptors (subimages).

        Args:
            min_length (int):
                Minimal length of trajectory to return
            run_gmm (bool):
                Optional GMM separation into different classes
            rmax (int):
                Max allowed distance (projected on xy plane) between defect
                in one frame and the position of its nearest neigbor
                in the next one
            **n_components (int):
                Number of components for  Gaussian mixture model
            **covariance (str):
                Type of covariance for Gaussian mixture model
                ('full', 'diag', 'tied', 'spherical')
            **random_state (int):
                Random state instance for Gaussian mixture model

        Returns:
            Python dictionary containing

            - list of numpy arrays with defects/atoms trajectories ("trajectories")
            - list of frames corresponding to the extracted trajectories ("frames")
            - GMM components when run_gmm=True ("gmm_components")
        """
        if run_gmm:
            n_components = kwargs.get("n_components", 5)
            covariance = kwargs.get("covariance", "diag")
            random_state = kwargs.get("random_state", 1)
            gmm_comps, _, classes = self.gmm(
                n_components, covariance, random_state)
            classes = classes[:, -2]
        else:
            classes = np.zeros(len(self.imgstack_frames))
        coord_class_dict = {
            i : np.concatenate(
                (self.imgstack_com[np.where(self.imgstack_frames == i)[0]],
                    classes[np.where(self.imgstack_frames == i)[0]][..., None]),
                    axis=-1)
            for i in self.imgstack_frames
        }
        all_trajectories = []
        all_frames = []
        for ck in coord_class_dict[list(coord_class_dict.keys())[0]][:, :2]:
            flow, frames = self.get_trajectory(coord_class_dict, ck, rmax)
            if len(flow) > min_length:
                all_trajectories.append(flow)
                all_frames.append(frames)
        return_dict = {"trajectories": all_trajectories,
                       "frames": all_frames}
        if run_gmm:
            return_dict["gmm_components"] = gmm_comps
        return return_dict

    @classmethod
    def renumerate_classes(cls, classes: np.ndarray) -> np.ndarray:
        """
        Helper functions for renumerating Gaussian mixture model
        classes for Markov transition analysis
        """
        diff = np.unique(classes) - np.arange(len(np.unique(classes)))
        diff_d = {cl: d for d, cl in zip(diff, np.unique(classes))}
        classes_renum = [cl - diff_d[cl] for cl in classes]
        return np.array(classes_renum, dtype=np.int64)

    def transition_matrix(self,
                          n_components: int,
                          covariance: str = 'diag',
                          random_state: int = 1,
                          rmax: int = 10,
                          min_length: int = 0,
                          sum_all_transitions: bool = False) -> Dict:
        """
        Applies Gaussian mixture model to a stack of
        local descriptors (subimages). Extracts trajectories for
        the detected defects starting from the first frame.
        Calculates transition probability for each trajectory.

        Args:
            n_components (int):
                Number of components for  Gaussian mixture model
            covariance (str):
                Type of covariance for Gaussian mixture model
                ('full', 'diag', 'tied', 'spherical')
            random_state (int):
                Random state instance for Gaussian mixture model
            rmax (int):
                Max allowed distance (projected on xy plane) between defect
                in one frame and the position of its nearest neigbor
                in the next one
            min_length (int):
                Minimal length of trajectory to return

        Returns:
            Pyhton dictionary containing

            - List of defects/atoms trajectories ("trajectories")
            - List of transition matrices for each trajectory ("transitions")
            - List of frames corresponding to the extracted trajectories ("frames")
            - GMM components as images ("gmm_components")
        """
        dict_to_return = self.get_all_trajectories(
            min_length, run_gmm=True, n_components=n_components, rmax=rmax,
            covariance=covariance, random_state=random_state)
        transitions_all = []
        for traj in dict_to_return["trajectories"]:
            classes = self.renumerate_classes(traj[:, -1])
            m = calculate_transition_matrix(classes)
            transitions_all.append(m)
        dict_to_return["transitions"] = transitions_all
        if sum_all_transitions:
            dict_to_return["all_transitions"] = sum_transitions(
                dict_to_return, n_components)
        return dict_to_return


def calculate_transition_matrix(trace: Union[List, np.ndarray]) -> np.ndarray:
    """
    Calculates Markov transition matrix

    Args:
        trace (1D numpy array or python list):
            sequence of states/classes

    Returns:
        Computed 2D matrix of transition probabilities
    """
    n = 1 + max(trace)  # number of states
    M = np.zeros(shape=(n, n))
    for (i, j) in zip(trace, trace[1:]):
        M[i][j] += 1
    # convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M


def sum_transitions(trans_dict: Dict, msize: int,
                    plot_results: bool = False, **kwargs: int) -> np.ndarray:
    """
    Sums and normalizes transitions associated with individual trajectories

    Args:
        trans_dict (dict):
            Python dictionary containing trajectories, frame numbers,
            transitions and the averaged GMM components. Usually this is
            an output of atomstat.transition_matrix
        msize (int):
            (m, m) size of full transition matrix
        plot_results (bool):
            plot transition matrix and GMM components
            associated with highest transition frequencies
        **transitions_to_plot (int):
            number of transitions (associated with largest prob values) to plot

    Returns:
        Full transition matrix as 2D numpy array
    """
    transmat_all = np.zeros((msize, msize))
    for traj, trans in zip(trans_dict["trajectories"], trans_dict["transitions"]):
        states = np.unique(traj[:, -1]).astype(np.int64)
        for (i, j), v in np.ndenumerate(trans):
            transmat_all[states[i]-1, states[j]-1] += v
    transmat_all = transmat_all/transmat_all.sum(axis=1, keepdims=1)
    if plot_results:
        plot_transitions(
            transmat_all,
            gmm_components=trans_dict["gmm_components"],
            **kwargs)
    return transmat_all


def update_classes(coordinates: Union[Dict[int, np.ndarray], np.ndarray],
                   nn_input: np.ndarray,
                   method: str = 'threshold',
                   **kwargs: float) -> Dict[int, np.ndarray]:
    """
    Updates atomic/defect classes based on the calculated intensities
    at each predicted position or local neighborhood analysis based on
    subimages cropped around each predicted position

    Args:
        coordinates (dict):
            Output of atomnet.predictor. It is also possible to pass a single
            dictionary value associated with a specific image in a stack. In
            this case, the same image needs to be passed as 'nn_input'.
        nn_input (numpy array):
            Image(s) served as an input to neural network
        method (str):
            Method for intensity-based update of atomic classes
            ('threshold', 'kmeans', 'gmm_local')
        **thresh (float or int):
            Intensity threshold value. Values above/below are set to 1/0
        **n_components (int):
            Number of components for k-means clustering

    Returns:
        Updated coordinates
    """
    if isinstance(coordinates, np.ndarray):
        coordinates = {0: coordinates}
    if np.ndim(nn_input) == 2:
        nn_input = nn_input[None, ..., None]
    elif np.ndim(nn_input) == 3 and nn_input.shape[-1] > 10:  # assuming we never have more than 10 classes
        nn_input = nn_input[..., None]
    elif np.ndim(nn_input) == 3 and nn_input.shape[-1] < 10:
        nn_input = nn_input[None, ...]
    coordinates_ = copy.deepcopy(coordinates)
    if method == 'threshold':
        r = kwargs.get("window_size", 3)
        intensities = get_intensities(coordinates_, nn_input, 3)
        intensities_ = np.concatenate(intensities)
        thresh = kwargs.get('thresh')
        if thresh is None:
            raise AttributeError(
                "Specify intensity threshold value ('thresh'), e.g. thresh=.5")
        for i, iarray in enumerate(intensities):
            iarray[iarray < thresh] = 0
            iarray[iarray >= thresh] = 1
            coordinates_[i][:, -1] = iarray
        plt.figure(figsize=(5, 5))
        counts = plt.hist(intensities_, bins=20)[0]
        plt.vlines(thresh, np.min(counts), np.max(counts),
                   linestyle='dashed', color='red', label='threshold')
        plt.legend()
        plt.title('Intensities (arb. units)')
        plt.show()
    elif method == 'kmeans':
        r = kwargs.get("window_size", 3)
        intensities = get_intensities(coordinates_, nn_input, r)
        intensities_ = np.concatenate(intensities)
        n_components = kwargs.get('n_components')
        if n_components is None:
            raise AttributeError(
                "Specify number of components ('n_components')")
        kmeans = cluster.KMeans(
            n_clusters=n_components, random_state=42).fit(intensities_[:, None])
        for i, iarray in enumerate(intensities):
            coordinates_[i][:, -1] = kmeans.predict(iarray[:, None])
    elif method == "meanshift":
        r = kwargs.get("window_size", 3)
        intensities = get_intensities(coordinates_, nn_input, r)
        intensities_ = np.concatenate(intensities)
        bandwidth = cluster.estimate_bandwidth(
            intensities_[:, None], quantile=kwargs.get("q", .25))
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(intensities_[:, None])
        for i, iarray in enumerate(intensities):
            coordinates_[i][:, -1] = ms.predict(iarray[:, None])
    elif method == "gmm_local":
        n_components = kwargs.get('n_components')
        window_size = kwargs.get("window_size")
        coord_class = kwargs.get("coord_class", 0)
        if None in (n_components, window_size):
            raise AttributeError(
                "Specify number of components ('n_components') and window size ('window_size')"
            )
        s = imlocal(nn_input, coordinates_, window_size, coord_class)
        _, _, com_frames = s.gmm(n_components, plot_results=True)
        for i in coordinates_.keys():
            coordinates_[i] = com_frames[com_frames[:, -1] == float(i)][:, :3]
        for i in coordinates_.keys():
            coordinates_[i][:, -1] = coordinates_[i][:, -1] - 1
    else:
        raise NotImplementedError(
            "Choose between 'threshold', 'kmeans', 'meanshift' and 'gmm_local' methods")
    return coordinates_
