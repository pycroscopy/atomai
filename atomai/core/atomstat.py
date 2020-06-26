"""
atomstat.py
===========

Module for statistical analysis of local image descriptors

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import os
import copy
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import spatial
from sklearn import cluster, decomposition, mixture

from atomai.utils import (get_intensities, get_nn_distances, peak_refinement,
                          plot_transitions, extract_subimages)


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
                 network_output,
                 coord_class_dict_all,
                 crop_size,
                 coord_class):
        """
        Initializes parameters and collects a stack of subimages
        for the statistical analysis of local descriptors
        """
        self.network_output = network_output
        self.nb_classes = network_output.shape[-1]
        self.coord_all = coord_class_dict_all
        self.coord_class = np.float(coord_class)
        self.r = crop_size
        (self.imgstack,
         self.imgstack_com,
         self.imgstack_frames) = self.extract_subimages()
        self.d0, self.d1, self.d2, self.d3 = self.imgstack.shape

    def extract_subimages(self):
        """
        Extracts subimages centered at certain atom class/type
        in the neural network output

        Returns:
            3-element tuple containing
            i) stack of subimages,
            ii) (x, y) coordinates of their centers,
            iii) frame number associated with each subimage
        """
        imgstack, imgstack_com, imgstack_frames = extract_subimages(
            self.network_output, self.coord_all, self.r, self.coord_class)
        return imgstack, imgstack_com, imgstack_frames

    def gmm(self,
            n_components,
            covariance='diag',
            random_state=1,
            plot_results=False):
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
            i) 4D numpy array (*n_components* x *height* x *width* x *channels*)
            containing averaged images for each GMM class
            (the 1st dimension correspond to individual mixture components),
            ii) List where each element contains 4D images belonging to each GMM class,
            iii) 2D numpy array (*N* x 4) with *xy* coordinates of the center of mass,
            labels and a frame number for each subimage
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
            n_components,
            random_state=1,
            plot_results=False):
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
            i) 4D numpy array (*n_components* x *height* x *width* x *channels*)
            with computed (and reshaped) principal axes for stack of subimages,
            ii) 2D numpy array (*N* x *n_components*) with projection of X_vec
            on the first principal components,
            iii) 2D numpy array (*N* x 3) with center-of-mass coordinates
            and the corresponding frame number for each subimage
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
            n_components,
            random_state=1,
            plot_results=False):
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
            i) 4D numpy array (*n_components* x *height* x *width* x *channels*)
            with computed (and reshaped) independent sources
            for stack of subimages,
            ii) 2D numpy array (*N* x *n_components*) numpy array with
            recovered sources from X_vec,
            iii) 2D numpy aray (*N* x 3) with center-of-mass coordinates
            and the corresponding frame number for each subimage
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
            n_components,
            random_state=1,
            plot_results=False,
            **kwargs):
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
            i) 4D numpy array (*n_components* x *height* x *width* x *channels*)
            with computed (and reshaped) sources for stack of subimages,
            ii) 2D numpy array (*N* x *n_components*)  with
            transformed data X_vec according to the trained NMF model,
            iii) 2D numpy array(*N* x 3) with center-of-mass coordinates
            and the corresponding frame number for each subimage
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
                n_components_gmm,
                n_components_pca,
                plot_results=False,
                covariance_type='diag',
                random_state=1):
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
            i) 4D numpy array (*n_components_gmm* x *height* x *width* x *channels*)
            containing averaged images for each gmm class,
            ii) List of 4D numpy arrays with PCA components
            (*n_components_pca* x *height* x *width* x *channels*),
            iii) List of PCA-transformed data,
            iv) 2D numpy array (*N* x 4) with *xy* coordinates of the center of mass
            for each subimage from the stack used for GMM, GMM-assigned label
            for every subimage and a frame number for each label
        """
        gmm_components, gmm_imgs, com_class_frames = self.gmm(
            n_components_gmm, covariance_type, random_state, plot_results)
        if type(n_components_pca) == np.int:
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

    def pca_scree_plot(self, plot_results=True):
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
                           n_components_gmm,
                           covariance_type='diag',
                           random_state=1,
                           plot_results=True):
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
            List of PCA explained variance
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
                    n_components,
                    random_state=1,
                    plot_results=False,
                    **kwargs):
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
            2-element tuple containing
            i) 4D numpy array (*n_components* x *height* x *width* x *channels*)
            with computed (and reshaped) principal axes
            for stack of subimages and ii) 2D numpy array (*N* x *n_components*)
            with projection of X_vec on the first principal components
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
                    n_components,
                    random_state=1,
                    plot_results=False,
                    **kwargs):
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
            2-element tuple containing
            i) 4D numpy array (*n_components* x *height* x *width* x *channels*)
            with computed (and reshaped) independent sources
            for stack of subimages and ii) 2D numpy array (*N* x *n_components*)
            with recovered sources from X_vec
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
                    n_components,
                    random_state=1,
                    plot_results=False,
                    **kwargs):
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
                Controls marker size for loading maps plot

        Returns:
            2-element tuple containing
            i) 4D numpy array (*n_components* x *height* x *width* x *channels*)
            with computed (and reshaped) sources
            for stack of subimages and ii) 2D numpy array (*N* x *n_components*)
            with transformed data X_vec according to the trained NMF model
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
                                   components,
                                   X_vec_t,
                                   image_hw=None,
                                   xy_centers=None,
                                   plot_loading_maps=True,
                                   **kwargs):
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
                Controls marker size for loading maps plot
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
    def get_trajectory(cls, coord_class_dict, start_coord, rmax):
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
            i) Numpy array of defect/atom coordinates form a single trajectory
            and ii) frames corresponding to this trajectory
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
                             min_length=0,
                             run_gmm=False,
                             rmax=10,
                             **kwargs):
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
            i) list of numpy arrays with defects/atoms trajectories ("trajectories"),
            ii) list of frames corresponding to the extracted trajectories ("frames"),
            iii) gmm components when run_gmm=True ("gmm_components")
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
    def renumerate_classes(cls, classes):
        """
        Helper functions for renumerating Gaussian mixture model
        classes for Markov transition analysis
        """
        diff = np.unique(classes) - np.arange(len(np.unique(classes)))
        diff_d = {cl: d for d, cl in zip(diff, np.unique(classes))}
        classes_renum = [cl - diff_d[cl] for cl in classes]
        return np.array(classes_renum, dtype=np.int64)

    def transition_matrix(self,
                          n_components,
                          covariance='diag',
                          random_state=1,
                          rmax=10,
                          min_length=0,
                          sum_all_transitions=False):
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
            i) list of defects/atoms trajectories ("trajectories"),
            ii) list of transition matrices for each trajectory ("transitions"),
            iii) list of frames corresponding to the extracted trajectories ("frames"),
            iv) GMM components as images ("gmm_components")
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


def calculate_transition_matrix(trace):
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


def sum_transitions(trans_dict, msize, plot_results=False, **kwargs):
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
            number of transitions (associated with largerst prob values) to plot
    
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


def plot_lattice_bonds(distances,
                       atom_pairs,
                       distance_ideal=None,
                       frame=0,
                       display_results=True,
                       **kwargs):
    """
    Plots a map of lattice bonds

    Args:
        distances (numpy array):
            :math:`n_atoms \\times nn` array,
            where *nn* is a number of nearest neighbors
        atom_pairs (numpy array):
            :math:`n_atoms \\times (nn+1) \\times 3`,
            where *nn* is a number of nearest neighbors
        distance_ideal (float):
            Bond distance in ideal lattice.
            Defaults to average distance in the frame
        frame (int):
            frame number (used in filename when saving plot)
        display_results (bool):
            Plot bond maps
        **savedir (str):
            directory to save plots
        **h (int):
            image height
        **w (int):
            image width
    """
    savedir = kwargs.get("savedir", './')
    h, w = kwargs.get("h"), kwargs.get("w")
    if h is None or w is None:
        w = int(np.amax(atom_pairs[..., 0]) - np.amin(atom_pairs[..., 0])) + 10
        h = int(np.amax(atom_pairs[..., 1]) - np.amin(atom_pairs[..., 1])) + 10
    if w != h:
        warnings.warn("Currently supports only square images", UserWarning)
    if distance_ideal is None:
        distance_ideal = np.mean(distances)
    distances = (distances - distance_ideal) / distance_ideal
    d_uniq = np.sort(np.unique(distances))
    colormap = cm.RdYlGn_r
    colorst = [colormap(i) for i in np.linspace(0, 1, d_uniq.shape[0])]
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    ax1.imshow(np.zeros((h, w)), cmap='gray')
    for a, d in zip(atom_pairs, distances):
        for i in range(a.shape[-1]):
            x = [a[0][0], a[i+1][0]]
            y = [a[0][1], a[i+1][1]]
            color = colorst[np.where(d[i] == d_uniq)[0][0]]
            ax1.plot(x, y, c=color)
    ax1.axis(False)
    ax1.set_aspect('auto')
    clrbar = np.linspace(np.amin(d_uniq), np.amax(d_uniq), d_uniq.shape[0]-1).reshape(-1, 1)
    ax2 = fig.add_axes([0.11, 0.08, .8, .2])
    img = ax2.imshow(clrbar, colormap)
    plt.gca().set_visible(False)
    clrbar_ = plt.colorbar(img, ax=ax2, orientation='horizontal')
    clrbar_.set_label('Variation in bond length (%)', fontsize=14, labelpad=10)
    if display_results:
        plt.show()
    fig.savefig(os.path.join(savedir, 'frame_{}'.format(frame)))


def map_bonds(coordinates,
              nn=2,
              upper_bound=None,
              distance_ideal=None,
              plot_results=True,
              **kwargs):
    """
    Generates plots with lattice bonds
    (color-coded according to the variation in their length)

    Args:
        coordinates (dict):
            Dictionary where keys are frame numbers and values are
            :math:`N \\times 3` numpy arrays with atomic coordinates.
            In each array the first two columns are *xy* coordinates and
            the third column is atom class.
        nn (int): Number of nearest neighbors to search for.
        upper_bound (float or int, non-negative):
            Upper distance bound for Query the kd-tree for nearest neighbors.
            Only distances below this value will be counted.
        distance_ideal (float):
            Bond distance in ideal lattice.
            Defaults to average distance in the frame
        plot_results (bool):
            Plot bond maps
        **savedir (str):
            directory to save plots
        **h (int):
            image height
        **w (int):
            image width

    Returns:
        Array of distances to nearest neighbors for each atom
    """
    distances_all, atom_pairs_all = get_nn_distances(coordinates, nn, upper_bound)
    if distance_ideal is None:
        distance_ideal = np.mean(np.concatenate((distances_all)))
    for i, (dist, at) in enumerate(zip(distances_all, atom_pairs_all)):
        plot_lattice_bonds(dist, at, distance_ideal, i, plot_results, **kwargs)
    return np.concatenate((distances_all))


def cluster_coord(coord_class_dict, eps, min_samples=10):
    """
    Collapses coordinates from an image stack onto xy plane and
    performs clustering in the xy space. Works for non-overlapping
    trajectories.

    Args:
        coord_class_dict (dict):
            Dictionary of atomic coordinates (:math:`N \\times 3` numpy arrays])
            (same format as produced by atomnet.locator)
            Can also be a list of :math:`N \\times 3` numpy arrays
            Typically, these are coordinates from a 3D image stack
            where each element in dict/list corresponds
            to an individual movie frame
        eps (float):
            Max distance between two points for one to be considered
            as in the neighborhood of the other
            (see sklearn.cluster.DBSCAN).
        min_samples (int):
            Minmum number of points for a "cluster"

    Returns:
        3-element tuple containing
        i) coordinates of points in each identified cluster,
        ii) center of the mass for each cluster,
        iii) variance of points in each cluster
    """
    coordinates_all = np.empty((0, 3))
    for k in range(len(coord_class_dict)):
        coordinates_all = np.append(
            coordinates_all, coord_class_dict[k], axis=0)
    clustering = cluster.DBSCAN(
        eps=eps, min_samples=min_samples).fit(coordinates_all[:, :2])
    labels = clustering.labels_
    clusters, clusters_var, clusters_mean = [], [], []
    for l in np.unique(labels)[1:]:
        coord = coordinates_all[np.where(labels == l)]
        clusters.append(coord)
        clusters_mean.append(np.mean(coord[:, :2], axis=0))
        clusters_var.append(np.var(coord[:, :2], axis=0))
    return (np.array(clusters), np.array(clusters_mean),
            np.array(clusters_var))


def find_coord_clusters(coord_class_dict_1, coord_class_dict_2, rmax):
    """
    Takes a single array of xy coordinates (usually associated
    with a single image) and for each coordinate finds
    its nearest neighbors (within specified radius) from a separate list of
    arrays with xy coordinates (where each element in the list usually
    corresponds to a single image from an image stack). Works for
    non-overlapping trajectories in atomic movies.

    Args:
        coord_class_dict_1 (dict ot list):
            One-element dictionary or list with atomic coordinates
            as N x 3 numpy array.
            (usually from an output of atomnet.predictor for a single image;
            can be from other source but should be in the same format)
        coord_class_dict_2 (dict or list):
            Dictionary or list of atomic coordinates
            (:math:`N \\times 3` numpy arrays)
            These can be coordinates from a 3D image stack
            where each element in dict/list corresponds
            to an individual frame in the stack.
            (usually from an output from atomnet.locator for an image stack;
            can be from other source but should be in the same format)
        rmax (int):
            Maximum search radius in pixels

    Returns:
        3-element tuple containing
        i) coordinates of points in each identified cluster,
        ii) center of the mass for each cluster,
        iii) standard deviation of points in each cluster
    """
    coordinates_all = np.empty((0, 3))
    for k in range(len(coord_class_dict_2)):
        coordinates_all = np.append(
            coordinates_all, coord_class_dict_2[k], axis=0)

    clusters, clusters_mean, clusters_std = [], [], []
    tree = spatial.cKDTree(coordinates_all[:, :2])
    for c0 in coord_class_dict_1[0][:, :2]:
        _, idx = tree.query(
            c0, k=len(coordinates_all), distance_upper_bound=rmax)
        idx = np.delete(idx, np.where(idx == len(coordinates_all))[0])
        cluster_coord = coordinates_all[idx]
        clusters_mean.append(np.mean(cluster_coord[:, :2], axis=0))
        clusters_std.append(np.std(cluster_coord[:, :2], axis=0))
        clusters.append(cluster_coord)
    return (np.array(clusters_mean), np.array(clusters_std), clusters)


def update_classes(coordinates,
                   nn_input,
                   method='threshold',
                   **kwargs):
    """
    Updates atomic/defect classes based on the calculated intensities
    at each predicted position

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
    coordinates_ = copy.deepcopy(coordinates)
    if method == 'threshold':
        intensities = get_intensities(coordinates_, nn_input)
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
        intensities = get_intensities(coordinates_, nn_input)
        intensities_ = np.concatenate(intensities)
        n_components = kwargs.get('n_components')
        if n_components is None:
            raise AttributeError(
                "Specify number of components ('n_components')")
        kmeans = cluster.KMeans(
            n_clusters=n_components, random_state=42).fit(intensities_[:, None])
        for i, iarray in enumerate(intensities):
            coordinates_[i][:, -1] = kmeans.predict(iarray[:, None])
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
        for i in range(len(coordinates_)):
            coordinates_[i] = com_frames[:, :3]
    else:
        raise NotImplementedError(
            "Choose between 'threshold', 'kmeans', and 'gmm_local' methods")
    return coordinates_


def update_positions(coordinates, nn_input, d=None):
    """
    Updates atomic/defect coordinates based on
    peak refinement procedure at each predicted position

    Args:
        coordinates (dict or ndarray):
            Dictionary with coordinates (output of atomnet.predictor).
            Can be also a single N x 3 ndarray.
        nn_input (numpy array):
            Image(s) served as an input to neural network
        d (int):
            Half of the side of the square box where the fitting is performed;
            defaults to 1/4 of mean nearest neighbor distance in the system

    Returns:
        Updated coordinates
    """
    if isinstance(coordinates, np.ndarray):
        coordinates = {0: coordinates}
    if np.ndim(nn_input) == 2:
        nn_input = nn_input[None, ..., None]
    print('\rRefining atomic positions... ', end="")
    coordinates_r = {}
    for i, (img, coord) in enumerate(zip(nn_input, coordinates.values())):
        coordinates_r[i] = peak_refinement(img[..., 0], coord, d)
    print("Done")
    return coordinates_r
