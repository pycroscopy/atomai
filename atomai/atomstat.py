"""
atomstat.py
===========

Module for statistical analysis of local image descriptors

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import numpy as np
from sklearn import mixture, decomposition, cluster
from scipy import spatial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class imlocal:
    """
    Class for extraction and statistical analysis of local image descriptors.
    It assumes that input image data is an output of a neural network, but
    it can also work with regular experimental images
    (make sure you have extra dimensions for channel and batch size).

    Args:
        network_output (4D numpy array):
            Batch_size x height x width x channels
        coord_class_dict_all (dict):
            Prediction from atomnet.locator
            (can be from other but must be the same format)
            Each element is a N x 3 numpy array,
            where N is a number of detected atoms/defects,
            the first 2 columns are xy coordinates
            and the third columns is class (starts with 0)
        crop_size (int):
            Half of the side of the square for subimage cropping
        coord_class (int):
            Class of atoms/defects around around which
            the subimages will be cropped; in the atomnet.locator output
            the class is the 3rd column (the first two are xy positions)

    Examples:

    Identification of distortion domains in a single atomic image:

        >>> # First obtain a "cleaned" image and atomic coordinates using a trained model
        >>> nn_input, (nn_output, coordinates) = atomnet.predictor(expdata, model, use_gpu=False).run()
        >>> # Now get local image descriptors using atomstat.imlocal
        >>> imstack = atomstat.imlocal(nn_output, coordinates, crop_size=16, coord_class=1)
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
        self.coord_class = coord_class
        self.r = crop_size
        (self.imgstack,
         self.imgstack_com,
         self.imgstack_frames) = self.extract_subimages()
        self.d0, self.d1, self.d2, self.d3 = self.imgstack.shape
        self.X_vec = self.imgstack.reshape(self.d0, self.d1*self.d2*self.d3)

    def extract_subimages(self):
        """
        Extracts subimages centered at certain atom class/type
        in the neural network output

        Args:
            imgdata (4D numpy array):
                Prediction of a neural network with dimensions
                (batch_size x height x width x channels)
            coord (N x 3 numpy array):
                (x, y, class) coordinates data
            d (int):
                Defines size of a square subimage

        Returns:
            stack of subimages,
            (x, y) coordinates of their centers,
            frame number associated with each subimage
        """
        imgstack, imgstack_com, imgstack_frames = [], [], []
        for i, (img, coord) in enumerate(
                zip(self.network_output, self.coord_all.values())):
            c = coord[np.where(coord[:,2]==self.coord_class)][:,0:2]
            img_cr_all, com = self._extract_subimages(img, c, self.r)
            imgstack.append(img_cr_all)
            imgstack_com.append(com)
            imgstack_frames.append(np.ones(len(com), int) * i)
        imgstack = np.concatenate(imgstack, axis=0)
        imgstack_com = np.concatenate(imgstack_com, axis=0)
        imgstack_frames = np.concatenate(imgstack_frames, axis=0)
        return imgstack, imgstack_com, imgstack_frames

    @classmethod
    def _extract_subimages(cls, imgdata, coord, r):
        """
        Extracts subimages centered at specified coordinates
        for a single image

        Args:
            imgdata (3D numpy array):
                Prediction of a neural network with dimensions
                (height x width x channels)
            coord (N x 2 numpy array):
                (x, y) coordinates
            r (int):
                Square subimage side is 2*r

        Returns:
            stack of subimages and
            (x, y) coordinates of their centers
        """
        img_cr_all = []
        com = []
        for c in coord:
            cx = int(np.around(c[0]))
            cy = int(np.around(c[1]))
            img_cr = np.copy(
                imgdata[cx-r:cx+r, cy-r:cy+r, :])
            if img_cr.shape[0:2] == (int(r*2), int(r*2)):
                img_cr_all.append(img_cr[None, ...])
                #com.append(np.array([cx, cy])[None, ...])
                com.append(c[None, ...])

        img_cr_all = np.concatenate(img_cr_all, axis=0)
        com = np.concatenate(com, axis=0)
        return img_cr_all, com

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
            3D numpy array containing averaged images for each gmm class
            (the 1st dimension correspond to individual mixture components)
            and 1D numpy array with labels for every subimage
            in the input image stack.
        """
        clf = mixture.GaussianMixture(
            n_components=n_components,
            covariance_type=covariance,
            random_state=random_state)
        classes = clf.fit_predict(self.X_vec) + 1
        cla = np.ndarray(shape=(
            np.amax(classes), int(self.r*2), int(self.r*2), self.nb_classes))
        if plot_results:
            rows = int(np.ceil(float(n_components)/5))
            cols = int(np.ceil(float(np.amax(classes))/rows))
            fig = plt.figure(figsize=(4*cols, 4*(1+rows//2)))
            gs = gridspec.GridSpec(rows, cols)
            print('GMM components')
        for i in range(np.amax(classes)):
            cl = self.imgstack[classes == i + 1]
            cla[i] = np.mean(cl, axis=0)
            if plot_results:
                ax = fig.add_subplot(gs[i])
                if self.nb_classes == 3:
                    ax.imshow(cla[i], Interpolation='Gaussian')
                elif self.nb_classes == 1:
                    ax.imshow(cla[i, :, :, 0], Interpolation='Gaussian')
                else:
                    raise NotImplementedError(
                        "Can plot only images with 3 and 1 channles")
                ax.axis('off')
                ax.set_title('Class '+str(i+1)+'\nCount: '+str(len(cl)))
        if plot_results:
            plt.subplots_adjust(hspace=0.6, wspace=0.4)
            plt.show()
        return cla, classes

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
            Numpy array of defect/atom coordinaes form a single trajectory
            and frames corresponding to this trajectory
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

    @classmethod
    def cluster_coord(cls, coord_class_dict, eps, min_samples=10):
        """
        Collapses coordinates from an image stack onto xy plane and
        performs clustering in the xy space. Works for non-overlapping
        trajectories.

        Args:
            coord_class_dict (dict):
                Dictionary of atomic coordinates (N x 3 numpy arrays])
                (same format as produced by atomnet.locator)
                Can also be a list of N x 3 numpy arrays
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
            Coordinates of points in each identified cluster,
            center of the mass for each cluster,
            standard deviation of points in each cluster
        """
        coordinates_all = np.empty((0, 3))
        for k in range(len(coord_class_dict)):
            coordinates_all = np.append(
                coordinates_all, coord_class_dict[k], axis=0)
        clustering = cluster.DBSCAN(
            eps=eps, min_samples=min_samples).fit(coordinates_all[:, :2])
        labels = clustering.labels_
        clusters, clusters_std, clusters_mean = [], [], []
        for l in np.unique(labels)[1:]:
            coord = coordinates_all[np.where(labels == l)]
            clusters.append(coord)
            clusters_mean.append(np.mean(coord[:, :2], axis=0))
            clusters_std.append(np.std(coord[:, :2], axis=0))
        return (np.array(clusters), np.array(clusters_mean),
                np.array(clusters_std))

    @classmethod
    def find_coord_clusters(cls, coord_class_dict_1, coord_class_dict_2, rmax):
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
                (usually an output from atomnet.locator for a single image;
                can be from other source but should be in the same format)
            coord_class_dict_2 (dict or list):
                Dictionary or list of atomic coordinates (N x 3 numpy arrays)
                These can be coordinates from a 3D image stack
                where each element in dict/list corresponds
                to an individual frame in the stack.
                (usually an output from atomnet.locator for an image stack;
                can be from other source but should be in the same format)
            rmax (int):
                Maximum search radius in pixels

        Returns:
            Coordinates of points in each identified cluster,
            center of the mass for each cluster,
            standard deviation of points in each cluster
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
            list defects/atoms trajectories (each trajectory is numpy array),
            list of frames corresponding to the extracted trajectories
        """
        if run_gmm:
            n_components = kwargs.get("n_components", 5)
            covariance = kwargs.get("covariance", "diag")
            random_state = kwargs.get("random_state", 1)
            classes = self.gmm(
                n_components, covariance, random_state)[1]
        else:
            classes = np.zeros(len(self.imgstack_frames))
        coord_class_dict = {
            i : np.concatenate(
                (self.imgstack_com[np.where(self.imgstack_frames == i)[0]],
                    classes[np.where(self.imgstack_frames == i)[0]][..., None]),
                    axis=-1)
            for i in range(int(np.ptp(self.imgstack_frames)+1))
        }
        all_trajectories = []
        all_frames = []
        for ck in coord_class_dict[0][:, :2]:
            flow, frames = self.get_trajectory(coord_class_dict, ck, rmax)
            if len(flow) > min_length:
                all_trajectories.append(flow)
                all_frames.append(frames)
        return all_trajectories, all_frames

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
                          min_length=0):
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
            list defects/atoms trajectories,
            list of transition matrices for each trajectory,
            list of frames corresponding to the extracted trajectories
        """
        trajectories_all, frames_all = self.get_all_trajectories(
            min_length, run_gmm=True, n_components=n_components,
            covariance=covariance, random_state=random_state, rmax=rmax)
        transitions_all = []
        for traj in trajectories_all:
            classes = self.renumerate_classes(traj[:, -1])
            m = transitions(classes).calculate_transition_matrix()
            transitions_all.append(m)
        return trajectories_all, transitions_all, frames_all

    def pca_scree_plot(self, plot_results=True):
        """
        Calculates and plots PCA 'scree plot'
        (explained variance ratio vs number of components)
        """
        # PCA decomposition
        pca = decomposition.PCA()
        pca.fit(self.X_vec)
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

    @classmethod
    def plot_decomposition_results(cls,
                                   components,
                                   X_vec_t,
                                   image_hw,
                                   xy_centers,
                                   **kwargs):
        """
        Plots decomposition "eigenvectors" and their loading maps

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
            com_ (n x 2 numpy array):
                (x, y) coordinates of the extracted subimages
            **marker_size (int):
                Controls marker size for loading maps plot
        """
        m_s = kwargs.get("marker_size", 32)
        com_ = xy_centers
        nc = components.shape[0]
        y, x = com_.T
        img_h, img_w = image_hw
        rows = int(np.ceil(float(nc)/5))
        cols = int(np.ceil(float(nc)/rows))
        y, x = com_.T
        print('NUMBER OF COMPONENTS: ' + str(nc))
        # plot eigenvectors
        gs1 = gridspec.GridSpec(rows, cols)
        fig1 = plt.figure(figsize=(4*cols, 4*(1+rows//2)))
        for i in range(nc):
            ax1 = fig1.add_subplot(gs1[i])
            ax1.imshow(
                np.sum(components[i, :, :, :-1], axis=-1),
                cmap='seismic', Interpolation='Gaussian')
            ax1.set_aspect('equal')
            ax1.axis('off')
            ax1.set_title('Component '+str(i + 1)+'\nComponent')
        plt.show()
        # plot loading maps
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

    def imblock_pca(self,
                    n_components,
                    random_state=1,
                    plot_results=True,
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
            4D numpy array with computed (and reshaped) principal axes
            for stack of subimages and 2D numpy array with projection of X_vec
            on the first principal components
        """

        m_s = kwargs.get('marker_size')
        pca = decomposition.PCA(
            n_components=n_components,
            random_state=random_state)
        X_vec_t = pca.fit_transform(self.X_vec)
        components = pca.components_
        components = components.reshape(
            n_components, self.d1, self.d2, self.d3)
        if plot_results:
            assert self.network_output.shape[0] == 1,\
            "The 'mother image' dimensions must be (1 x h x w x c)"
            self.plot_decomposition_results(
                components, X_vec_t,
                self.network_output.shape[1:3],
                self.imgstack_com, marker_size=m_s)
        return components, X_vec_t

    def imblock_ica(self,
                    n_components,
                    random_state=1,
                    plot_results=True,
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
            4D numpy array with computed (and reshaped) independent sources
            for stack of subimages and 2D numpy array with recovered sources
            from X_vec
        """

        m_s = kwargs.get('marker_size')
        ica = decomposition.FastICA(
            n_components=n_components,
            random_state=random_state)
        X_vec_t = ica.fit_transform(self.X_vec)
        components = ica.components_
        components = components.reshape(
            n_components, self.d1, self.d2, self.d3)
        if plot_results:
            assert self.network_output.shape[0] == 1,\
            "The 'mother image' dimensions must be (1 x h x w x c)"
            self.plot_decomposition_results(
                components, X_vec_t,
                self.network_output.shape[1:3],
                self.imgstack_com, marker_size=m_s)
        return components, X_vec_t

    def imblock_nmf(self,
                    n_components,
                    random_state=1,
                    plot_results=True,
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
            4D numpy array with computed (and reshaped) sources
            for stack of subimages and 2D numpy array with
            transformed data X_vec according to the trained NMF model
        """

        m_s = kwargs.get('marker_size')
        max_iter = kwargs.get('max_iterations', 1000)
        nmf = decomposition.NMF(
            n_components=n_components,
            random_state=random_state,
            max_iter=max_iter)
        X_vec_t = nmf.fit_transform(self.X_vec)
        components = nmf.components_
        components = components.reshape(
            n_components, self.d1, self.d2, self.d3)
        if plot_results:
            if self.network_output.shape[0] != 1:
                raise AssertionError(
                    "The 'mother image' dimensions must be (1 x h x w x c)")
            self.plot_decomposition_results(
                components, X_vec_t,
                self.network_output.shape[1:3],
                self.imgstack_com, marker_size=m_s)
        return components, X_vec_t


class transitions:
    """
    Calculates and displays (optionally) Markov transition matrix

    Args:
        trace (1D numpy array or python list):
            sequence of states/classes
    """
    def __init__(self, trace):
        self.trace = trace

    def calculate_transition_matrix(self,
                                    plot_results=False,
                                    plot_values=False):
        """
        Calculates Markov transition matrix

        Args:
            plot_results (bool):
                Plot calculated transition matrix
            plot_values (bool):
                Show calculated transition rates

        Returns:
            Calculated transition matrix as 2D numpy array
        """
        n = 1 + max(self.trace) # number of states
        M = np.zeros(shape=(n, n))
        for (i, j) in zip(self.trace, self.trace[1:]):
            M[i][j] += 1
        # now convert to probabilities:
        for row in M:
            s = sum(row)
            if s > 0:
                row[:] = [f/s for f in row]
        if plot_results:
            self.plot_transition_matrix(M, plot_values)
        return M

    @classmethod
    def plot_transition_matrix(cls, m, plot_values=False):
        """
        Plots transition matrix

        Args:
            m (2D numpy array):
                Transition matrix
            plot_values (bool):
                Show calculated transtion rates
        """
        print('Transition matrix')
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.matshow(m, cmap='Reds')
        xt = np.arange(len(m))
        yt = np.arange(len(m))
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.set_xticklabels((xt+1).tolist(), rotation='horizontal', fontsize=14)
        ax.set_yticklabels((yt+1).tolist(), rotation='horizontal', fontsize=14)
        ax.set_title('Transition matrix', y=1.1, fontsize=20)
        for (j, i), v in np.ndenumerate(m):
            ax.text(i, j, np.around(v, 2), ha='center', va='center', c='b')
        ax.set_xlabel('Transition class', fontsize=18)
        ax.set_ylabel('Starting class', fontsize=18)
        plt.show()

