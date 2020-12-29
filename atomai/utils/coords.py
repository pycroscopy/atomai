"""
coords.py
=========

Module for working with atom/defect/particle coordinates

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Tuple, Optional, Union, List, Dict
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial, ndimage, optimize
from sklearn import cluster
import torch
from .viz import plot_lattice_bonds


def find_com(image_data: np.ndarray) -> np.ndarray:
    """
    Find atoms via center of mass methods

    Args:
        image_data (2D numpy array):
            2D image (usually an output of neural network)
    """
    labels, nlabels = ndimage.label(image_data)
    coordinates = np.array(
        ndimage.center_of_mass(
            image_data, labels, np.arange(nlabels) + 1))
    coordinates = coordinates.reshape(coordinates.shape[0], 2)
    return coordinates


def imcoordgrid(im_dim: Tuple) -> torch.Tensor:
    """
    Returns a grid with pixel coordinates (used e.g. in rVAE)
    """
    xx = torch.linspace(-1, 1, im_dim[0])
    yy = torch.linspace(1, -1, im_dim[1])
    x0, x1 = torch.meshgrid(xx, yy)
    x_coord = torch.stack(
        [x0.T.flatten(), x1.T.flatten()], axis=1)
    return x_coord


def transform_coordinates(coord: Union[np.ndarray, torch.Tensor],
                          phi: float,
                          coord_dx: Union[np.ndarray, torch.Tensor, int] = 0,
                          ) -> torch.Tensor:
    """
    Pytorch-based 2D rotation of coordinates followed by translation.
    Operates on batches.

    Args:
        coord (numpy array or torch tensor): batch with initial coordinates
        phi (float): rotational angle in rad
        coord_dx (numpy array or torch tensor): translation vector

    Returns:
        Transformed coordinates batch
    """

    if isinstance(coord, np.ndarray):
        coord = torch.from_numpy(coord).float()
    if isinstance(coord_dx, np.ndarray):
        coord_dx = torch.from_numpy(coord_dx).float()
    rotmat_r1 = torch.stack([torch.cos(phi), torch.sin(phi)], 1)
    rotmat_r2 = torch.stack([-torch.sin(phi), torch.cos(phi)], 1)
    rotmat = torch.stack([rotmat_r1, rotmat_r2], axis=1)
    coord = torch.bmm(coord, rotmat)

    return coord + coord_dx


def get_nn_distances_(coordinates: np.ndarray, nn: int = 2,
                      upper_bound: Optional[float] = None) -> Tuple[np.ndarray]:
    """
    Calculates nearest-neighbor distances for a single image

    Args:
        coordinates (numpy array):
            :math:`N \\times 3` array with atomic coordinates where first two
            columns are *xy* coordinates and the third column is atom class
        nn (int): Number of nearest neighbors to search for.
        upper_bound (float or int, non-negative):
            Upper distance bound for Query the kd-tree for nearest neighbors.
            Only di
    Returns:
        Tuple with :math:`atoms \\times nn` array of distances to nearest
        neighbors and :math:`atoms \\times (nn+1) \\times 3` array of coordinates
        (including coordinates of the "center" atom), where n_atoms is less or
        equal to the total number of atoms in the 'coordinates'
        (due to 'upper_bound' criterion)
    """
    upper_bound = np.inf if upper_bound is None else upper_bound
    tree = spatial.cKDTree(coordinates[:, :2])
    d, nn = tree.query(
        coordinates[:, :2], k=nn+1, distance_upper_bound=upper_bound)
    idx_to_del = np.where(d == np.inf)[0]
    nn = np.delete(nn, idx_to_del, axis=0)
    d = np.delete(d, idx_to_del, axis=0)
    return d[:, 1:], coordinates[nn]


def get_nn_distances(coordinates: Union[Dict[int, np.ndarray], np.ndarray],
                     nn: int = 2, upper_bound: Optional[float] = None
                     ) -> Tuple[List[np.ndarray]]:
    """
    Calculates nearest-neighbor distances for a stack of images

    Args:
        coordinates:
            Dictionary where keys are frame numbers and values are
            :math:`N \\times 3` numpy arrays with atomic coordinates.
            In each array the first two columns are *xy* coordinates and
            the third column is atom class. One can also pass a single
            numpy array (if all the coordiantes correspond to a single image)
        nn:
            Number of nearest neighbors to search for.
        upper_bound (float or int, non-negative):
            Upper distance bound for Query the kd-tree for nearest neighbors.
            Only distances below this value will be counted.
            
    Returns:
        Tuple with list of :math:`atoms \\times nn` arrays of distances
        to nearest neighbors and list of :math:`atoms \\times (nn+1) \\times 3`
        array of coordinates (including coordinates of the "center" atom),
        where n_atoms is less or equal to the total number of atoms in the
        'coordinates' (due to 'upper_bound' criterion)
    """
    if isinstance(coordinates, np.ndarray):
        coordinates = {0: coordinates}
    distances_all, atom_pairs_all = [], []
    for coord in coordinates.values():
        distances, atom_pairs = get_nn_distances_(coord, nn, upper_bound)
        distances_all.append(distances)
        atom_pairs_all.append(atom_pairs)
    return distances_all, atom_pairs_all


def gaussian_2d(xy: Tuple[np.ndarray], amp: float, xo: float, yo: float,
                sigma_x: float, sigma_y: float, theta: float, offset: float
                ) -> np.ndarray:
    """
    Models 2D Gaussian

    Args:
        xy (tuple): two M x N arrays
        amp (float): peak amplitude
        xo (float): x-coordinate of peak center
        yo (float): y-coordinate of peak center
        sigma_x (float): peak width (x-projection)
        sigma_y (float): peak height (y-projection)
        theta (float): parameter of 2D Gaussian
        offset (float): parameter of 2D Gaussian

    Returns:
        Flattened numpy array
    """
    x, y = xy
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amp*np.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.flatten()


def peak_refinement(imgdata: np.ndarray, coordinates: np.ndarray,
                    d: Optional[int] = None) -> np.ndarray:
    """
    Performs a refinement of atomic postitions by fitting
    2d Gaussian where the neural network predictions serve
    as initial guess.

    Args:
        imgdata (2D numpy array):
            Single experimental image/frame
        coordinates (N x 3 numpy array):
            Atomic coordinates where first two columns are *xy* coordinates
            and the third column is atom class
        d (int):
            Half-side of a square around the identified atom for peak fitting
            If d is not specified, it is set to 1/4 of average nearest neighbor
            distance in the lattice.

    Returns:
        Refined array of coordinates
    """
    if d is None:
        warnings.warn(
            "The d-value for bounding box not found. Defaulting to 1/4 of mean atomic distance.",
            stacklevel=2
        )
        d = get_nn_distances_(coordinates)[0]
        d = np.concatenate((d))
        d = int(np.mean(d)*0.25)
    xyc_all = []
    for i, c in enumerate(coordinates[:, :2]):
        cx = int(np.around(c[0]))
        cy = int(np.around(c[1]))
        img = imgdata[cx-d:cx+d, cy-d:cy+d]
        if img.shape == (int(2*d), int(2*d)):
            e1, e2 = img.shape
            x, y = np.mgrid[:e1:1, :e2:1]
            initial_guess = (img[d, d], d, d, 1, 1, 0, 0)
            try:
                popt, _ = optimize.curve_fit(
                        gaussian_2d, (x, y), img.flatten(), p0=initial_guess)
                if np.linalg.norm(popt[1:3] - d) < 3:
                    xyc = popt[1:3] + np.around(c) - d
                else:
                    xyc = c
            except RuntimeError:
                xyc = c
        else:
            xyc = c
        xyc_all.append(xyc)
    xyc_all = np.concatenate(
        (np.array(xyc_all), coordinates[:, 2:3]), axis=-1)
    return xyc_all


def get_intensities_(coordinates, img, r=3):
    """
    Calculates intensities in a 3x3 square around each predicted position
    for a single image. The size of the square can be adjusted using `r` arg
    """
    intensities_all = []
    for c in coordinates:
        cx = int(np.around(c[0]))
        cy = int(np.around(c[1]))
        if r % 2 != 0:
            img_cr = np.copy(
                img[cx-r//2:cx+r//2+1, cy-r//2:cy+r//2+1])
        else:
            img_cr = np.copy(
                img[cx-r//2:cx+r//2, cy-r//2:cy+r//2])
        intensity = np.mean(img_cr)
        intensities_all.append(intensity)
    intensities_all = np.array(intensities_all)
    return intensities_all


def get_intensities(coordinates_all, nn_input, r=3):
    """
    Calculates intensities in a 3x3 square around each predicted position
    for a stack of images. The size of the square can be adjusted using `r` arg
    """
    intensities_all = []
    for k, coord in coordinates_all.items():
        intensities_all.append(get_intensities_(coord, nn_input[k]))
    return intensities_all


def compare_coordinates(coordinates1: np.ndarray,
                        coordinates2: np.ndarray,
                        d_max: float,
                        plot_results: bool = False,
                        **kwargs: Union[int, np.ndarray]) -> Tuple[np.ndarray]:
    """
    Finds difference between predicted ('coordinates1')
    and "true" ('coordinates2') coordinates using scipy.spatialcKDTree method.
    Use 'd_max' to set maximum search radius. For plotting, pass figure size
    and experimental image using keyword arguments 'fsize' and 'expdata'.
    """
    coordinates1_ = np.empty((0, 3))
    coordinates2_ = np.empty((0, 3))
    delta_r = []
    for c in coordinates1:
        dist, idx = spatial.cKDTree(coordinates2).query(c)
        if dist < d_max:
            coordinates1_ = np.append(coordinates1_, [c], axis=0)
            coordinates2_ = np.append(
                coordinates2_, [coordinates2[idx]], axis=0)
            delta_r.append(dist)
    if plot_results:
        fsize = kwargs.get('fsize', 20)
        expdata = kwargs.get('expdata')
        if expdata is None:
            raise AssertionError(
                "For plotting, provide 2D image via 'expdata' keyword")
        plt.figure(figsize=(int(fsize*1.25), fsize))
        plt.imshow(expdata, cmap='gray')
        im = plt.scatter(
            coordinates1_[:, 1], coordinates1_[:, 0],
            c=np.array(delta_r), cmap='jet', s=5)
        clrbar = plt.colorbar(im)
        clrbar.set_label('Position deviation (px)')
        plt.show()
    return coordinates1_, coordinates2_, np.array(delta_r)


def cluster_coord(coord_class_dict: Dict[int, np.ndarray], eps: float,
                  min_samples: int = 10) -> Tuple[np.ndarray]:
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

        - coordinates of points in each identified cluster
        - center of the mass for each cluster
        - variance of points in each cluster
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


def find_coord_clusters(coord_class_dict_1: Dict[int, np.ndarray],
                        coord_class_dict_2: Dict[int, np.ndarray],
                        rmax: int) -> Tuple[np.ndarray, List]:
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

        - coordinates of points in each identified cluster
        - center of the mass for each cluster
        - standard deviation of points in each cluster
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


class subimg_trajectories:
    """
    Extracts a trajectory of a single defect/atom from image stack
    together with the associated subimages

    Args:
        imgdata (np.ndarray):
            Stack of images (can be raw data or NN output)
        coord_class_dict (dict):
            Dictionary of atomic coordinates
            (same format as produced by atomnet.locator)
        window_size (int):
            size of window for subimage cropping
        min_length (int):
            Minimal length of trajectory to return
        rmax (int):
            Max allowed distance (projected on xy plane) between defect
            in one frame and the position of its nearest neighbor in the next one
    """
    def __init__(self,
                 imgdata: np.ndarray,
                 coord_class_dict: Dict[int, np.ndarray],
                 window_size: int,
                 min_length: int = 0,
                 rmax: int = 10) -> None:
        self.imgdata = imgdata
        self.coord_class_dict = coord_class_dict
        self.r = window_size
        self.min_length = min_length
        self.rmax = rmax

    def get_trajectory(self, img: np.ndarray,
                       start_coord: np.ndarray
                       ) -> Tuple[np.ndarray]:
        """
        Extracts a single trajectory
        """
        def crop_(img_, c_):
            cx = int(np.around(c_[0]))
            cy = int(np.around(c_[1]))
            img_cr = img_[cx-self.r//2:cx+self.r//2, cy-self.r//2:cy+self.r//2]
            return img_cr

        flow, frames, img_cr_all = [], [], []
        c0 = start_coord
        for k, c in self.coord_class_dict.items():
            d, index = spatial.cKDTree(
                c[:, :2]).query(c0, distance_upper_bound=self.rmax)
            if d != np.inf:
                img_cr = crop_(self.imgdata[k], c[index])
                if img_cr.shape[0:2] == (self.r, self.r):
                    flow.append(c[index])
                    img_cr_all.append(img_cr)
                    frames.append(k)
                    c0 = c[index][:2]
        return np.array(flow), np.array(frames), np.array(img_cr_all)

    def get_all_trajectories(self) -> Tuple[List[np.ndarray]]:
        """
        Extracts all trajectories
        """
        trajectories_all, frames_all = [], []
        subimgs_all = []
        for ck in self.coord_class_dict[list(self.coord_class_dict.keys())[0]][:,:2]:
            flow, frames, subimgs = self.get_trajectory(self.coord_class_dict, ck)
            if len(flow) > self.min_length:
                trajectories_all.append(flow)
                frames_all.append(frames)
                subimgs_all.append(subimgs)
        return trajectories_all, frames_all, subimgs_all


def map_bonds(coordinates: Dict[int, np.ndarray],
              nn: int = 2,
              upper_bound: float = None,
              distance_ideal: float = None,
              plot_results: bool = True,
              **kwargs: Union[str, int]) -> np.ndarray:
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
            Upper distance bound (in px) for Query the kd-tree for nearest neighbors.
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


def remove_edge_coord(coordinates: np.ndarray, dim: Tuple,
                      dist_edge: int) -> np.ndarray:
    """
    Removes coordinates at the image edges
    """

    def coord_edges(coordinates, h, w):
        return [coordinates[0] > w - dist_edge,
                coordinates[0] < dist_edge,
                coordinates[1] > h - dist_edge,
                coordinates[1] < dist_edge]

    h, w = dim
    coord_to_rem = [
                    idx for idx, c in enumerate(coordinates)
                    if any(coord_edges(c, h, w))
                    ]
    coord_to_rem = np.array(coord_to_rem, dtype=int)
    coordinates = np.delete(coordinates, coord_to_rem, axis=0)
    return coordinates
