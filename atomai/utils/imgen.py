"""
imgen.py
========

Utility functions for generating training images

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""


from typing import Tuple, Callable, Union, List, Dict
import numpy as np


class MakeAtom:
    """
    Creates an image of atom modelled as
    2D Gaussian and a corresponding mask
    """
    def __init__(self, sc: int = 5, r_mask: int = 3,
                 intensity: int = 1, theta: int = 0, offset: int = 0):
        """
        Args:
            sc (int): scale parameter, which determines Gaussian width
            r_mask (int): radius of mask corresponding to atom
            theta (int): parameter of 2D gaussian function
            offset (int): parameter of 2D gaussian function
        """
        if sc % 2 == 0:
            sc += 1
        self.xo, self.yo = sc/2, sc/2
        x = np.linspace(0, sc, sc)
        y = np.linspace(0, sc, sc)
        self.x, self.y = np.meshgrid(x, y)
        self.sigma_x, self.sigma_y = sc/4, sc/4
        self.intensity = intensity
        self.theta = theta
        self.offset = offset
        self.r_mask = r_mask

    def atom2dgaussian(self) -> np.ndarray:
        """
        Models atom as 2d Gaussian
        """
        a = (np.cos(self.theta)**2)/(2*self.sigma_x**2) +\
            (np.sin(self.theta)**2)/(2*self.sigma_y**2)
        b = -(np.sin(2*self.theta))/(4*self.sigma_x**2) +\
             (np.sin(2*self.theta))/(4*self.sigma_y**2)
        c = (np.sin(self.theta)**2)/(2*self.sigma_x**2) +\
            (np.cos(self.theta)**2)/(2*self.sigma_y**2)
        g = self.offset + self.intensity*np.exp(
            -(a*((self.x-self.xo)**2) + 2*b*(self.x-self.xo)*(self.y-self.yo) +\
            c*((self.y-self.yo)**2)))
        return g

    def circularmask(self, image: np.ndarray, radius: int) -> np.ndarray:
        """
        Returns a mask with specified radius
        """
        h, w = self.x.shape
        X, Y = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X-self.xo+0.5)**2 + (Y-self.yo+0.5)**2)
        mask = dist_from_center <= radius
        image[~mask] = 0
        return image

    def gen_atom_mask(self) -> Tuple[np.ndarray]:
        """
        Creates a mask for specific type of atom
        """
        atom = self.atom2dgaussian()
        mask = self.circularmask(atom.copy(), self.r_mask/2)
        mask = mask[np.min(np.where(mask > 0)[0]):
                    np.max(np.where(mask > 0)[0]+1),
                    np.min(np.where(mask > 0)[1]):
                    np.max(np.where(mask > 0)[1])+1]
        mask[mask > 0] = 1

        return atom, mask


def create_lattice_mask(lattice: np.ndarray, xy_atoms: np.ndarray,
                        *args: Callable[[int, int], Tuple[np.ndarray, np.ndarray]],
                        **kwargs: int) -> np.ndarray:
    """
    Given experimental image and *xy* atomic coordinates
    creates ground truth image. Currently works only for the case
    where all atoms are one class. Notice that it will round fractional pixels.

    Args:
        lattice (2D numpy array):
            Experimental image as 2D numpy array
        xy_atoms (N x 2 numpy array):
            Position of atoms in the experimental data
        *arg (python function):
            Function that creates a 2D numpy array with atom and
            corresponding mask for each atomic coordinate. It must have
            two parameters, 'scale' and 'rmask' that control sizes of simulated
            atom and corresponding mask

            Example:

            >>> def create_atomic_mask(scale=7, rmask=5):
            >>>     atom = MakeAtom(r).atom2dgaussian()
            >>>     _, mask = cv2.threshold(atom, thresh, 1, cv2.THRESH_BINARY)
            >>>     return atom, mask

        **scale (int):
            Controls the atom size (width of 2D Gaussian)
        **rmask (int):
            Controls the atomic mask size
    Returns:
        2D numpy array with ground truth data
    """
    if len(args) == 1:
        create_mask_func = args[0]
    else:
        create_mask_func = create_atom_mask_pair
    scale = kwargs.get("scale", 7)
    rmask = kwargs.get("rmask", 5)
    lattice_mask = np.zeros_like(lattice)
    for xy in xy_atoms:
        x, y = xy
        x = int(np.around(x))
        y = int(np.around(y))
        _, mask = create_mask_func(scale, rmask)
        r_m = mask.shape[0] / 2
        r_m1 = int(r_m + .5)
        r_m2 = int(r_m - .5)
        lattice_mask[x-r_m1:x+r_m2, y-r_m1:y+r_m2] = mask
    return lattice_mask


def create_multiclass_lattice_mask(imgdata: np.ndarray,
                                   coord_class_dict: Union[Dict[int, np.ndarray], np.ndarray],
                                   *args: Callable[[int, int], Tuple[np.ndarray, np.ndarray]],
                                   **kwargs: int) -> Union[List[np.ndarray], np.ndarray]:
    """
    Given a stack of experimental images and dictionary with atomic coordinates and classes
    creates a ground truth image. Notice that it will round fractional pixels.

    Args:
        lattice (3D numpy array):
            Experimental image as 2D numpy array
        coord_class_dict (dict or N x 3 numpy array):
            Dictionary with arrays containing coordiantes and classes for each atom/defect
            In each array, the first two columns are position of atoms.
            The third column is the "intensity"/class of each atom.
            It is also possible to pass a single N x 3 ndarray, which will be
            wrapped into a dictioanry automatically.
        *arg (python function):
            Function that creates two 2D numpy arrays with atom and
            corresponding mask for each atomic coordinate. It must have
            three parameters, 'scale', 'rmask', and 'intensity' that control
            size and intensity of simulated atom and corresponding atomic mask
        **scale (int):
            Controls the atom size (width of 2D Gaussian)
        **rmask (int):
            Controls the atomic mask size

    Returns:
        4D numpy array with ground truth data or list of 3D numpy arrays
    """
    if np.ndim(imgdata) == 2:
        imgdata = imgdata[None, ...]
    if isinstance(coord_class_dict, np.ndarray):
        coord_class_dict = {0: coord_class_dict}
    masks = []
    for i, img in enumerate(imgdata):
        masks.append(create_multiclass_lattice_mask_(
                        img, coord_class_dict[i], *args, **kwargs))
    shapes = [m.shape for m in masks]
    if len(set(shapes)) <= 1:
        masks = np.array(masks)
    return masks


def create_multiclass_lattice_mask_(lattice: np.ndarray, xyz_atoms: np.ndarray,
                                    *args: Callable[[int, int], Tuple[np.ndarray, np.ndarray]],
                                    **kwargs: int) -> np.ndarray:
    """
    Given experimental image and *xyz* atomic coordinates
    creates ground truth image. Notice that it will round fractional pixels.

    Args:
        lattice (2D numpy array):
            Experimental image as 2D numpy array
        xyz_atoms (N x 3 numpy array):
            The first two columns are position of atoms.
            The third column is the intensity of each atom.
        *arg (python function):
            Function that creates two 2D numpy arrays with atom and
            corresponding mask for each atomic coordinate. It must have
            three parameters, 'scale', 'rmask', and 'intensity' that control
            size and intensity of simulated atom and corresponding atomic mask
        **scale: int
            Controls the atom size (width of 2D Gaussian)
        **rmask: int
            Controls the atomic mask size

    Returns:
        3D numpy array with ground truth data
    """
    if len(args) == 1:
        create_mask_func = args[0]
    else:
        create_mask_func = create_atom_mask_pair
    scale = kwargs.get("scale", 7)
    rmask = kwargs.get("rmask", 7)
    lattice_mask = np.zeros(
        (lattice.shape[0], lattice.shape[1], len(np.unique(xyz_atoms[:, -1]))))
    if 0 in np.unique(xyz_atoms[:, -1]):
        xyz_atoms[:, -1] = xyz_atoms[:, -1] + 1
    atom_ch_d = {}
    for i, s in enumerate(np.unique(xyz_atoms[:, -1])):
        atom_ch_d[s] = i
    for atom in xyz_atoms:
        x, y, z = atom
        x = int(np.around(x))
        y = int(np.around(y))
        _, mask = create_mask_func(scale, rmask, z)
        r_m = mask.shape[0] / 2
        r_m1 = int(r_m + .5)
        r_m2 = int(r_m - .5)
        lattice_mask[x-r_m1:x+r_m2, y-r_m1:y+r_m2, atom_ch_d[z]] = mask
    lattice_mask_b = 1 - np.sum(lattice_mask, axis=-1)
    lattice_mask = np.concatenate((lattice_mask, lattice_mask_b[..., None]), axis=-1)
    lattice_mask[lattice_mask < 0] = 0
    return lattice_mask


def create_atom_mask_pair(sc: int = 5, r_mask: int = 5, intensity: int = 1):
    """
    Helper function for creating atom-label pair
    """
    amaker = MakeAtom(sc, r_mask, intensity)
    atom, mask = amaker.gen_atom_mask()
    return atom, mask
