"""
viz.py
======

Utility functions for plotting

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

from typing import Union, List, Optional, Dict

import os
import warnings

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import numpy as np


def plot_losses(train_loss: Union[List[float], np.ndarray],
                test_loss: Union[List[float], np.ndarray]) -> None:
    """
    Plots train and test losses
    """
    print('Plotting training history')
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(train_loss, label='Train')
    ax.plot(test_loss, label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()


def plot_coord(img: np.ndarray,
               coord: np.ndarray,
               fsize: int = 6, **kwargs) -> None:
    """
    Plots coordinates (colored according to atom class)
    """
    cmap_ = kwargs.get("cmap", "RdYlGn")
    y, x, c = coord.T
    plt.figure(figsize=(fsize, fsize))
    plt.imshow(img, cmap='gray', origin='lower')
    plt.scatter(x, y, c=c, cmap=cmap_, s=8)
    plt.show()


def draw_boxes(imgdata: np.ndarray, defcoord: np.ndarray,
               bbox: int = 16, fsize: int = 6) -> None:
    """
    Draws boxes centered around the extracted dedects
    """
    _, ax = plt.subplots(1, 1, figsize=(fsize, fsize))
    ax.imshow(imgdata, cmap='gray')
    for point in defcoord:
        startx = int(round(point[0] - bbox))
        starty = int(round(point[1] - bbox))
        p = patches.Rectangle(
            (starty, startx), bbox*2, bbox*2,
            fill=False, edgecolor='orange', lw=2)
        ax.add_patch(p)
    ax.grid(False)
    plt.show()


def plot_trajectories(traj: np.ndarray, frames: np.ndarray,
                      **kwargs: Union[int, str]) -> None:
    """
    Plots individual trajectory (as position (radius) vector)

    Args:
        traj (n x 3 ndarray):
            numpy array where first two columns are coordinates
            and the 3rd columd are classes
        frames ((n,) ndarray):
            numpy array with frame numbers
        **lv (int):
            latent variable value to visualize (Default: 1)
        **fov (int or list):
            field of view or scan size
        **fsize (int):
            figure size (Default: 6)
        **cmap (str):
            colormap (Default: jet)
    """
    fov = kwargs.get("fov")
    cmap = kwargs.get("cmap", "jet")
    fsize = kwargs.get("fsize", 6)
    r_coord = np.linalg.norm(traj[:, :2], axis=1)
    if traj.shape[1] == 3:
        c_ = traj[:, -1]
    elif traj.shape[1] > 3:
        lv = kwargs.get("lv", 3)
        c_ = traj[:, 2 + lv]
    plt.figure(figsize=(fsize*2, fsize))
    plt.scatter(frames, r_coord, c=c_, cmap=cmap)
    if fov:
        if isinstance(fov, list) and len(fov) == 2:
            fov = np.sqrt(fov[0]**2 + fov[1]**2)
        elif isinstance(fov, int):
            fov = np.sqrt(2*fov**2)
        else:
            raise ValueError("Pass 'fov' argument as integer or 2-element list")
        plt.ylim(0, fov)
    plt.xlabel("Time step (a.u.)", fontsize=18)
    plt.ylabel("Position vector", fontsize=18)
    cbar = plt.colorbar()
    cbar_lbl = "States" if traj.shape[1] == 3 else "Latent variable {}".format(lv)
    cbar.set_label(cbar_lbl, fontsize=16)
    plt.clabel
    plt.title("Trajectory", fontsize=20)
    plt.show()


def plot_transitions(matrix: np.ndarray,
                     states: Optional[np.ndarray] = None,
                     gmm_components: Optional[np.ndarray] = None,
                     plot_values: bool = False,
                     **kwargs: Union[bool, int, str]) -> None:
    """
    Plots transition matrix and (optionally) most frequent/probable transitions

    Args:
        m (2D numpy array):
            Transition matrix
        states (numpy array):
            Array with states (e.g. [2, 5, 7])
        gmm_components (4D numpy array):
            GMM components (optional)
        plot_values (bool):
            Show calculated transtion rates
        **transitions_to_plot (int):
            number of transitions (associated with largest prob values) to plot
        **plot_toself (bool):
            Skips transitions into self when plotting transitions with largest probs
        **fsize (int): figure size
        **cmap (str): color map
    """
    fsize = kwargs.get("fsize", 6)
    cmap = kwargs.get("cmap", "Reds")
    transitions_to_plot = kwargs.get("transitions_to_plot", 6)
    plot_toself = kwargs.get("plot_toself", True)
    m = matrix
    _, ax = plt.subplots(1, 1, figsize=(fsize, fsize))
    ax.matshow(m, cmap=cmap)
    if states is None:
        states = np.arange(len(m)) + 1
    xt = states
    ax.set_xticks(np.arange(len(xt)))
    ax.set_yticks(np.arange(len(xt)))
    ax.set_xticklabels((xt).tolist(), rotation='horizontal', fontsize=14)
    ax.set_yticklabels((xt).tolist(), rotation='horizontal', fontsize=14)
    ax.set_title('Transition matrix', y=1.1, fontsize=20)
    if plot_values:
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, np.around(v, 2), ha='center', va='center', c='b')
    ax.set_xlabel('Transition class', fontsize=18)
    ax.set_ylabel('Starting class', fontsize=18)
    plt.show()
    if gmm_components is not None:
        idx_ = np.unravel_index(np.argsort(m.ravel()), m.shape)
        idx_ = np.dstack(idx_)[0][::-1]
        print()
        i_ = 0
        for i in idx_:
            if plot_toself is False and i[0] == i[1]:
                continue
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(fsize, fsize//2))
            if gmm_components.shape[-1] == 3:
                start_comp = gmm_components[states[i[0]]-1]
                trans_comp = gmm_components[states[i[1]]-1]
            else:
                start_comp = np.sum(gmm_components[states[i[0]]-1], axis=-1)
                trans_comp = np.sum(gmm_components[states[i[1]]-1], axis=-1)
            print("Starting class  --->  Transition class (Prob: {})".
                  format(m[tuple(i)]))
            ax1.imshow(start_comp, cmap=cmap)
            ax1.set_title("GMM component {}".format(states[i[0]]))
            ax2.imshow(trans_comp, cmap=cmap)
            ax2.set_title("GMM_component {}".format(states[i[1]]))
            plt.show()
            i_ = i_ + 1
            if i_ == transitions_to_plot - 1:
                break
    return


def plot_trajectories_transitions(trans_dict: Dict, k: int,
                                  plot_values: bool = False,
                                  **kwargs: Union[bool, int, str]) -> None:
    """
    Plots trajectory witht he associated transitions.

    Args:
        trans_dict (dict):
            Python dictionary containing trajectories, frame numbers,
            transitions and the averaged GMM components. Usually this is
            an output of atomstat.transition_matrix
        k (int): Number of trajectory to vizualize
        plot_values (bool): Show calculated transtion rates
        **transitions_to_plot (int):
            number of transitions (associated with largerst prob values) to plot
        **fsize (int): figure size
        **cmap (str): color map
        **fov (int or list): field of view (scan size)
    """
    traj = trans_dict["trajectories"][k]
    frames = trans_dict["frames"][k]
    trans = trans_dict["transitions"][k]
    plot_trajectories(traj, frames, **kwargs)
    print()
    s_true = np.unique(traj[:, -1]).astype(np.int64)
    plot_transitions(
        trans, s_true, trans_dict["gmm_components"],
        plot_values, **kwargs)
    return


def plot_lattice_bonds(distances: np.ndarray,
                       atom_pairs: np.ndarray,
                       distance_ideal: float = None,
                       frame: int = 0,
                       display_results: bool = True,
                       **kwargs: Union[str, int]) -> None:
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
            ax1.plot(y, x, c=color)
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


def animation_from_png(png_dir: str, moviename: str = 'anim',
                       duration: int = 1, savedir: str = './',
                       remove_dir: bool = True) -> None:
    """
    Create animation from saved png files
    """
    import imageio, shutil
    images = []
    if ".ipynb_checkpoints" in os.listdir(png_dir):
        shutil.rmtree(os.path.join(png_dir, ".ipynb_checkpoints"))
    for file_name in sorted(os.listdir(png_dir),
                            key=lambda fname: int(fname.split('.')[0])):
        if file_name.endswith('.png'):
            images.append(imageio.imread(os.path.join(png_dir, file_name)))
    imageio.mimsave(os.path.join(savedir, moviename + '.gif'), images)
    if remove_dir:
        shutil.rmtree(png_dir)
