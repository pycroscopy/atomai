from typing import Dict, Union

import os
import urllib.request

import numpy as np
import progressbar


def stem_smbfo(download: bool = True,
               filedir: str = './'
               ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Downloads the scanning transmission electron microscopy (STEM) datasets
    from the combinatorial library of the Sm-doped BiFeO3 (BFO) grown to cover
    the composition range from pure ferroelectric BFO to orthorhombic 20% Sm-doped BFO.
    For details, see npj Comput Mater 6, 127 (2020). https://doi.org/10.1038/s41524-020-00396-2.

    Args:
        download: downloads the dataset from the public repository
        filedir: directory to save the downloaded data

    Returns:
        Nested dictionary where the 1st dictionary describes different
        Sm concentrations, and the 2nd dictionary has chemical and physical
        descriptors for each concentration.

    Examples:

        >>> # Download the dataset
        >>> dataset = atomai.utils.datasets.stem_smbfo()
        >>> # Plot main image and polarization values for each Sm concentration
        >>> for k, d in dataset.items():
        >>>     _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        >>>     y, x = d["xy_COM"].T  # get the center of the mass for each unit cell
        >>>     ax1.imshow(d["main_image"], origin="lower", cmap='gray')
        >>>     ax1.set_title(k)
        >>>     ax2.scatter(x, y, c=d["Pxy"][:, 0], s=3, cmap='RdBu_r')
        >>>     ax3.scatter(x, y, c=d["Pxy"][:, 1], s=3, cmap='RdBu_r')
        >>>     plt.show()
    """
    print("If you use this dataset in your work, please consider citing it"
          " using the following DOI: https://doi.org/10.13139/ORNLNCCS/1773704.")
    if download:
        print("Downloading the dataset. This may take a few minutes.")
        url = "https://zenodo.org/record/4876786/files/composition_series_dict_full.npy"
        urllib.request.urlretrieve(
            url, os.path.join(filedir, "SmBFO_composition_series.npy"), ProgressBar())
    dataset = np.load(
        os.path.join(filedir, "SmBFO_composition_series.npy"), allow_pickle=True)[()]
    return dataset


def stem_graphene(download: bool = True,
                  filedir: str = './'
                  ) -> Dict[int, Dict[str, Union[np.ndarray, Dict]]]:
    """
    Downloads the scanning transmission electron microscopy (STEM) datasets
    from the graphene samples. See https://doi.ccs.ornl.gov/ui/doi/338 for details.

    Args:
        download: downloads the dataset from the public repository
        filedir: directory to save the downloaded data

    Returns:
        Nested dictionary with STEM movies in the form of M x N x L arrays
        and the corresponding metadata

    Examples:

        >>> # Download the dataset
        >>> dataset = atomai.utils.datasets.stem_graphene()
        >>> # Get one STEM movie and the associated metadata
        >>> data = dataset[3]["image_data"]  # ndarray of size (50, 1024, 1024)
        >>> metadata = dataset[3]["metadata] # dictionary with experimental parameters
    """
    print("If you use this dataset in your work, please consider citing it"
          " using the following DOI: https://doi.org/10.13139/ORNLNCCS/1773705.")
    if download:
        print("Downloading the dataset. This may take a few minutes.")
        url = "https://zenodo.org/record/4885132/files/graphene_imgstacks_dict.npy"
        urllib.request.urlretrieve(
            url, os.path.join(filedir, "graphene_imgstacks_dict.npy"), ProgressBar())
    dataset = np.load(
        os.path.join(filedir, "graphene_imgstacks_dict.npy"), allow_pickle=True)[()]
    return dataset


class ProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()
