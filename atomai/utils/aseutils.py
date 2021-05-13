"""
aseutils.py

=================

Module for working with atom/defect/particle coordinates and converting
to ASE object

Created by Ayana Ghosh (email: research.aghosh@gmail.com)
"""

from typing import Dict, Union

import numpy as np


def ase_obj_basic(coords_dict: Union[Dict[int, np.ndarray], np.ndarray],
                  frame_number: int, material_system: str,
                  map_dict: Dict[int, str],
                  filepath: str,
                  px2ang: float) -> None:
    """
    Takes the atomic coordinates and classes predicted by AtomAI's Segmentor
    models and uses them to create a structure file readable by packages
    such as Atomic Simulation Environment (ASE), VESTA, etc.
    It uses a simple cubic cell.

    Args:
        coords_dict: dictionary object of coordinates produced by AtomAI
        frame_number: image frame number (assumes a stack of images)
        material_system: name of material
        map_dict: dictionary which maps atomic classes from the NN output
            (keys) to strings corresponding to chemical elements (values)
        filepath: Savepath for the ASE object
        px2ang: Pixels-to-angstroms conversion coefficient,
            which is specific to each experiment

    Examples:

    >>> # Save coordinates for specific frame (0) as ASE object
    >>> ase_obj_basic(coordinates, 0, "Graphene",
    >>>               map_dict = {0: "C", 1: "Si"},
    >>>               "/content/Drive/POSCAR",
    >>>               px2ang=0.104)
    >>> # Read the saved files using ASE reader
    >>> cell = ase.io.vasp.read_vasp("/content/Drive/POSCAR")
    """
    ang2px = 1 / px2ang
    # make a list of dictionaries for all classes of atoms seperately
    all_dicts = []
    for c_atom in range(len(map_dict)):
        coordinates_filtered = {}
        for k, c in coords_dict.items():
            coordinates_filtered[k] = c[c[:, -1] == c_atom]
        all_dicts.append(coordinates_filtered)

    all_atoms = []
    length_coords = []
    for m in range(len(all_dicts)):
        pick_one_aoi = np.array(all_dicts[m][frame_number])  # np array
        pick_one_aoi = pick_one_aoi / ang2px  # pixel to angstrom conversion
        all_atoms.append(pick_one_aoi)
        length_coords.append(pick_one_aoi.shape[0])

    all_atoms_arr = all_atoms[0]
    for a in range(1, len(all_atoms)):
        all_atoms_arr = np.concatenate([all_atoms_arr, all_atoms[a]], axis=0)

    a_lattice = np.max(all_atoms_arr) + 0.2
    b_lattice = a_lattice
    c_lattice = a_lattice

    c_coords_aoi = (np.max(all_atoms_arr))
    all_atoms_arr[:, 2] = c_coords_aoi
    # open a text file and write to it following the format of .vasp file
    file1 = open(str(filepath), "w")
    file1.write(str(material_system) + "\n")
    file1.write(" 1.0000 \n")
    file1.write("  " + str(a_lattice) + " 0.0000 0.0000 \n")
    file1.write("  0.0000 " + str(b_lattice) + " 0.0000 \n")
    file1.write("  0.0000 0.0000 " + str(c_lattice) + "\n")
    for j in range(len(map_dict)):
        file1.write(" " + list(map_dict.values())[j] + " ")
    file1.write("\n")
    for s in range(len(length_coords)):
        file1.write(" " + str(length_coords[s]))
    file1.write("\n")
    file1.write("Cartesian \n")

    for i in range(all_atoms_arr.shape[0]):
        file1.write(str(all_atoms_arr[i][0]) + "\t" +
                    str(all_atoms_arr[i][1]) + "\t" +
                    str(all_atoms_arr[i][2]) + "\n")
    file1.close()

    print("You have successfully created an ASE object. \n")
    print("This is a cubic cell of " + material_system + ". \n")
    print("Now you can read it in using ase.io.vasp.read_vasp. \n")


def ase_obj_adv(a_lattice: float, b_lattice: float, c_lattice: float,
                coords_dict: Union[Dict[int, np.ndarray], np.ndarray],
                frame_number: int, material_system: str,
                map_dict: Dict[int, str],
                filepath: str,
                px2ang: float) -> None:
    """
    Takes the atomic coordinates and classes predicted by AtomAI's Segmentor
    models and uses them to create a structure file readable by packages
    such as Atomic Simulation Environment (ASE), VESTA, etc.
    It uses a customized cell with multiple atoms per user's choice.

    Args:
        a_lattice: list of lattice vector in a direction ([a1,a2,a3])
        b_lattice: list of lattice vector in a direction ([b1,b2,c3])
        c_lattice: list of lattice vector in a direction ([c1,c2,c3])
        coords_dict: dictionary object of coordinates produced by AtomAI
        frame_number: image frame number
        material_system: name of material
        map_dict: dictionary which maps atomic classes from the NN output
            (keys) to strings corresponding to chemical elements (values)
        filepath: Savepath for the ASE object
        px2ang: Pixels-to-Angstrom conversion coefficient,
            which is specific to each experiment

    Examples:

    >>> # Save coordinates for specific frame (0) as ASE object
    >>> ase_obj_adv([86.00000,0.00000,0.00000],
    >>>             [0.00000,86.00000,0.00000],
    >>>             [0.00000,0.00000,86.00000], coordinates, 0,
    >>>             "Graphene", map_dict = {0: "C", 1: "Si"},
    >>>             "/content/Drive/POSCAR_adv",
    >>>             px2ang=0.104)
    >>> # Read the saved file using ASE reader
    >>> cell = ase.io.vasp.read_vasp("/content/Drive/POSCAR_adv")
    """
    ang2px = 1 / px2ang
    all_dicts = []
    for c_atom in range(len(map_dict)):
        coordinates_filtered = {}
        for k, c in coords_dict.items():
            coordinates_filtered[k] = c[c[:, -1] == c_atom]
        all_dicts.append(coordinates_filtered)

    all_atoms = []
    length_coords = []
    for m in range(len(all_dicts)):
        pick_one_aoi = np.array(all_dicts[m][frame_number])  # np array
        pick_one_aoi = pick_one_aoi / ang2px  # pixel to angstrom conversion
        all_atoms.append(pick_one_aoi)
        length_coords.append(pick_one_aoi.shape[0])

    all_atoms_arr = all_atoms[0]
    for a in range(1, len(all_atoms)):
        all_atoms_arr = np.concatenate([all_atoms_arr, all_atoms[a]], axis=0)

    c_coords_aoi = (np.max(all_atoms_arr))
    all_atoms_arr[:, 2] = c_coords_aoi

    # open a text file and write to it following the format of .vasp file
    file1 = open(str(filepath), "w")
    file1.write(str(material_system) + "\n")
    file1.write(" 1.0000 \n")
    file1.write("  " + str(a_lattice[0]) + " " + str(a_lattice[1]) + " " +
                str(a_lattice[2]) + "\n")
    file1.write("  " + str(b_lattice[0]) + " " + str(b_lattice[1]) + " " +
                str(b_lattice[2]) + "\n")
    file1.write("  " + str(c_lattice[0]) + " " + str(c_lattice[1]) + " " +
                str(c_lattice[2]) + "\n")

    for j in range(len(map_dict)):
        file1.write(" " + list(map_dict.values())[j] + " ")
    file1.write("\n")
    for s in range(len(length_coords)):
        file1.write(" " + str(length_coords[s]))
    file1.write("\n")
    file1.write("Cartesian \n")

    for i in range(all_atoms_arr.shape[0]):
        file1.write(str(all_atoms_arr[i][0]) + "\t" +
                    str(all_atoms_arr[i][1]) + "\t" +
                    str(all_atoms_arr[i][2]) + "\n")
    file1.close()
    print("You have successfully created an ASE object. \n")
    print("You have prepared " + material_system + ". \n")
    print("Now you can read it in using ase.io.vasp.read_vasp. \n")
