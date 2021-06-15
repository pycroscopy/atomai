import sys
import os
import numpy as np
import matplotlib
import pickle
matplotlib.use('Agg')

sys.path.append("../../../")

from atomai.utils import ase_obj_basic, ase_obj_adv

def dump_with_pickle(pyObj, filename):
    fobj1 = open(filename, 'wb')
    pickle.dump(pyObj, fobj1)
    fobj1.close()
def load_with_pickle(filename):
    fobj1 = open(filename, 'rb')
    pyObj = pickle.load(fobj1)
    fobj1.close()
    return pyObj

map_dict_atoms_ = os.path.join(
    os.path.dirname(__file__), "test_data/map_dict_chk")
coords_dict_ = os.path.join(
    os.path.dirname(__file__), "test_data/coordinates_dict")
map_dict_ck = load_with_pickle(map_dict_atoms_)
coord_dict = load_with_pickle(coords_dict_)

pos_basic = os.path.join(
    os.path.dirname(__file__), "test_data/POSCAR_general")
pos_adv = os.path.join(
    os.path.dirname(__file__), "test_data/POSCAR_adv_general")

ase_obj_basic(coord_dict, 0, "Graphene", map_dict_ck,
              pos_basic, 0.104)

ase_obj_adv([86.00000, 0.00000, 0.00000],
            [0.00000, 86.00000, 0.00000],
            [0.00000, 0.00000, 86.00000],
            coord_dict, 0, "Graphene", map_dict_ck,
            pos_adv, 0.104)