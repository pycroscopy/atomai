import sys
import os
import numpy as np
import pytest
from numpy.testing import assert_equal
import matplotlib
matplotlib.use('Agg')

sys.path.append("../../../")


test_file_m_ = os.path.join(
    os.path.dirname(__file__), 'test_data/POSCAR_adv_general')


def test_ase_linecount():
    items_per_line = []
    target_header_count = [1, 1, 3, 3, 3, 2, 2, 1]
    body_item_count = 3
    with open(test_file_m_) as fp:
        line = fp.readline()
        cnt = 0
        while line:
            line = line.strip()
            line_split = line.split()
            num_items = len(line_split)
            items_per_line.append(num_items)
            line = fp.readline()
            cnt += 1

    for i in range(len(target_header_count)):
        assert_equal(items_per_line[i], target_header_count[i],
                     "Item count mismatch at Line #"+str(i))

    for i in range(8, len(items_per_line)):
        assert_equal(items_per_line[i], body_item_count,
                     "Item count mismatch at Line #"+str(i))


def test_ase_lattice_unit():
    with open(test_file_m_) as fp:
        line = fp.readline()
        cnt = 0
        while line:
            line = line.strip()
            line_split = line.split()

            if cnt == 1:
                val = np.int(np.float(line_split[0]))
                assert_equal(val, 1, 'Mismatch of lattice unit')

            line = fp.readline()
            cnt += 1


def test_ase_coodinate_system():
    with open(test_file_m_) as fp:
        line = fp.readline()
        cnt = 0
        while line:
            line = line.strip()
            line_split = line.split()

            if cnt == 7:
                matchVal = np.int(line_split[0] == 'Cartesian')
                assert_equal(matchVal, 1, 'Should be Cartesian System')

            line = fp.readline()
            cnt += 1


def test_ase_atom_names():
    with open(test_file_m_) as fp:
        line = fp.readline()
        cnt = 0
        while line:
            line = line.strip()
            line_split = line.split()

            if cnt == 5:
                matchVal_C = np.int(line_split[0] == 'C')
                matchVal_Si = np.int(line_split[1] == 'Si')
                assert_equal(matchVal_C, 1, 'Should be C')
                assert_equal(matchVal_Si, 1, 'Should be Si')

            line = fp.readline()
            cnt += 1