"""
graphx.py
=========

Module with utility functions for working with graphs.
Many parts were adapted from Jaap Kroes's Polypy project (https://github.com/jaapkroe/polypy)

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ao4microscopy.com)
"""

import itertools
from copy import copy
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
from mendeleev import element
from scipy import spatial


class Node:
    """
    Node representing an atom

    Args:
        idx (int): node id in a graph
        pos (list): xyz coordinates of atom
        atom (str): atomic element (e.g. 'Si', 'C')
    """
    def __init__(self,
                 idx: int = 0,
                 pos: List[float] = None,
                 atom: str = 'C'
                 ) -> None:
        """
        Initializes a single node
        """
        pos = [] if pos is None else pos
        self.neighbors = []
        self.neighborscopy = []
        self.nn = 0
        self.id = idx
        self.pos = pos
        self.atom = atom
        self.ingraph = True
        self.visited = False


class Graph:
    """
    Class for constructing and analyzing a graph from atomic coordinates

    Args:
        coordinates (numpy array):
            3D or 4D numpy array where the last column is atom class (0, 1, ...)
            and all the columns before last are atomic coordinates in angstroms
        map_dict (dict):
            dictionary which maps atomic classes from the NN output (keys)
            to strings corresponding to chemical elements (values)
    """

    def __init__(self, coordinates: np.ndarray,
                 map_dict: Dict) -> None:
        """
        Initializes a graph object
        """
        self.vertices = []
        if coordinates.shape[-1] == 3:
            coordinates = np.concatenate(
                (coordinates[:, :2],
                 np.zeros_like(coordinates)[:, 0:1],
                 coordinates[:, 2:3]),
                axis=-1)
        for i, coords in enumerate(coordinates):
            v = Node(i, coords[:-1].tolist(), map_dict[coords[-1]])
            self.vertices.append(v)
        self.coordinates = coordinates
        self.map_dict = map_dict
        self.size = len(coordinates)
        self.rings = []
        self.path = []
        self.improper = []

    def find_neighbors(self, **kwargs: float):
        """
        Identifies neighbors of each graph node

        Args:
            **expand (float):
                coefficient determining the maximum allowable expansion of
                atomic bonds when constructing a graph. For example, the two
                C atoms separated by 1.7 ang will be connected only if the
                expansion coefficient is 1.2 or larger. The default value is 1.2
        """
        for v in self.vertices:
            del v.neighbors[:]
        Rij = get_interatomic_r
        e = kwargs.get("expand", 1.2)
        tree = spatial.cKDTree(self.coordinates[:, :3])
        uval = np.unique(self.coordinates[:, -1])
        if len(uval) == 1:
            rmax = Rij([self.map_dict[uval[0]], self.map_dict[uval[0]]], e)
            neighbors = tree.query_ball_point(self.coordinates[:, :3], r=rmax)
            for v, nn in zip(self.vertices, neighbors):
                for n in nn:
                    if self.vertices[n] != v:
                        v.neighbors.append(self.vertices[n])
                        v.neighborscopy.append(self.vertices[n])
        else:
            uval = [self.map_dict[u] for u in uval]
            apairs = [(p[0], p[1]) for p in itertools.product(uval, repeat=2)]
            rij = [Rij([a[0], a[1]], e) for a in apairs]
            rmax = np.max(rij)
            rij = dict(zip(apairs, rij))
            for v, coords in zip(self.vertices, self.coordinates):
                atom1 = self.map_dict[coords[-1]]
                nn = tree.query_ball_point(coords[:3], r=rmax)
                for n, coords2 in zip(nn, self.coordinates[nn]):
                    if self.vertices[n] != v:
                        atom2 = self.map_dict[coords2[-1]]
                        eucldist = np.linalg.norm(
                            coords[:3] - coords2[:3])
                        if eucldist <= rij[(atom1, atom2)]:
                            v.neighbors.append(self.vertices[n])
                            v.neighborscopy.append(self.vertices[n])

    def find_rings(self,
                   v: Type[Node],
                   rings: List[List[Type[Node]]] = [],
                   max_depth: Optional[int] = None,
                   visited: List[Type[Node]] = [],
                   depth: int = 0,
                   root: Type[Node] = None,
                   ) -> None:
        """
        Recursive depth first search (dfs)

        Args:
            v (Node object): starting node
            rings (list): list of identified rings (lists of nodes)
            max_depth: maximum depth value for dfs
        """
        if root is None:
            root = v
            root.ingraph = False
        if max_depth:
            if depth >= max_depth:
                return False
        visited.append(v)
        depth += 1
        for i, n in enumerate(v.neighbors):
            if depth > 2 and n is root:
                rings.append(copy(visited))
            elif n.ingraph:
                n.ingraph = False
                self.find_rings(n, rings,  max_depth, visited, depth, root)
                n.ingraph = True
        if depth == 2:
            if root in v.neighbors:
                v.neighbors.remove(root)
        visited.pop()

    def polycount(self, max_depth: int) -> None:
        """
        Find the rings from every atom (node)

        Args:
            max_depth (int): maximum depth for dfs algorithm
        """
        for i in range(self.size):
            self.find_rings(
                self.vertices[i], self.rings, max_depth)
        for v in self.vertices:
            self.neighbors = copy(v.neighborscopy)

    def remove_filled_polygons(self):
        """
        Removes rings that are not shortest-path rings
        """
        for v in self.vertices:
            v.ingraph = True
        size = len(self.rings)
        to_be_removed = []
        for i in range(size):
            r = self.rings[i]
            l = len(r)
            remove = False
            for j in range(l):
                for k in range(j+2, l):
                    if not remove:
                        v, n = r[j], r[k]
                        djk = abs(j-k)
                        dist_r = min(djk, abs(djk-l))+1
                        self.path = []
                        self.shortest_path(v, n, depth=0, max_depth=dist_r)
                        dist_g = len(self.path)
                        if dist_g < dist_r:
                            remove = True
            if remove:
                to_be_removed.append(r)
        for r in to_be_removed:
            self.rings.remove(r)

    def shortest_path(self,
                      v: Type[Node],
                      goal: Type[Node],
                      max_depth: int,
                      visited: List[Type[Node]] = [],
                      depth: int = 1) -> None:
        """
        Computes shortest path in a (sub-)graph

        Args:
            v (Node object): starting node for path
            goal (Node object): ending node for path
            max_depth (int): maximum search depth
        """
        if depth < max_depth:         # start searching
            depth += 1                  # go one layer below
            visited.append(v)           # add this point to visited points
            if v == goal:
                lp = len(self.path)
                if depth < lp or not lp:  # current path shorter or first path found
                    self.path = copy(visited)
                    max_depth = depth
            else:
                for n in v.neighborscopy: # search all neighbors
                    if n.ingraph:           # not already searched
                        n.ingraph = False
                        self.shortest_path(n, goal, max_depth, visited, depth)
                        n.ingraph = True
            visited.pop()


def get_interatomic_r(atoms: Union[Tuple[str], List[str]],
                      expand: Optional[float] = None) -> float:
    """
    Calculates bond length between two elements

    Args:
        atoms (list or tuple):
            pair of atoms for which the bond length needs to be calculated
        expand (float):
            coefficient determining the maximum allowable expansion of
            atomic bonds when constructing a graph. For example, the two
            C atoms separated by 1.7 ang will be connected only if the
            expansion coefficient is 1.2 or larger. The default value is 1.2

    Returns:
        Interatomic bond length
    """
    atom1, atom2 = element(atoms)
    r12 = (atom1.covalent_radius + atom2.covalent_radius) / 100
    if expand:
        r12 = expand * r12
    return r12


def find_cycles(coordinates: np.ndarray,
                cycles: Union[int, List[int]],
                map_dict: Dict[int, str],
                px2ang: float,
                **kwargs: float) -> np.ndarray:
    """
    Finds coordinates of cycles (rings) with a specific number of elements
    (can be used for identifying e.g. non-hexagonal rings in graphene)

    Args:
        coordinates (numpy array):
            3D or 4D numpy array where the last column is atom class (0, 1, ...)
            and all the columns before last are atomic coordinates in angstroms
        cycles (list or int):
            List with lengths of rings to be identified;
            can also be a single integer
        map_dict (dict):
            dictionary which maps atomic classes from the NN output (keys)
            to strings corresponding to chemical elements (values)
        px2ang (float):
            coefficient used to convert pixel values to angstrom values as
            coordinates_in_angstroms = px2ang * coordiantes_in_pixels
        **expand (float):
                coefficient determining the maximum allowable expansion of
                atomic bonds when constructing a graph. For example, the two
                C atoms separated by 1.7 ang will be connected only if the
                expansion coefficient is 1.2 or larger. The default value is 1.2

    Returns
        Coordinates of the requested cycle types
    """
    if isinstance(cycles, int):
        cycles = [cycles]
    coordinates[:, :-1] = coordinates[:, :-1] * px2ang
    e = kwargs.get("expand", 1.2)
    G = Graph(coordinates, map_dict)
    G.find_neighbors(expand=e)
    G.polycount(max_depth=max(cycles))
    G.remove_filled_polygons()
    rl = [sorted([int(v.id) for v in r]) for r in G.rings]
    rl = sorted(rl, key=lambda x: (len(x), x[0], x[1], x[2]))
    coordinates_ = [coordinates[r] for r in rl if len(r) in cycles]
    coordinates_ = np.concatenate(coordinates_)
    coordinates_[:, :-1] = coordinates_[:, :-1] * (1 / px2ang)
    return coordinates_