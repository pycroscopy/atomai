"""
graphx.py
=========

Module with utility functions for working with graphs.
Many parts were adapted from Jaap Kroes's Polypy project (https://github.com/jaapkroe/polypy)

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ao4microscopy.com)
"""

import itertools
from copy import copy, deepcopy
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
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
        if depth < max_depth:
            depth += 1
            visited.append(v)
            if v == goal:
                lp = len(self.path)
                if depth < lp or not lp:
                    self.path = copy(visited)
                    max_depth = depth
            else:
                for n in v.neighborscopy:
                    if n.ingraph:
                        n.ingraph = False
                        self.shortest_path(n, goal, max_depth, visited, depth)
                        n.ingraph = True
            visited.pop()

    def rings_to_nx_graph(self, ring_size: int) -> Type[nx.Graph]:
        """
        Transform detected rings into networkx graph object
        """
        g_nx = nx.Graph()
        for ring in self.rings:
            if len(ring) not in ring_size:
                continue
            for v in ring:
                g_nx.add_node(v.id, pos=tuple(v.pos), atom=v.atom)
                for nn in v.neighbors:
                    g_nx.add_node(nn.id, pos=tuple(nn.pos), atom=nn.atom)
                for nn in v.neighbors:
                    g_nx.add_edge(v.id, nn.id)
        nodes_to_remove = [node for node, degree in g_nx.degree() if degree < 2]
        g_nx.remove_nodes_from(nodes_to_remove)
        return g_nx

    def nx_graph(self) -> Type[nx.Graph]:
        """
        Transforms the entire graph to networkx object
        """
        g_nx = nx.Graph()
        d = False
        if np.all(self.coordinates[0, 2] == self.coordinates[:, 2]):
            d = True
        for v in self.vertices:
            g_nx.add_node(
                v.id, pos=tuple(v.pos[:2] if d else v.pos), atom=v.atom)
            for nn in v.neighbors:
                g_nx.add_node(
                    nn.id, pos=tuple(nn.pos[:2] if d else nn.pos), atom=nn.atom)
            for nn in v.neighbors:
                g_nx.add_edge(v.id, nn.id)
        return g_nx


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


def find_cycles(coordinate_data: np.ndarray,
                cycles: Union[int, List[int]],
                map_dict: Dict[int, str],
                px2ang: float,
                **kwargs: float) -> np.ndarray:
    """
    Finds coordinates of cycles (rings) with a specific number of elements
    (can be used for identifying e.g. non-hexagonal rings in graphene)

    Args:
        coordinate_data (numpy array):
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
    coordinates = deepcopy(coordinate_data)
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


def find_cycle_clusters(coordinate_data: np.ndarray,
                        cycles: Union[int, List[int]],
                        map_dict: Dict[int, str],
                        px2ang: float,
                        **kwargs: float) -> List[np.ndarray]:
    """
    Finds clusters of cycles with a specific number of elements
    (can be used for identifying e.g. topological defects in graphene)
    
    Args:
        coordinate_data (numpy array):
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
        List of coordinates of the clusters with requested cycle types

    Examples:

        >>> # Dictionary for mapping classes from NN output to atomic elements
        >>> map_dict = {0: "C", 1: "Si"}
        >>> # conversion coefficent (pixels to angstroms)
        >>> px2ang = 0.104
        >>> # Identify coordinates of clusters with 5-member and 7-member rings
        >>> coords_clusters = graphx.find_cycle_clusters(
        >>>        coordinates, cycles=[5, 7], map_dict=map_dict, px2ang=px2ang)
    """
    if isinstance(cycles, int):
        cycles = [cycles]
    coordinates = deepcopy(coordinate_data)
    coordinates[:, :-1] = coordinates[:, :-1] * px2ang
    e = kwargs.get("expand", 1.2)
    G = Graph(coordinates, map_dict)
    G.find_neighbors(expand=e)
    G.polycount(max_depth=max(cycles))
    G.remove_filled_polygons()
    g_nx = G.rings_to_nx_graph(cycles)
    sub_graphs = list(
        g_nx.subgraph(c).copy() for c in nx.connected_components(g_nx))
    coordinates_filtered_all = []
    for sg in sub_graphs:
        atom_idx = [i for i in sg.nodes.keys()]
        coordinates_filtered = coordinates[atom_idx]
        coordinates_filtered = coordinates_filtered[:, :-1] * (1 / px2ang)
        coordinates_filtered_all.append(coordinates_filtered)
    return coordinates_filtered_all


def plot_graph(G: Type[nx.Graph],
               img: Optional[np.ndarray] = None,
               fsize: Union[int, Tuple[int, int]] = 8,
               show_labels: bool = False,
               **kwargs: Union[int, str, float]) -> None:

    """
    Plots graph overlayed on the original image (raw or NN/VAE output)

    Args:
        G (networkx object): Graph object
        img (numpy array): 2D image (used to find coordinates for constructing graph)
        fsize (int or tuple): figure size
        show_labels (bool): display node labels (e.g. '1', '13' or 'C', 'Si')
        **kwargs: additional plotting parameters
            (node_size, node_color, edge_color, label_size, label_color, cmap, alpha)
    """
    fsize = fsize if isinstance(fsize, tuple) else (fsize, fsize)
    _, ax = plt.subplots(1, 1, figsize=fsize)
    if isinstance(G, Graph):
        G = G.nx_graph()
    for k, v in nx.get_node_attributes(G, 'pos').items():
        G.nodes[k]['pos'] = v[::-1]
    pos = nx.get_node_attributes(G, 'pos')
    if img is not None:
        ax.imshow(img, origin="lower", cmap=kwargs.get("cmap", "gray"))
    nx.draw_networkx_nodes(
        G, pos=pos, nodelist=G.nodes(), ax=ax,
        node_size=kwargs.get("node_size", 30),
        node_color=kwargs.get("node_color", "#1f78b4"),
        alpha=kwargs.get("alpha", None))
    nx.draw_networkx_edges(
        G, pos, width=1, ax=ax,
        edge_color=kwargs.get("edge_color", "orange"),
        alpha=kwargs.get("alpha", None))
    if show_labels:
        atomic_labels = None
        if kwargs.get("show_elements"):
            atomic_labels = nx.get_node_attributes(G, 'atom')
        nx.draw_networkx_labels(G, pos, labels=atomic_labels, ax=ax,
                                font_size=kwargs.get("label_size", 7),
                                font_color= kwargs.get("label_color", "black"))
    plt.show()


def filter_subgraphs_(coordinate_arr: np.ndarray,
                      map_dict: Dict[int, str],
                      px2ang: float,
                      **kwargs: float) -> np.ndarray:
    """
    Filters atomic coordinates using connected subgraphs.

    Args:
        coordinates (ndarray):
            Atomic coordinates as a numpy array where the first 2 columns
            are x and y coordinates and the third columns is atomic class
        map_dict (dict):
            dictionary which maps atomic classes from the NN output (dict keys)
            to strings corresponding to chemical elements (dict values)
        px2ang (float):
            coefficient used to convert pixel values to angstrom values as
            coordinates_in_angstroms = px2ang * coordiantes_in_pixels
        **expand (float):
            coefficient determining the maximum allowable expansion of
            atomic bonds when constructing a graph. For example, the two
            C atoms separated by 1.7 ang will be connected only if the
            expansion coefficient is 1.2 or larger. The default value is 1.2

    Returns:
        Filtered atomic coordinates
    """
    coordinates = deepcopy(coordinate_arr)
    coordinates[:, :-1] = coordinates[:, :-1] * px2ang
    e = kwargs.get("expand", 1.2)
    G = Graph(coordinates, map_dict)
    G.find_neighbors(expand=e)
    G_nx = G.nx_graph()
    map_dict_inv = {v: k for (k, v) in map_dict.items()}
    sub_graphs = list(G_nx.subgraph(c).copy() for c in nx.connected_components(G_nx))
    i = np.argmax([len(sg) for sg in sub_graphs])
    main_graph = sub_graphs[i]
    pos = nx.get_node_attributes(main_graph, 'pos')
    names = nx.get_node_attributes(main_graph, 'atom')
    coordinates_filtered = []
    for n, c in zip(names.values(), pos.values()):
        cls = map_dict_inv[n]
        c_arr = np.array([c[0]/px2ang, c[1]/px2ang, cls]).reshape(1, -1)
        coordinates_filtered.append(c_arr)
    coordinates_filtered = np.concatenate(coordinates_filtered)

    return coordinates_filtered


def filter_subgraphs(coordinates: Union[Dict[int, np.ndarray], np.ndarray],
                     map_dict: Dict[int, str],
                     px2ang: float,
                     **kwargs: float) -> Dict[int, np.ndarray]:
    """
    Filters atomic coordinates using connected subgraphs.

    Args:
        coordinates (dict or ndarray):
            Atomic coordinates (e.g. from the output of atomnet.predictor)
        map_dict (dict):
            dictionary which maps atomic classes from the NN output (dict keys)
            to strings corresponding to chemical elements (dict values)
        px2ang (float):
            coefficient used to convert pixel values to angstrom values as
            coordinates_in_angstroms = px2ang * coordiantes_in_pixels
        **expand (float):
            coefficient determining the maximum allowable expansion of
            atomic bonds when constructing a graph. For example, the two
            C atoms separated by 1.7 ang will be connected only if the
            expansion coefficient is 1.2 or larger. The default value is 1.2

    Returns:
        Filtered atomic coordinates
    """
    if isinstance(coordinates, np.ndarray):
        coordinates = {0: coordinates}
    coordinates_filtered_d = {}
    for k, coord in coordinates.items():
        coordinates_filtered_d[k] = filter_subgraphs_(
            coord, map_dict, px2ang, **kwargs)
    return coordinates_filtered_d
