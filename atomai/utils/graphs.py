"""
graphs.py
=========

Module with utility functions for work with graphs

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ao4microscopy.com)
"""

from typing import Dict, List, Union, Type, Tuple
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def construct_graph(coord: np.ndarray,
                    max_edge_length: int,
                    map_dict: Dict,
                    **kwargs
                    ) -> Type[nx.Graph]:
    """
    Constructs undirected graph from atomic coordiantes

    Args:
        coord (ndarray):
            Atomic coordinates as a numpy array where the first 2 columns
            are x and y coordinates and the third columns is atomic class
        max_edge_length (int):
            Maximum graph edge length (nodes above this length will not be connected)
        map_dict (dict):
            dictionary which maps atomic classes from the NN output (dict keys)
            to strings corresponding to chemical elements (dict values)
    """
    min_edge_length = kwargs.get("min_edge_length")
    if min_edge_length is None:
        min_edge_length = max_edge_length // 2
    # unique classes corresponding to different atomic species
    unique_idx = np.unique(coord[:, -1])
    # create graph object
    G = nx.Graph()
    # add nodes
    for u in unique_idx:
        coord_i = coord[coord[:, -1] == u][:, :-1]
        for i, xy in enumerate(coord_i):
            G.add_node(map_dict[u]+'_{}'.format(i), pos=(xy[1], xy[0]))
    # add edges
    for p1 in G.nodes():
        for p2 in G.nodes():
            distance = dist(G, G, p1, p2)
            if min_edge_length < distance < max_edge_length:
                G.add_edge(p1, p2)
    return G


def dist(G1: Type[nx.Graph], G2: Type[nx.Graph], p1: str, p2: str) -> float:
    """
    Calculates distances between nodes of a given graph(s)
    """
    return np.sqrt((G1.nodes[p1]['pos'][1]-G2.nodes[p2]['pos'][1])**2 +
                   (G1.nodes[p1]['pos'][0]-G2.nodes[p2]['pos'][0])**2)


def plot_graph(G: Type[nx.Graph],
               img: np.ndarray,
               fsize: Union[int, Tuple[int, int]] = 8,
               show_labels: bool = True,
               **kwargs: Union[int, str, float]) -> None:

    """
    Plots graph overlayed on the original image (raw or NN/VAE output)

    Args:
        G (networkx object): Graph object
        img (numpy array): 2D image (used to construct graph)
        fsize (int or tuple): figure size
        show_labels (bool): display node labels (e.g. C_1, C_13)
        **kwargs: additional plotting parameters
    """
    fsize = fsize if isinstance(fsize, tuple) else (fsize, fsize)
    plt.figure(figsize=fsize)
    pos = nx.get_node_attributes(G, 'pos')
    plt.imshow(img, origin="lower", cmap=kwargs.get("cmap", "gnuplot2"))
    nx.draw_networkx_nodes(
        G, pos=pos, nodelist=G.nodes(),
        node_size=kwargs.get("node_size", 30),
        node_color=kwargs.get("node_color", "#1f78b4"),
        alpha=kwargs.get("alpha", None))
    nx.draw_networkx_edges(
        G, pos, width=1,
        edge_color=kwargs.get("edge_color", "orange"),
        alpha=kwargs.get("alpha", None))
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=14, font_color='black')
    plt.show()


def filter_subgraphs_(coordinates: np.ndarray,
                      max_edge_length: int,
                      map_dict: Dict[int, str]) -> np.ndarray:
    """
    Filters atomic coordinates using connected subgraphs.

    Args:
        coordinates (ndarray):
            Atomic coordinates as a numpy array where the first 2 columns
            are x and y coordinates and the third columns is atomic class
        max_edge_length (int):
            Maximum graph edge length (nodes above this length will not be connected)
        map_dict (dict):
            dictionary which maps atomic classes from the NN output (dict keys)
            to strings corresponding to chemical elements (dict values)

    Returns:
        Filtered atomic coordinates
    """
    map_dict_inv = {v: k for (k, v) in map_dict.items()}
    G = construct_graph(coordinates, max_edge_length, map_dict)
    sub_graphs = list(G.subgraph(c).copy() for c in nx.connected_components(G))
    i = np.argmax([len(sg) for sg in sub_graphs])
    main_graph = sub_graphs[i]
    pos = nx.get_node_attributes(main_graph, 'pos')
    coordinates_filtered = []
    for k, c in pos.items():
        cls = map_dict_inv[k.split('_')[0]]
        c_arr = np.array([c[1], c[0], cls]).reshape(1, -1)
        coordinates_filtered.append(c_arr)
    coordinates_filtered = np.concatenate(coordinates_filtered)

    return coordinates_filtered


def filter_subgraphs(coordinates: Union[Dict[int, np.ndarray], np.ndarray],
                     max_edge_length: int,
                     map_dict: Dict[int, str]) -> Dict[int, np.ndarray]:
    """
    Filters atomic coordinates using connected subgraphs.

    Args:
        coordinates (dict or ndarray):
            Atomic coordinates (e.g. from the output of atomnet.predictor)
        max_edge_length (int):
            Maximum graph edge length (nodes above this length will not be connected)

    Returns:
        Filtered atomic coordinates
    """
    if isinstance(coordinates, np.ndarray):
        coordinates = {0: coordinates}
    coordinates_filtered_d = {}
    for k, coord in coordinates.items():
        coordinates_filtered_d[k] = filter_subgraphs_(
            coord, max_edge_length, map_dict)
    return coordinates_filtered_d


def find_all_cycles(G: nx.Graph,
                    min_cycle_len: int = 5,
                    max_cycle_len: int = 8
                    ) -> List[str]:
    """
    Finds all cycles in a graph
    """
    g_dir = nx.to_directed(G)
    rings = nx.simple_cycles(g_dir)

    rings_filt, rings_filt_s = [], []
    for r in rings:
        if min_cycle_len <= len(r) <= max_cycle_len:
            if sorted(r) not in rings_filt_s:
                rings_filt.append(r)
                rings_filt_s.append(sorted(r))
    return rings_filt_s


def adj_cycles(G: Type[nx.Graph],
               cycles: List[str],
               c: Tuple[float, float]
               ) -> List:
    """
    Find cycles (rings) containing a particular node
    """
    xc, yc = c
    pos = nx.get_node_attributes(G, 'pos')
    d_all, n_all = [], []
    for n, p in pos.items():
        d = np.sqrt((p[0] - xc)**2 + (p[1] - yc)**2)
        d_all.append(d)
        n_all.append(n)
    central_node = n_all[np.argmin(d_all)]
    nv = []
    for r in cycles:
        if central_node in r:
            nv.append(len(r))
    return nv
