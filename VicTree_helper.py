import copy
import logging
import random
import pandas as pd

import networkx as nx
import numpy as np
import torch

import matplotlib
from networkx import maximum_spanning_arborescence

import matplotlib.pyplot as plt

from networkx.drawing.nx_pydot import graphviz_layout
from typing import Tuple

import time



"""
VICTREE ALGORITHM: (helper functions)
  Code related to the Labeled Arborescence Importance Sampling (LArIS).
SOURCE:
  https://github.com/Lagergren-Lab/victree/blob/main/src/sampling/laris.py
"""

## CREATE NEW GRAPHS
def new_graph_force_arc(u, v, graph: nx.DiGraph) -> nx.DiGraph:
    # remove all incoming arcs for v except u,v
    arcs_to_v_no_u = [(a, b) for a, b in graph.in_edges(v) if a != u]
    new_graph = copy.deepcopy(graph)
    new_graph.remove_edges_from(arcs_to_v_no_u)
    return new_graph

def new_graph_with_arcs(ebunch, graph: nx.DiGraph) -> nx.DiGraph:
    new_graph = copy.deepcopy(graph)
    for u, v in ebunch:
        # remove all incoming arcs for v except u,v
        arcs_to_v_no_u = [(a, b) for a, b in graph.in_edges(v) if a != u]
        new_graph.remove_edges_from(arcs_to_v_no_u)
    return new_graph

def new_graph_without_arc(u, v, graph: nx.DiGraph) -> nx.DiGraph:
    new_graph = copy.deepcopy(graph)
    new_graph.remove_edge(u, v)
    return new_graph

def _sample_feasible_arc(weighted_arcs):
    # weighted_arcs is a list of 3-tuples (u, v, weight)
    # weights are negative: need transformation
    unnorm_probs = 1 / (-torch.tensor([w for u, v, w in weighted_arcs], dtype=torch.float32))
    probs = unnorm_probs / unnorm_probs.sum()
    c = np.random.choice(np.arange(len(weighted_arcs)), p=probs.numpy())
    return weighted_arcs[c][:2], probs[c]

def get_ordered_arcs(graph: nx.DiGraph, method='random'):
    edges_list = list(graph.edges)
    if method == 'random':
        order = np.random.permutation(len(edges_list))
        ordered_edges = []
        for i in order:
            ordered_edges.append(edges_list[i])
    elif method == 'edmonds':
        mst_graph = nx.maximum_spanning_arborescence(graph).edges
        mst_arcs = list(mst_graph)
        graph.remove_edges_from(mst_arcs)
        ordered_edges = mst_arcs + list(graph.edges)
    else:
        raise ValueError(f'Method {method} is not available.')

    return ordered_edges

def isValidTree(G, tree):
    if not tree:
        return False

    a = tree.size() == len(G) - 1
    b = nx.is_directed_acyclic_graph(tree)
    c = len(G) == len(tree)

    return a and b and c