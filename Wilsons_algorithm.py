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
from Wilsons_helper import AddSelfLoops, RandomSuccessor, RandomPredecessor, Chance


'''
WILSON'S ALGORITHM:
  WilsonTreeWithRoot() builds spanning trees from a graph with a specified root.
  WilsonTree() builds a random spanning tree from a graph.
  Translated from source: https://dl.acm.org/doi/10.1145/237814.237880
'''

'''
INPUT: A vertex root r and graph G
OUTPUT: A random spanning tree named tree rooted at r
        where node i points to tree[i]
'''
def WilsonTreeWithRoot(G, r):
    n = len(G)

    Next = [None] * n
    InTree = [False] * n
    InTree[r] = True

    for i in range(n):
        u = i
        while not InTree[u]:
            Next[u] = RandomSuccessor(G, u)
            u = Next[u]
        u = i
        while not InTree[u]:
            InTree[u] = True
            u = Next[u]

    return Next

'''
INPUT: A graph G
OUTPUT: A random spanning tree as a tuple of edges as node pairs.
'''
def WilsonTree(G) -> tuple:
    n = len(G.nodes)

    def Attempt(epsilon):
        Next = [None] * n
        InTree = [False] * n
        num_roots = 0

        for i in range(n):
            u = i
            while not InTree[u]:
                if Chance(epsilon):
                    Next[u] = None
                    InTree[u] = True
                    num_roots += 1
                    if num_roots > 1:
                        return None
                else:
                    Next[u] = RandomPredecessor(G, u)
                    u = Next[u]
            u = i
            while not InTree[u]:
                InTree[u] = True
                u = Next[u]

        return Next

    G = AddSelfLoops(G)
    epsilon = 1
    tree = None

    while tree == None:
        epsilon = epsilon / 2
        tree = Attempt(epsilon)

    edges = tuple(sorted((b, a) for a, b in enumerate(tree) if b is not None))

    return edges