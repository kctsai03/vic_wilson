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



''' HELPER FUNCTIONS FOR CALCULATING SAMPLE PROBABILITIES '''

## GROUP THE TREES THAT ARE EQUIVALENT AND GET THEIR COUNTS

# check if two graphs are the same
def are_equal(g1, g2):
    return set(g1.edges()) == set(g2.edges())

# group the graphs by equality and their counts
def get_equal_trees(graphs):
    # dictionary called equal_groups that has the key as the graph
    # and the index as the number of times we sampled this graph
    equal_groups = {}
    for g in graphs:
        # Check which existing graph g is the same as + add 1 if true
        graph_exists = False
        for key in list(equal_groups.keys()):
            if are_equal(g, key):
                equal_groups[key] += 1
                graph_exists = True
                break
        if not graph_exists:
            # If it doesn't exist in dict, add a new entry
            equal_groups[g] = 1
    return equal_groups

''' HELPER FUNCTIONS FOR ALGORITHMS ANALYSIS '''

# Random digraph generator with specified node counts - complete graph with random edge weights
def generate_random_digraph(num_nodes):
    G = nx.DiGraph()

    # Add num_nodes nodes
    G.add_nodes_from(range(num_nodes))

    # Add all the possible directed edges to make it a complete graph
    for u in G.nodes():
      for v in G.nodes():
        if u != v:
            weight = 1 #np.random.poisson(3)
            G.add_edge(u, v, weight = weight)

    return G

def drawGraph(G):
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    nx.draw(G, ax = axes[0], with_labels = True, font_weight = 'bold')
    plt.show()

def printGraph(G):
    for u, v, data in G.edges(data=True):
        print(f"({u}, {v}, weight={data['weight']})")

def Hellinger_Distance(P, Q):
    return np.sqrt(0.5 * np.sum((np.sqrt(P) - np.sqrt(Q))**2))


