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


'''
UNIT TESTING FOR VICTREE AND WILSONS WITH A SAMPLE GIVEN GRAPH G
'''



''' CREATE SAMPLE GRAPH G '''
# create the weighted DiGraph G
G = nx.DiGraph()
nodes_list = [0, 1, 2, 3]
weighted_edges = [(0, 1, 2), (1, 2, 3), (1, 3, 5), (2, 0, 7), (3, 2, 2)]
G.add_nodes_from(nodes_list)
G.add_weighted_edges_from(weighted_edges)

# list for the returned sampled trees
sampled_trees_victree = []
num_samples = 1000


''' VICTREE UNIT TESTING '''
# run the algorithm 1000 times and add the tree to sampled_trees
for i in range(num_samples):
    sampled_g, log_g = sample_arborescence_from_weighted_graph(G, debug=True)
    sampled_trees_victree.append(sampled_g)

equal_trees = get_equal_trees(sampled_trees)
# get the sample proportions of the diff graphs from the sample
sample_freq = np.ones(6)
for g, count in equal_trees.items():
  for graph in all_graphs:
    if set(g.edges()) == set(graph):
      sample_freq[all_graphs.index(graph)] = count
total = np.sum(sample_freq)
sample_freq = sample_freq / total

non_st = 0
for g, count in equal_trees.items():
  if len(g) < len(G):
    non_st += count
  elif len(g.edges()) != (len(G) - 1):
    non_st *= count
  elif not nx.is_directed_acyclic_graph(g):
      non_st += count
print("Frequencies of the sampled trees:", sample_freq)

# equal_trees is a dictionary: key is the graph, index is counts for that graph

print()
for g, count in equal_trees.items():
    print(f"Graph: {g.nodes()}, Edges: {g.edges()} Count: {count} Proportion: {count / (num_samples - non_st)}" )

'''
TESTING (PREDECESSOR):
  Wilson's algorithm builds reversed trees (i.e. roots as sinks).
  To suit our purposes, we consider predecessors during Wilson's, and
  subsequently reverse its output, so as to build unreversed trees.
'''
NUM_SAMPLES = 10000
sampled_trees_wilsons = {}

for i in range(NUM_SAMPLES):
    edges = WilsonTree(G)
    sampled_trees_wilsons[edges] = sampled_trees_wilsons.get(edges, 0) + 1

for g, count in sorted(sampled_trees.items()):
    print(f"Edges: {g}, Count: {count}, Proportion: {count / NUM_SAMPLES}")




''' UNIT TEST TRUE PROBABILIY CALCULATION '''

# list all edges of possible spanning trees
graph1_edges = [(0, 1), (1, 3), (3, 2)]
graph2_edges = [(1, 3), (1, 2), (2, 0)]
graph3_edges = [(0, 1), (1, 2), (1, 3)]
graph4_edges = [(0, 1), (3, 2), (2, 0)]
graph5_edges = [(1, 3), (3, 2), (2, 0)]
graph6_edges = [(2, 0), (0, 1), (1, 3)]
all_graphs = [graph1_edges,
              graph2_edges,
              graph3_edges,
              graph4_edges,
              graph5_edges,
              graph6_edges]

# calculate the weight --> weight = product of the edges
weight = np.zeros(6)
for i in range(len(all_graphs)):
    for edge in all_graphs[i]:
      weight[i] += G[edge[0]][edge[1]]['weight']
total_weight = np.sum(weight)

# get the true proportions of the diff graphs
true_freq = weight / total_weight

for g, freq in sorted(zip(all_graphs, true_freq)):
    print(f"Graph: {g}, Probability: {freq}")