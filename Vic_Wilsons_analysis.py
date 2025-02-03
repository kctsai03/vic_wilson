import copy
import logging
import random
import pandas as pd

import networkx as nx
import numpy as np

import matplotlib
from networkx import maximum_spanning_arborescence

import matplotlib.pyplot as plt

from networkx.drawing.nx_pydot import graphviz_layout
from typing import Tuple

import time
from Vic_Wilsons_analysis_helper import generate_random_digraph
from VicTree_algorithm import VicTree
from Wilsons_algorithm import WilsonTree
from VicTree_helper import isValidTree


''' ALGORITHMS ANALYSIS '''

NUM_SAMPLES = 100 # Number of times we sample from the graph
NUM_TRIALS = 2 # Number of distinct graphs with random weights generated per nnumber of nodes

NODE_SIZES = list(range(5, 101, 5)) # Testing graphs of node size 5, 10, 15, ... 100

NUM_TEST_CASES = len(NODE_SIZES) # Number of graph sizes that we're testing

print(f"Nodes in each Test Case:    {NODE_SIZES}")
print(f"Trials per Test Case:       {NUM_TRIALS}")
print(f"Samples per Trial:          {NUM_SAMPLES}\n")

vicTree_times = []
wilsonTree_times = []

distances = []

# Each loop is a certain graph size (5, 10, 15, ... 100)
for test_case in range(NUM_TEST_CASES):
    node_size = NODE_SIZES[test_case]

    print(f"**** # Nodes: {node_size} ****")

    # Average sample time per trial (per graph)
    vicTree_trial_times = []
    wilsonTree_trial_times = []

    # L1 Norm Distance per trial
    trial_distances = []

    # Generate a random graph NUM_TRIALS times (2 times)
    for i in range(NUM_TRIALS):
        # Create the random Digraph (complete graph with random weights)
        G = generate_random_digraph(node_size)

        # Stores all sample times (len = NUM_SAMPLES)
        vicTree_sample_times = []
        wilsonTree_sample_times = []

        # Stores count of all sampled trees
        vicTrees = {}
        wilsonTrees = {}
        invalid_vicTrees = 0

        for _ in range(NUM_SAMPLES):

            # VICTREE
            start = time.perf_counter()
            vicTree_graph, vicTree = VicTree(G)
            vic_time = time.perf_counter() - start

            # Update sample tree counter and time array
            vicTree_sample_times.append(vic_time)
            if isValidTree(G, vicTree_graph):
                vicTrees[vicTree] = vicTrees.get(vicTree, 0) + 1
            else: invalid_vicTrees += 1

            # WILSONTREE
            start = time.perf_counter()
            wilsonTree = WilsonTree(G)
            wilson_time = time.perf_counter() - start

            # Update sample tree counter and time array
            wilsonTree_sample_times.append(wilson_time)
            wilsonTrees[wilsonTree] = wilsonTrees.get(wilsonTree, 0) + 1

        # Append average sample time for this trial (this graph)
        vicTree_trial_times.append(np.mean(vicTree_sample_times))
        wilsonTree_trial_times.append(np.mean(wilsonTree_sample_times))

        # Compute L1 Norm Distance for this trial

        sampled_wilsonTrees = wilsonTrees.keys()
        tree_cnts = [wilsonTrees[tree] for tree in sampled_wilsonTrees]

        wilson_dist = np.array(tree_cnts) / NUM_SAMPLES

        # If all VicTree samples were invalid, then
        # set VicTree proportions to infinity
        # Otherwise, set VicTree proportions for valid trees.
        length = len(wilson_dist)
        vic_dist = np.full(length, np.inf)
        vicTree_samples = NUM_SAMPLES - invalid_vicTrees
        if vicTree_samples:
            tree_cnts = [vicTrees.get(tree, 0) for tree in sampled_wilsonTrees]
            vic_dist = np.array(tree_cnts) / vicTree_samples

        l1_norm_distance = np.sum(np.abs(vic_dist - wilson_dist))
        trial_distances.append(l1_norm_distance)

    # Append average trial time
    vicTree_times.append(np.mean(vicTree_trial_times))
    wilsonTree_times.append(np.mean(wilsonTree_trial_times))

    print(f"VicTree:            {vicTree_times[-1]:.5f} s")
    print(f"WilsonTree:         {wilsonTree_times[-1]:.5f} s")

    # Append average trial L1 Norm
    distances.append(np.mean(trial_distances))

    print(f"L1 Norm: {distances[-1]:.5f}\n")