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

NUM_SAMPLES = 20 # Number of times we sample from the graph
NUM_TRIALS = 1 # Number of distinct graphs with random weights generated per nnumber of nodes

NODE_SIZES = list(range(5, 101, 5)) # Testing graphs of node size 5, 10, 15, ... 100
#NODE_SIZES = [10, 20, 40, 80, 160]

NUM_TEST_CASES = len(NODE_SIZES) # Number of graph sizes that we're testing

print(f"Nodes in each Test Case:    {NODE_SIZES}")
print(f"Trials per Test Case:       {NUM_TRIALS}")
print(f"Samples per Trial:          {NUM_SAMPLES}\n")

vicTree_times = []
wilsonTree_times = []
wilsonTree_expected_calls = []

distances = []

# Each loop is a certain graph size (5, 10, 15, ... 100)
for test_case in range(NUM_TEST_CASES):
    node_size = NODE_SIZES[test_case]

    print(f"**** # Nodes: {node_size} ****")

    # Average sample time per trial (per graph)
    vicTree_trial_times = []
    wilsonTree_trial_times = []
    wilsonTree_expected_calls_inner = []

    # L1 Norm Distance per trial
    trial_distances = []

    # Generate a random graph NUM_TRIALS times (2 times)
    for i in range(NUM_TRIALS):
        # Create the random Digraph (complete graph with random weights)
        G = generate_random_digraph(node_size)

        # Stores all sample times (len = NUM_SAMPLES)
        vicTree_sample_times = []
        wilsonTree_sample_times = []
        wilsonTree_num_calls = []

        # Stores count of all sampled trees
        vicTrees = {}
        wilsonTrees = {}
        invalid_vicTrees = 0

        num_samples = 1 if node_size >= 1000 else NUM_SAMPLES
        for _ in range(num_samples):

            # VICTREE
            # start = time.perf_counter()
            # vicTree_graph, vicTree = VicTree(G)
            # vic_time = time.perf_counter() - start

            # # Update sample tree counter and time array
            # vicTree_sample_times.append(vic_time)
            # if isValidTree(G, vicTree_graph):
            #     vicTrees[vicTree] = vicTrees.get(vicTree, 0) + 1
            # else: invalid_vicTrees += 1

            # WILSONTREE
            start = time.perf_counter()
            wilsonTree, num_rand_pred_calls = WilsonTree(G)
            wilson_time = time.perf_counter() - start

            # Update sample tree counter and time array
            wilsonTree_sample_times.append(wilson_time)
            wilsonTree_num_calls.append(num_rand_pred_calls)
            wilsonTrees[wilsonTree] = wilsonTrees.get(wilsonTree, 0) + 1

        # Append average sample time for this trial (this graph)
        # vicTree_trial_times.append(np.mean(vicTree_sample_times))
        wilsonTree_trial_times.append(np.mean(wilsonTree_sample_times))
        wilsonTree_expected_calls_inner.append(np.mean(wilsonTree_num_calls))
        # Compute L1 Norm Distance for this trial

        sampled_wilsonTrees = wilsonTrees.keys()
        tree_cnts = [wilsonTrees[tree] for tree in sampled_wilsonTrees]

        wilson_dist = np.array(tree_cnts) / num_samples

        # If all VicTree samples were invalid, then
        # set VicTree proportions to infinity
        # Otherwise, set VicTree proportions for valid trees.
        length = len(wilson_dist)
        vic_dist = np.full(length, np.inf)
        vicTree_samples = num_samples - invalid_vicTrees
        if vicTree_samples: 
            tree_cnts = [vicTrees.get(tree, 0) for tree in sampled_wilsonTrees]
            vic_dist = np.array(tree_cnts) / vicTree_samples

        l1_norm_distance = np.sum(np.abs(vic_dist - wilson_dist))
        trial_distances.append(l1_norm_distance)

    # Append average trial time
    vicTree_times.append(np.mean(vicTree_trial_times))
    wilsonTree_times.append(np.mean(wilsonTree_trial_times))
    wilsonTree_expected_calls.append(np.mean(wilsonTree_expected_calls_inner))

    # print(f"Number of invalid VicTree samples: {invalid_vicTrees}")
    print(f"VicTree:            {vicTree_times[-1]:.5f} s")
    print(f"WilsonTree:         {wilsonTree_times[-1]:.5f} s")
    print(f"WilsonTree Calls:   {wilsonTree_expected_calls[-1]:.2f}")

    # Append average trial L1 Norm
    distances.append(np.mean(trial_distances))

    print(f"L1 Norm: {distances[-1]:.5f}\n")

results_df = pd.DataFrame({
    "Nodes": NODE_SIZES,
    "VicTree Time": vicTree_times,
    "Expected WilsonTree Time": wilsonTree_times,
    "Expected WilsonTree Calls": wilsonTree_expected_calls,
    "L1 Norm Distance": distances
})

results_df['Samples'] = NUM_SAMPLES
results_df.to_csv("results.csv", index=False)

print(results_df)
print(wilsonTree_times)