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

# function to run the given algorithm (victree or wilsontree) on given graph and number of samples
def run_algorithm(algorithm, G, num_samples):
    sample_times = []
    num_tree_calls = []
    trees = {}
    invalid_trees = 0
    for _ in range(num_samples):
        start = time.perf_counter()
        if algorithm == 'VicTree':
            vicTree_graph, vicTree = VicTree(G)
            sample_time = time.perf_counter() - start
            sample_times.append(sample_time)
            if isValidTree(G, vicTree_graph):
                trees[vicTree] = trees.get(vicTree, 0) + 1
            else: invalid_trees += 1

        elif algorithm == 'WilsonTree':
            wilsonTree, num_rand_pred_calls = WilsonTree(G)
            sample_time = time.perf_counter() - start
            sample_times.append(sample_time)
            trees[wilsonTree] = trees.get(wilsonTree, 0) + 1
            num_tree_calls.append(num_rand_pred_calls)
    return sample_times, trees, invalid_trees, num_tree_calls

# function to calculate the L1 norm distance between the two distributions
def l1_norm(wilsonTrees, vicTrees):
    # Find all the different graphs that WilsonTrees found 
    # and the number of times they were found to calculate the proportion
    sampled_wilsonTrees = wilsonTrees.keys()
    tree_cnts = [wilsonTrees[tree] for tree in sampled_wilsonTrees]
    wilson_dist = np.array(tree_cnts) / sum(wilsonTrees.values())

    # If all VicTree samples were invalid, then
    # set VicTree proportions to infinity
    # Otherwise, set VicTree proportions for valid trees.
    length = len(wilson_dist)
    vic_dist = np.full(length, np.inf)
    vicTree_samples = sum(vicTrees.values())
    if vicTree_samples: 
        tree_cnts = [vicTrees.get(tree, 0) for tree in sampled_wilsonTrees]
        vic_dist = np.array(tree_cnts) / sum(vicTrees.values())
    return np.sum(np.abs(vic_dist - wilson_dist))


# function to run the analysis for a certain number of test cases and node sizes
def vic_wilson_analysis(NUM_TEST_CASES, NODE_SIZES, NUM_TRIALS, NUM_SAMPLES, SMALL_NUM_SAMPLES):
    vicTree_times = []
    wilsonTree_times = []
    wilsonTree_expected_calls = []
    invalid_vicTree_samples = []

    distances = []  

    # Each loop is a certain graph size (5, 10, 15, ... 100)
    for test_case in range(NUM_TEST_CASES):
        node_size = NODE_SIZES[test_case]

        print(f"**** # Nodes: {node_size} ****")

        # Average sample time per trial (per graph)
        vicTree_trial_times = []
        wilsonTree_trial_times = []
        wilsonTree_expected_calls_inner = []
        invalid_vicTree_samples_inner = []

        # L1 Norm Distance per trial
        trial_distances = []

        # Generate a random graph NUM_TRIALS times (2 times)
        for i in range(NUM_TRIALS):
            # Create the random Digraph (complete graph with random weights)
            G = generate_random_digraph(node_size)
            num_samples = SMALL_NUM_SAMPLES if node_size >= 20 else NUM_SAMPLES

            # vic_sample_times = all the sample times for this trial
            # vicTrees = counts for all sampled trees
            # invalid_trees = number of invalid trees
            vic_sample_times, vicTrees, invalid_trees, _ = run_algorithm('VicTree', G, num_samples)

            # wilson_sample_times = all the sample times for this trial
            # wilsonTrees = counts for all sampled trees
            # wilsonTree_num_calls = number of random predecessor calls
            wilson_sample_times, wilsonTrees, _, wilsonTree_num_calls = run_algorithm('WilsonTree', G, NUM_SAMPLES)

            # Append average sample time for this trial (this graph) for VicTree and WilsonTree
            vicTree_trial_times.append(np.mean(vic_sample_times))
            wilsonTree_trial_times.append(np.mean(wilson_sample_times))

            # Append number of calls to random predecessor for WilsonTree
            wilsonTree_expected_calls_inner.append(np.mean(wilsonTree_num_calls))

            # Append number of invalid victrees to invalid_vicTree_samples
            invalid_vicTree_samples_inner.append(invalid_trees)


            # Compute L1 Norm Distance for this trial
            l1_norm_distance = l1_norm(wilsonTrees, vicTrees)
            trial_distances.append(l1_norm_distance)

        # Append average trial time
        vicTree_times.append(np.mean(vicTree_trial_times))
        wilsonTree_times.append(np.mean(wilsonTree_trial_times))
        wilsonTree_expected_calls.append(np.mean(wilsonTree_expected_calls_inner))
        invalid_vicTree_samples.append(np.mean(invalid_vicTree_samples_inner))

        # Append average trial L1 Norm
        distances.append(np.mean(trial_distances))

        print(f"Number of invalid VicTree samples: {invalid_vicTree_samples[-1]}")
        print(f"VicTree:            {vicTree_times[-1]:.5f} s")
        print(f"WilsonTree:         {wilsonTree_times[-1]:.5f} s")
        print(f"WilsonTree Calls:   {wilsonTree_expected_calls[-1]:.2f}")
        print(f"L1 Norm: {distances[-1]:.5f}\n")

    victree_sample_size = [SMALL_NUM_SAMPLES if size > 20 else NUM_SAMPLES for size in NODE_SIZES]

    results_df = pd.DataFrame({
        "Nodes": NODE_SIZES,
        "VicTree Time": vicTree_times,
        "Expected WilsonTree Time": wilsonTree_times,
        "Expected WilsonTree Calls": wilsonTree_expected_calls,
        "Invalid VicTree Samples": invalid_vicTree_samples,
        "VicTree Samples": victree_sample_size,
        "WilsonTree Samples": NUM_SAMPLES,
        "L1 Norm Distance": distances
    })
    return results_df



NUM_SAMPLES = 1000 # Number of times we sample from the graph
SMALL_NUM_SAMPLES = 1 
NUM_TRIALS = 2 # Number of distinct graphs with random weights generated per number of nodes

NODE_SIZES = list(range(5, 101, 5)) # Testing graphs of node size 5, 10, 15, ... 100


NUM_TEST_CASES = len(NODE_SIZES) # Number of graph sizes that we're testing

print(f"Nodes in each Test Case:    {NODE_SIZES}")
print(f"Trials per Test Case:       {NUM_TRIALS}")
print(f"Samples per Trial:          {NUM_SAMPLES}\n")

results_df = vic_wilson_analysis(NUM_TEST_CASES, NODE_SIZES, NUM_TRIALS, NUM_SAMPLES, SMALL_NUM_SAMPLES)
results_df.to_csv("results.csv", index=False)

print(results_df)
