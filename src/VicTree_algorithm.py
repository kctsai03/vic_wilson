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
from VicTree_helper import get_ordered_arcs, new_graph_force_arc, new_graph_with_arcs, new_graph_without_arc, _sample_feasible_arc, isValidTree

'''
VICTREE ALGORITHM
  Source: https://github.com/Lagergren-Lab/victree/blob/main/src/sampling/laris.py
'''

def sample_arborescence_from_weighted_graph(graph: nx.DiGraph,
                                            root: int = 0,
                                            debug: bool = False,
                                            order_method='random',
                                            temp=1.):
    s = nx.DiGraph()
    s.add_node(root)
    # copy graph so to remove arcs which shouldn't be considered
    # while S gets constructed
    tempered_graph = copy.deepcopy(graph)
    if temp != 1.:
        for e in tempered_graph.edges:
            tempered_graph.edges[e]['weight'] = tempered_graph.edges[e]['weight'] * (1/temp)
    skimmed_graph = copy.deepcopy(graph)
    log_W = torch.tensor(nx.to_numpy_array(skimmed_graph)) * (1 / temp)
    # counts how many times arborescences cannot be found
    miss_counter = 0
    log_g = 0.
    candidate_arcs = get_ordered_arcs(skimmed_graph, method=order_method)

    while s.number_of_edges() < graph.number_of_nodes() - 1:
        # new graph with all s arcs
        g_with_s = new_graph_with_arcs(s.edges, tempered_graph)
        num_candidates_left = len(candidate_arcs)

        feasible_arcs = []
        for u, v in candidate_arcs:

            g_w = new_graph_force_arc(u, v, g_with_s)
            g_wo = new_graph_without_arc(u, v, g_with_s)
            t_w = t_wo = nx.DiGraph()  # empty graph
            try:
                t_w = maximum_spanning_arborescence(g_w, preserve_attrs=True)
                # save max feasible arcs
                feasible_arcs.append((u, v, tempered_graph.edges[u, v]['weight']))
                t_wo = maximum_spanning_arborescence(g_wo, preserve_attrs=True)
            except nx.NetworkXException as nxe:
                # go to next arc if, once some arcs are removed, no spanning arborescence exists
                miss_counter += 1
                # if miss_counter in [100, 1000, 2000, 10000]:
                    # logging.log(logging.WARNING, f'LArIS num misses: {miss_counter}')
                num_candidates_left -= 1
                if num_candidates_left > 0:
                    continue

            if num_candidates_left == 0 and len(feasible_arcs) > 0:
                # no arc allows for both t_w and t_wo to exist
                # must choose one of the feasible ones (for which t_w exists)
                # obliged choice -> theta = 1
                # theta = torch.tensor(1.)
                # randomize selection based on weights
                (u, v), theta = _sample_feasible_arc(feasible_arcs)
            elif num_candidates_left == 0:
                # heuristic: reset s
                logging.debug("No more candidates in LArIS tree reconstruction. Restarting algorithm.")
                s = nx.DiGraph()
                s.add_node(root)
                # skimmed_graph = copy.deepcopy(graph)
                break
            else:
                if t_w.number_of_nodes() == 0 or t_wo.number_of_nodes() == 0:
                    raise Exception('t_w and t_wo are empty but being called')
                w_Tw = torch.tensor([log_W[u, v] for (u, v) in t_w.edges()]).sum()
                w_To = torch.tensor([log_W[u, v] for (u, v) in t_wo.edges()]).sum()
                theta = torch.exp(w_Tw - torch.logaddexp(w_Tw, w_To))
                # theta2 = torch.exp(t_w.size(weight='weight') -
                #                   torch.logaddexp(t_w.size(weight='weight'), t_wo.size(weight='weight')))

            if torch.rand(1) < theta:
                s.add_edge(u, v, weight=graph.edges[u, v]['weight'])
                # remove all incoming arcs to v (including u,v)
                # skimmed_graph.remove_edges_from(graph.in_edges(v))
                # skimmed_graph.remove_edges_from([(v, u)])
                candidates_to_remove = list(graph.in_edges(v))
                candidates_to_remove.append((v, u))
                candidate_arcs = [a for a in candidate_arcs if a not in candidates_to_remove]
                # prob of sampling the tree: prod of bernoulli trials
                log_g += torch.log(theta)
                # go to while and check if s is complete
                break

    return s, log_g

def VicTree(G):
    s, _ = sample_arborescence_from_weighted_graph(G)
    return s, tuple(sorted(s.edges()))