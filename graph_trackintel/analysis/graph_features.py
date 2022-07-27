import networkx as nx
import numpy as np
import os
import time
from numpy.core.fromnumeric import sort
import pandas as pd
from scipy.optimize import curve_fit
import trackintel as ti


def random_walk(graph, return_resets=False, random_walk_iters=1000):
    # start at node with highest degree
    all_degrees = np.array(graph.out_degree())
    start_node = all_degrees[np.argmax(all_degrees[:, 1]), 0]
    current_node = start_node

    # check if we can walk somewhere at all
    if np.max(all_degrees[:, 1]) == 0:
        if return_resets:
            return [], []
        else:
            return []

    encountered_locations = [current_node]
    number_of_walks = 0
    # keep track of when we reset the position to home --> necessary for cycle count
    reset_to_home = []
    for step in range(random_walk_iters):
        # get out neighbors with corresponding transition number
        neighbor_edges = graph.out_edges(current_node, data=True)
        # check if we are at a dead end OR if we get stuck at one node and only make cycles of len 1 there
        at_dead_end = len(neighbor_edges) == 0
        at_inf_loop = len(neighbor_edges) == 1 and [n[1] for n in neighbor_edges][0] == current_node
        # or we have a transition weight of 0
        at_zero_transition = len(neighbor_edges) > 0 and np.sum(np.array([n[2]["weight"] for n in neighbor_edges])) == 0
        if at_dead_end or at_inf_loop or at_zero_transition:
            # increase number of walks counter
            number_of_walks += 1
            # reset current node
            current_node = start_node
            neighbor_edges = graph.out_edges(current_node, data=True)
            # we are again at the start node
            encountered_locations.append(start_node)
            # reset location is step + 2 because in encountered_locations
            prev_added = len(reset_to_home)
            reset_to_home.append(step + 1 + prev_added)

        out_weights = np.array([n[2]["weight"] for n in neighbor_edges])
        out_weights = out_weights[~np.isnan(out_weights)]
        out_probs = out_weights / np.sum(out_weights)
        next_node = [n[1] for n in neighbor_edges]
        if np.any(np.isnan(out_probs)):
            print(out_probs, out_weights, at_zero_transition)
        # draw one neightbor randomly, weighted by transition count
        current_node = np.random.choice(next_node, p=out_probs)
        # print("used node with weight", out_weights[next_node.index(current_node)])
        # collect node (features)
        encountered_locations.append(current_node)

    if return_resets:
        return encountered_locations, reset_to_home
    # simply save the encountered nodes here
    return encountered_locations


def home_cycle_lengths(graph):
    """Get cycle lengths of journeys (starting and ending at home"""
    nodes_on_rw, resets = random_walk(graph, return_resets=True)
    if len(nodes_on_rw) == 0:
        return []
    assert (
        len(resets) == 0 or len(np.unique(np.array(nodes_on_rw)[resets])) == 1
    ), "reset indices must always be a home node"
    cycle_lengths = []
    home_node = nodes_on_rw[0]
    at_home = np.where(np.array(nodes_on_rw) == home_node)[0]
    for i in range(len(at_home) - 1):
        if at_home[i + 1] not in resets:
            cycle_lengths.append(at_home[i + 1] - at_home[i])
    return cycle_lengths


def journey_length(graph):
    cycle_lengths = home_cycle_lengths(graph)
    if len(cycle_lengths) == 0:
        return np.nan
    return np.mean(cycle_lengths)


def transitions(graph):
    """Get all edge weights"""
    transition_counts = [edge[2]["weight"] for edge in graph.edges(data=True)]
    return transition_counts


def get_point_dist(p1, p2, crs_is_projected=False):
    if crs_is_projected:
        dist = p1.distance(p2)
    else:
        dist = ti.geogr.point_distances.haversine_dist(p1.x, p1.y, p2.x, p2.y)[0]
    return dist


def weighted_dists(graph):
    dist_list = []
    for (u, v, data) in graph.edges(data=True):
        loc_u = graph.nodes[u]["center"]
        loc_v = graph.nodes[v]["center"]
        weight = data["weight"]
        dist = get_point_dist(loc_u, loc_v, crs_is_projected=False)
        dist_list.extend([dist for _ in range(int(weight))])
    return dist_list


def median_trip_distance(graph):
    dist_list = weighted_dists(graph)
    if len(dist_list) == 0:
        print("dist list nan")
        for (u, v, data) in graph.edges(data=True):
            print(u, v, data["weight"])
        return np.nan
    return np.median(dist_list)


def highest_decile_distance(graph):
    dist_list = weighted_dists(graph)
    if len(dist_list) == 0:
        print("dist list nan")
        for (u, v, data) in graph.edges(data=True):
            print(u, v, data["weight"])
        return np.nan
    return np.quantile(dist_list, 0.9)


def get_degrees(graph, mode="out"):
    """
    Degree distribution of graph
    """
    # one function for in, out and all degrees
    use_function = {"all": graph.degree(), "out": graph.out_degree(), "in": graph.in_degree()}
    degrees = list(dict(use_function[mode]).values())
    return degrees


def func_simple_powerlaw(x, beta):
    return x ** (-beta)


def fit_powerlaw(item_list):
    if len(item_list) == 0 or np.sum(item_list) == 0:
        return 0

    sorted_vals = sorted(item_list)[::-1]
    # get relative probability
    normed_vals = sorted_vals / np.sum(sorted_vals)
    # Normalize by first value! Because: power function 1/x^beta always passes through (1,1) - we want to fit this
    normed_vals = normed_vals / normed_vals[0]
    params, _ = curve_fit(
        func_simple_powerlaw, np.arange(len(normed_vals)) + 1, normed_vals, maxfev=3000, bounds=(0, 5)
    )

    # Prev version: with cutoff and no normalization
    # sorted_vals = (sorted(item_list)[::-1])[:cutoff]
    # normed_degrees = sorted_vals / np.sum(sorted_vals)
    return params[0]


def degree_beta(graph):
    degrees = np.array(list(dict(graph.out_degree()).values()))
    return fit_powerlaw(degrees)


def transition_beta(graph):
    transitions = np.array([edge[2]["weight"] for edge in graph.edges(data=True)])
    beta = fit_powerlaw(transitions)
    if beta < 0.05:
        return np.nan
    return beta


def hub_size(graph, thresh=0.8):
    nodes_on_rw = random_walk(graph)
    if len(nodes_on_rw) < 2:
        return np.nan
    _, counts = np.unique(nodes_on_rw, return_counts=True)
    sorted_counts = np.sort(counts)[::-1]
    cumulative_counts = np.cumsum(sorted_counts)
    # number of nodes needed to cover thresh times the traffic
    nodes_in_core = np.where(cumulative_counts > thresh * np.sum(counts))[0][0] + 1
    return nodes_in_core / np.sqrt(graph.number_of_nodes())


def sp_length(graph, max_len=10):
    """
    Returns discrete histogram of path length occurences
    """
    all_sp = nx.floyd_warshall(graph)
    all_sp_lens = [v for sp_dict in all_sp.values() for v in list(sp_dict.values())]
    sp_len_counts, _ = np.histogram(all_sp_lens, bins=np.arange(1, max_len + 1))
    return sp_len_counts


def eigenvector_centrality(graph1):
    """
    Compute EV centrality of each node
    Returns:
        Dictionary of centralities per node
    """
    if isinstance(graph, nx.classes.multidigraph.MultiDiGraph):
        graph = nx.DiGraph(graph)
    try:
        centrality = nx.eigenvector_centrality_numpy(graph)
        return centrality
    except:
        return {0: 0}


def betweenness_centrality(graph):
    """
    Compute betweenness centrality of each node
    Returns:
        Dictionary of centralities per node
    """
    if isinstance(graph, nx.classes.multidigraph.MultiDiGraph):
        graph = nx.DiGraph(graph)
    centrality = nx.algorithms.centrality.betweenness_centrality(graph)
    return centrality


def centrality_dist(graph, max_centrality=1, centrality_fun=betweenness_centrality):
    """
    Compute distribution of centrality in fixed size histogram vector
    Returns:
        1D np array of length 10
    """
    centrality = centrality_fun(graph)
    centrality_vals = list(centrality.values())
    centrality_hist, _ = np.histogram(centrality_vals, bins=np.arange(0, max_centrality + 0.1, 0.1))
    return centrality_hist
