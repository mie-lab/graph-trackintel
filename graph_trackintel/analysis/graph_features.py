import networkx as nx
import numpy as np
import os
import time
from numpy.core.fromnumeric import sort
import pandas as pd
from scipy.optimize import curve_fit
import trackintel as ti
from warnings import warn
import copy
import powerlaw


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
    for u, v, data in graph.edges(data=True):
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
        for u, v, data in graph.edges(data=True):
            print(u, v, data["weight"])
        return np.nan
    return np.median(dist_list)


def highest_decile_distance(graph):
    dist_list = weighted_dists(graph)
    if len(dist_list) == 0:
        print("dist list nan")
        for u, v, data in graph.edges(data=True):
            print(u, v, data["weight"])
        return np.nan
    return np.quantile(dist_list, 0.9)


def get_degrees(graph, mode="out", sort_degrees=False, norm=None, weight=None):
    """
    Degree distribution of graph

    Parameters
    ----------
    graph: Networkx graph
    mode: str
        Can be {"all", "out", "in"}. The type of node degree to consider. Default is "out"
    sort_degrees: Boolean
        the degree distribution is sorted in descending order

    Returns
        list of node degrees
    -------

    """
    # one function for in, out and all degrees
    use_function = {
        "all": graph.degree(weight=weight),
        "out": graph.out_degree(weight=weight),
        "in": graph.in_degree(weight=weight),
    }
    degrees = copy.copy(list(dict(use_function[mode]).values()))

    if sort_degrees:
        degrees = sorted(degrees)[::-1]

    if norm == "max":
        degrees = list(map(lambda x: x / max(degrees), degrees))
    elif norm == "sum":
        degrees = list(map(lambda x: x / sum(degrees), degrees))

    return degrees


def get_edge_weights(graph, sort_weights=False, norm=None, weight="weight"):
    """
    Returns list of edge weight

    Parameters
    ----------
    graph: Networkx graph
    sort_degrees: Boolean
        edge weights are sorted in descending order

    Returns
        list of edge weights
    -------

    """
    # todo: what should the default for "weight" be? None?
    # one function for in, out and all degrees
    edge_weights = np.array([edge[2][weight] for edge in graph.edges(data=True)])

    if sort_weights:
        edge_weights = sorted(edge_weights)[::-1]

    if norm == "max":
        edge_weights = list(map(lambda x: x / max(edge_weights), edge_weights))
    elif norm == "sum":
        edge_weights = list(map(lambda x: x / sum(edge_weights), edge_weights))

    return edge_weights


def fit_degree_dist_power_law(graph, mode="in", weight=None, fit_kwargs={}):
    """
    Fit a powerlaw to the degree distribution of an activity graph

    Parameters
    ----------
    graph: Networkx graph object
     mode: str
        Can be {"all", "out", "in"}. The type of node degree to consider. Default is "out". See `get_degrees`
    fit_kwargs: dict
        Arguments for the powerlaw Fit object

    Returns
    powerlaw fit object, alpha and xmin parameters of powerlaw fit
    -------

    """
    degrees = get_degrees(graph, mode=mode, sort_degrees=True, norm=False, weight=weight)
    degrees = _cut_off_zeros_from_sorted_list(degrees)
    fit = powerlaw.Fit(degrees, discrete=True, estimate_discrete=False, **fit_kwargs)
    return fit, fit.alpha, fit.xmin


def fit_edge_weight_dist_power_law(graph, weight="weight", fit_kwargs={}):
    """
    Fit a powerlaw to the edge weight distribution of an activity graph

    Parameters
    ----------
    graph: Networkx graph object
     mode: str
        Can be {"all", "out", "in"}. The type of node degree to consider. Default is "out". See `get_degrees`
    fit_kwargs: dict
        Arguments for the powerlaw Fit object

    Returns
    powerlaw fit object, alpha and xmin parameters of powerlaw fit
    -------

    """
    edge_weights = get_edge_weights(graph, sort_weights=True, norm=False, weight=weight)
    if len(edge_weights) == 0:
        return np.nan, np.nan, np.nan
    edge_weights = _cut_off_zeros_from_sorted_list(edge_weights)
    fit = powerlaw.Fit(edge_weights, discrete=True, **fit_kwargs)

    return fit, fit.alpha, fit.xmin


def func_simple_powerlaw(x, beta):
    return x ** (-beta)


def func_truncated_powerlaw(x, beta, xo, cutoff):
    """e.g., as used in Gonzalez, Marta C., Cesar A. Hidalgo, and Albert-Laszlo Barabasi.
    "Understanding individual human mobility patterns." nature 453.7196 (2008): 779-782."""
    return (x + xo) ** (-beta) * np.exp(-x / cutoff)


def fit_function_on_node_degrees(item_list, fun, maxfev=3000, bounds=(0, 5), norm=True):
    if len(item_list) == 0 or np.sum(item_list) == 0:
        return 0

    sorted_vals = sorted(item_list)[::-1]
    # get relative probability

    if norm:
        sorted_vals = sorted_vals / sorted_vals[0]

    if fun == "simple_powerlaw":
        fun = func_simple_powerlaw
    elif fun == "truncated_powerlaw":
        fun = func_truncated_powerlaw

    params, _ = curve_fit(fun, np.arange(len(sorted_vals)) + 1, sorted_vals, maxfev=maxfev, bounds=bounds)

    return params[0]


def fit_powerlaw(item_list, maxfev=3000, bounds=(0, 5)):
    warn("fit_powerlaw is deprecated use fit_function_on_node_degrees instead", DeprecationWarning, stacklevel=2)
    if len(item_list) == 0 or np.sum(item_list) == 0:
        return 0

    sorted_vals = sorted(item_list)[::-1]
    # get relative probability
    normed_vals = sorted_vals / np.sum(sorted_vals)
    # Normalize by first value! Because: power function 1/x^beta always passes through (1,1) - we want to fit this
    normed_vals = normed_vals / normed_vals[0]
    params, _ = curve_fit(
        func_simple_powerlaw, np.arange(len(normed_vals)) + 1, normed_vals, maxfev=maxfev, bounds=bounds
    )
    # Prev version: with cutoff and no normalization
    # sorted_vals = (sorted(item_list)[::-1])[:cutoff]
    # normed_degrees = sorted_vals / np.sum(sorted_vals)
    return params[0]


def degree_beta(graph, k=None, degree_type="out"):
    """
    Returns powerlaw fit on degree distribution

    Parameters
    ----------
    graph: Networkx graph
    k: int
        Number of highest degree nodes to consider. If "None" all degrees are considered.
    degree_type: str
    Can be {"all", "out", "in"}. The type of node degree to consider. Default is "out"

    Returns
    -------

    """

    degrees = get_degrees(graph, mode=degree_type, sort_degrees=True)
    if k is not None:
        degrees = degrees[:k]

    return fit_powerlaw(np.asarray(degrees))


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


def _cut_off_zeros_from_sorted_list(input_list):
    """returns the input list without zeros. Input has to be sorted in ascending order."""
    if input_list[-1] > 0:
        return input_list

    for ix, el in enumerate(input_list[::-1]):
        if el > 0:
            return input_list[:-ix]

    return []
