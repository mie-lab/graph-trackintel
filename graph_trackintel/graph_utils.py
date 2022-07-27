# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:33:39 2019

@author: martinhe

These are some helper functions to analyze human mobility graphs
"""
import smopy
import networkx as nx
import numpy as np
import pandas as pd


def initialize_multigraph(user_id_this, locs_user_view, node_feature_names):
    """
    Initialize a networkx multigraph class based on location information.
    Graph gets assigned the user_id, the locations as nodes and all non-optional node_features (location_id,
    extent and center) as well as optional node features passed via `node_feature_names`. Optional node features have
    to be columns in `loc_user_view`

    Parameters
    ----------
    user_id_this
    locs_user_view
    node_feature_names

    Returns
    -------
    G: networkx MultiDiGraph
    """
    # create graph
    G = nx.MultiDiGraph()
    G.graph["user_id"] = user_id_this

    # add node information
    node_ids = np.arange(len(locs_user_view))
    node_features = locs_user_view.loc[:, ["location_id", "extent", "center"] + node_feature_names].to_dict("records")

    node_tuple = tuple(zip(node_ids, node_features))
    G.add_nodes_from(node_tuple)
    return G


def delete_zero_edges(graph):
    """
    graph: networkx graph
    """
    edges_to_delete = [(a, b) for a, b, attrs in graph.edges(data=True) if attrs["weight"] < 1]
    if len(edges_to_delete) > 0:
        graph.remove_edges_from(edges_to_delete)
    return graph


def get_largest_component(graph):
    """
    Get largest component of networkx graph

    graph: networkx Graph!
    """
    cc = sorted(
        nx.connected_components(graph.to_undirected()),
        key=len,
        reverse=True,
    )
    graph_cleaned = graph.subgraph(cc[0])
    return graph_cleaned.copy()


def remove_loops(graph):
    """
    graph: networkx Graph
    """
    graph.remove_edges_from(nx.selfloop_edges(nx.DiGraph(graph)))
    return graph


def keep_important_nodes(graph, number_of_nodes):
    """
    Reduce to the nodes with highest degree (in + out degree)

    graph: networkx Graph
    """
    sorted_dict = np.array(
        [
            [k, v]
            for k, v in sorted(
                dict(graph.degree()).items(),
                key=lambda item: item[1],
            )
        ]
    )
    use_nodes = sorted_dict[-number_of_nodes:, 0]
    graph = graph.subgraph(use_nodes)
    return graph


def get_adj_and_attr(activity_graph):
    """
    Given an ActivityGraph object, get the adjacency matrix and the node features separately

    Parameters
    ----------
    graph : ActivityGraph

    Returns
    -------
    adjacency: 2D scipy sparse matrix: adjacency matrix
    node_feat_df: pandas DataFrame, node features with one row per node
    """
    list_of_nodes = list(activity_graph.nodes())

    # get adjacency
    adjacency = nx.linalg.graphmatrix.adjacency_matrix(activity_graph, nodelist=list_of_nodes)
    # make a dataframe with the features
    node_dicts = []
    for i, node in enumerate(list_of_nodes):
        node_dict = activity_graph.nodes[node]
        node_dict["node_id"] = node
        node_dict["id"] = i
        node_dicts.append(node_dict)
    node_feat_df = pd.DataFrame(node_dicts).set_index("id")

    # add degrees
    in_degree = np.array(np.sum(adjacency, axis=0)).flatten()
    out_degree = np.array(np.sum(adjacency, axis=1)).flatten()
    node_feat_df["in_degree"] = in_degree
    node_feat_df["out_degree"] = out_degree
    return adjacency, node_feat_df
