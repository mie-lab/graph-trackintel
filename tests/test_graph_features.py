
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from graph_trackintel.analysis.graph_features import get_edge_weights

@pytest.fixture()
def example_weighted_graph():
    """ simple weighted graph
    https://networkx.org/documentation/networkx-2.3/auto_examples/drawing/plot_weighted_graph.html
    #sphx-glr-auto-examples-drawing-plot-weighted-graph-py



    """
    G = nx.Graph()
    G.add_edge('a', 'b', weight=0.6)
    G.add_edge('a', 'c', weight=0.2)
    G.add_edge('c', 'd', weight=0.1)
    G.add_edge('c', 'e', weight=0.7)
    G.add_edge('c', 'f', weight=0.9)
    G.add_edge('a', 'd', weight=0.3)

    return G



class TestGetEdgeWeights:
    def test_get_list_of_weights(self, example_weighted_graph):
        G = example_weighted_graph
        edge_weights = get_edge_weights(G, sort_weights=True)

        true_weights = np.asarray([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
        np.allclose(true_weights, edge_weights)


