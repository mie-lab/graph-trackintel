import ntpath
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import networkx as nx
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import functools
from trackintel.geogr.distances import haversine_dist
from graph_trackintel.graph_utils import (
    initialize_multigraph,
)
from graph_trackintel.plotting import draw_smopy_basemap, nx_coordinate_layout_smopy
from pathlib import Path


class ActivityGraph:
    """Class to represent mobiltiy as graphs (activity graph)

    Creates a graph representation based on staypoints and locations or trips and locations for a single user.
    Locations are used as
    nodes, staypoints or trips are used to determine the edges between the nodes.

    Parameters
    ----------
    staypoints: Geodataframe
        trackintel staypoints, staypoint id musst be set as index, Name has to be 'id'
    locations: Geodataframe
        trackintel locations, location id musst be set as index. Name has to be 'id'
    trips: Dataframe or Geodataframe
        trackintel trips. Trips need the columns ['origin_location_id', 'destination_location_id'] or staypoints have
        to be provided
    node_feature_names:

    gap_threshold: (hours)
    """

    def __init__(self, locations, staypoints=None, trips=None, node_feature_names=[], gap_threshold=100):
        assert staypoints is not None or trips is not None, (
            "Activity graph needs to be initialized with either " "staypoints or trips"
        )

        self.node_feature_names = node_feature_names
        self.gap_threshold = gap_threshold
        self.init_activity_dict()
        self.all_loc_ids = locations.index.unique().values
        assert locations.index.name == "id", "Location ID must be set as index!"

        if trips is not None:
            self.validate_user(trips, locations)
            self.user_id = trips["user_id"].iloc[0]
            self.check_add_location_id_to_trips(trips=trips, staypoints=staypoints)
            self.weights_transition_count_trips(trips=trips)

        elif staypoints is not None:
            self.validate_user(staypoints, locations)
            self.user_id = staypoints["user_id"].iloc[0]
            assert staypoints.index.name == "id", "Staypoints ID must be set as index!"
            self.weights_transition_count(staypoints, gap_threshold=self.gap_threshold)

        self.G = self.generate_activity_graphs(locations)

    def init_activity_dict(self):
        self.adjacency_dict = {}
        self.adjacency_dict["A"] = []
        self.adjacency_dict["location_id_order"] = []
        self.adjacency_dict["edge_name"] = []

    def validate_user(self, sp_or_trips, locations):
        """Verify that all data comes from a single valid user.

        A graph can only be constructed from a single user. Locations and staypoints or trips need to be from the
        same user.
        Parameters
        ----------
        sp_or_trips: Geodataframe or Dataframe
        locations: Geodataframe

        Returns
        -------

        """

        assert len(sp_or_trips["user_id"].unique()) == 1, (
            "An activity graph has to be user specific but your "
            "staypoints or trips have"
            f" these users: {sp_or_trips['user_id'].unique()}"
        )
        assert len(sp_or_trips["user_id"].unique()) == 1, (
            "An activity graph has to be user specific but your "
            "locations have"
            f" these users: {locations['user_id'].unique()}"
        )
        user_sp_or_trips = sp_or_trips["user_id"].unique()
        user_locations = locations["user_id"].unique()

        assert (user_sp_or_trips == user_locations).all(), (
            f"staypoints or trips and locations need the same user_id but your "
            f"data have staypoints or trips: {user_sp_or_trips} and locations: {user_locations}"
        )

    def check_add_location_id_to_trips(self, trips, staypoints=None):
        if not all(x in trips.columns for x in ["origin_location_id", "destination_location_id"]):
            assert staypoints is not None, (
                "trips require columns ['origin_location_id', "
                "'destination_location_id'] or staypoints need to be provided"
            )
            _join_location_id(trips=trips, staypoints=staypoints)

    def weights_transition_count_trips(self, trips, adjacency_dict=None):
        """
        # copy of weights_transition_count
        Calculate the number of transition between locations as graph weights.
        Graphs based on the activity locations (trackintel locations) can have several
        types of weighted edges. This function calculates the edge weight based
        on the number of transitions of an individual user between locations.
        The function requires the staypoints to have a cluster id field (e.g.
        staypoints.as_staypoints.extract_locations() was already used.

        Parameters
        ----------
        staypoints : GeoDataFrame

        Returns
        -------
        adjacency_dict : dictionary
                A dictionary of adjacency matrices of type scipy.sparse.coo_matrix
        """
        trips_a = trips.copy()
        trips_a = trips_a.sort_values(["started_at"])

        # delete trips with unknown start/end location
        trips_a.dropna(subset=["origin_location_id", "destination_location_id"], inplace=True)

        # make sure that all nodes are present when creating the edges
        # append a dataframe of self loops for all locations with weight 0

        try:
            counts = (
                trips_a.groupby(by=["user_id", "origin_location_id", "destination_location_id"])
                .size()
                .reset_index(name="counts")
            )
            counts = self._add_all_loc_ids_to_counts(counts)
        except ValueError:
            # If there are only rows with nans, groupby throws an error but should
            # return an empty dataframe
            counts = pd.DataFrame(columns=["user_id", "origin_location_id", "destination_location_id", "counts"])
            print("empty user?", trips.iloc[0]["user_id"])

        # create Adjacency matrix
        A, location_id_order, name = _create_adjacency_matrix_from_transition_counts(counts)

        self.adjacency_dict["A"].append(A)
        self.adjacency_dict["location_id_order"].append(location_id_order)
        self.adjacency_dict["edge_name"].append("transition_counts")

        return adjacency_dict

    def _add_all_loc_ids_to_counts(self, counts):
        """Add self loops with weight zero for all locations. This is important to include all locations in the
        adjacency matrix also if there are unconnected locations."""

        temp_df = pd.DataFrame(
            data=[
                pd.NA * np.ones(self.all_loc_ids.shape[0]),
                self.all_loc_ids,
                self.all_loc_ids,
                np.zeros(self.all_loc_ids.shape),
            ],
            index=["user_id", "origin_location_id", "destination_location_id", "counts"],
        ).transpose()
        temp_df["user_id"] = self.user_id

        # counts = counts.append(temp_df, ignore_index=True)
        counts = pd.concat([counts, temp_df], axis=0, ignore_index=True)
        return counts

    def weights_transition_count(self, staypoints, adjacency_dict=None, gap_threshold=None):
        """
        Calculate the number of transition between locations as graph weights.
        Graphs based on the activity locations (trackintel locations) can have several
        types of weighted edges. This function calculates the edge weight based
        on the number of transitions of an individual user between locations.
        The function requires the staypoints to have a cluster id field (e.g.
        staypoints.as_staypoints.extract_locations() was already used.

        Parameters
        ----------
        staypoints : GeoDataFrame

        Returns
        -------
        adjacency_dict : dictionary
                A dictionary of adjacency matrices of type scipy.sparse.coo_matrix
        gap_threshold: float
                Maximum time between the start of two staypoints so that they are still considered consecutive (hours)

        """
        gap_threshold = pd.to_timedelta("{}h".format(gap_threshold))
        staypoints_a = staypoints.sort_values(["user_id", "started_at"])
        # Deleting staypoints without cluster means that we count non-direct
        # transitions between two clusters e.g., 1 -> -1 -> 2 as direct transitions
        # between two clusters!
        # E.g., 1 -> 2

        staypoints_a.dropna(subset=["location_id"], inplace=True)
        staypoints_a = staypoints_a.loc[staypoints_a["location_id"] != -1]

        if gap_threshold is not None:
            if "finished_at" not in staypoints_a.columns or sum(staypoints_a["finished_at"].isna()) > 0:
                staypoints_a["finished_at"] = staypoints_a.groupby("user_id")["started_at"].shift(-1)

            duration = staypoints_a["finished_at"] - staypoints_a["started_at"]
            gap_flag = duration < gap_threshold
            staypoints_a = staypoints_a[gap_flag]

        # count transitions between cluster
        staypoints_a["origin_location_id"] = staypoints_a["location_id"]
        staypoints_a["destination_location_id"] = staypoints_a.groupby("user_id")["location_id"].shift(-1)

        # drop transitions without locations.
        # this means we only count locations between two valid locations
        staypoints_a.dropna(subset=["origin_location_id", "destination_location_id"], inplace=True)

        try:
            counts = (
                staypoints_a.groupby(by=["user_id", "origin_location_id", "destination_location_id"])
                .size()
                .reset_index(name="counts")
            )
            counts = self._add_all_loc_ids_to_counts(counts)
        except ValueError:
            # If there are only rows with nans, groupby throws an error but should
            # return an empty dataframe
            counts = pd.DataFrame(columns=["user_id", "origin_location_id", "destination_location_id", "counts"])

        # create Adjacency matrix
        A, location_id_order, name = _create_adjacency_matrix_from_transition_counts(counts)

        self.adjacency_dict["A"].append(A)
        self.adjacency_dict["location_id_order"].append(location_id_order)
        self.adjacency_dict["edge_name"].append("transition_counts")

        return adjacency_dict

    @property
    @functools.lru_cache()
    def edge_types(self):
        edge_type_list = []

        for n, nbrsdict in self.G.adjacency():  # iter all nodes
            for nbr, keydict in nbrsdict.items():  # iter all neighbors
                for edge_type_name, _ in keydict.items():  # iter all edges
                    if edge_type_name not in edge_type_list:
                        # append edge attribute to list
                        edge_type_list.append(edge_type_name)

        return edge_type_list

    def to_file(self, path):
        pass

    def get_k_importance_nodes(self, k):
        node_in_degree = np.asarray([(n, self.G.in_degree(n)) for n in self.G.nodes])
        best_ixs = np.argsort(
            node_in_degree[:, 1],
        )[
            ::-1
        ][:k]
        # we readdress the first column of node_in_degree with best_ixs in case that node degree are not
        # a serial starting from 0
        return node_in_degree[:, 0][best_ixs]

    def generate_activity_graphs(self, locations):
        """
        Generate user specific graphs based on activity locations (trackintel locations).
        This function creates a networkx graph per user based on the locations of
        the user as nodes and a set of (weighted) edges defined in adjacency dict.
        Parameters
        ----------
        locations : GeoDataFrame
            Trackintel dataframe of type locations
        adjacency_dict : dictionary or list of dictionaries
             A dictionary with adjacendy matrices of type: {user_id:
             scipy.sparse.coo_matrix}.
        edgenames : List
            List of names (stings) given to edges in a multigraph
        Returns
        -------
        G_dict : dictionary
            A dictionary of type: {user_id: networkx graph}.
        """
        # Todo: Enable multigraph input. E.g. adjacency_dict[user_id] = [edges1,
        #  edges2]
        # Todo: Should we do a check if locations is really a dataframe of trackintel
        #  type?

        locations = locations.copy()
        G_dict = {}
        # we want the location id
        locations.index.name = "location_id"
        assert locations.index.is_unique

        locations.reset_index(inplace=True)
        locations = locations.set_index("user_id", drop=False)
        locations.index.name = "user_id_ix"
        locations.sort_values(by="location_id", inplace=True)

        if "extent" not in locations.columns:
            locations["extent"] = pd.NA

        G = initialize_multigraph(self.user_id, locations, self.node_feature_names)
        G.graph["edge_keys"] = []

        # todo: put edge creation in extra function
        A_list = self.adjacency_dict["A"]
        location_id_order_list = self.adjacency_dict["location_id_order"]
        edge_name_list = self.adjacency_dict["edge_name"]

        for ix in range(len(A_list)):
            A = A_list[ix]
            location_id_order = location_id_order_list[ix]
            edge_name = edge_name_list[ix]

            # assert location_id_order
            for node_ix, location_id in enumerate(location_id_order):
                assert location_id == G.nodes[node_ix]["location_id"]

            G_temp = nx.from_scipy_sparse_matrix(A, create_using=nx.MultiDiGraph())
            edge_list = nx.to_edgelist(G_temp)

            # target structure for edge list:
            # [(0, 0, 'transition_counts', {'weight': 1.0, 'edge_name': 'transition_counts'}),
            # (0, 1, 'transition_counts', {'weight': 7.0, 'edge_name': 'transition_counts'}
            edge_list = [
                (
                    x[0],
                    x[1],
                    edge_name,
                    {
                        **x[2],
                        **{
                            "edge_name": edge_name,
                            "origin_location_id": G.nodes[x[0]]["location_id"],
                            "destination_location_id": G.nodes[x[1]]["location_id"],
                        },
                    },
                )
                for x in edge_list
            ]

            G.add_edges_from(edge_list, weight="weight")
            G.graph["edge_keys"].append(edge_name)

            assert len(G.nodes) == A.shape[0]

        return G

    def plot(
        self,
        filename=None,
        layout="spring",
        edge_attributes=None,
        filter_node_importance=None,
        filter_extent=False,
        filter_dist=100,
        dist_spring_layout=10,
        draw_edge_label=False,
        draw_edge_label_type="transition_counts",
        close_figure=True,
        ax=None,
        node_size_scale=1,
        width=None,
        iterations=50,
        spring_k=None,
        draw_kwargs={},
    ):
        """

        Parameters
        ----------
        image_folder
        layout [spring, coordinate]
        edge_attributes
        filter_node_importance
        filter_extent
        filter_dist
        dist_spring_layout

        Returns
        -------

        """
        if filename is not None:
            folder_name = ntpath.dirname(filename)
            Path(folder_name).mkdir(parents=True, exist_ok=True)

        if filter_node_importance is not None:
            important_nodes = self.get_k_importance_nodes(filter_node_importance)
        else:
            important_nodes = self.G.nodes()
        # filter graph extent:
        if filter_extent:
            center_node_id = int(self.get_k_importance_nodes(1))
            c_geom = self.G.nodes[center_node_id]["center"]
            filtered_nodes = [
                n
                for n in self.G.nodes
                if haversine_dist(self.G.nodes[n]["center"].x, self.G.nodes[n]["center"].y, c_geom.x, c_geom.y)
                < filter_dist * 1000
            ]
            important_nodes = np.intersect1d(filtered_nodes, important_nodes)

        G = self.G.subgraph(important_nodes)

        # get largest connected component
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc)

        # edge color management
        if edge_attributes is not None:
            for edge_attribute in edge_attributes:
                weights = [G[u][v][edge_attribute]["weight"] + 1 for u, v in G.edges()]
        else:
            # list(self.G[u][v])[0] is the edge_attribute key (e.g., 'transition_counts' of the edge
            weights = [G[u][v][list(G[u][v])[0]]["weight"] + 1 for u, v in G.edges()]

        if width is None:
            norm_width = np.log(weights) * 2
            width = norm_width

        deg = nx.degree(G)
        node_sizes = [10 * deg[iata] * node_size_scale for iata in G.nodes]

        if layout == "coordinate":
            # draw geographic representation
            ax, smap = draw_smopy_basemap(G, ax=ax)
            nx.draw_networkx(
                G,
                ax=ax,
                font_size=20,
                width=1,
                linewidths=norm_width,
                with_labels=False,
                node_size=node_sizes,
                pos=nx_coordinate_layout_smopy(G, smap),
                # connectionstyle="arc3, rad = 0.1",
                **draw_kwargs,
            )

        elif layout == "spring":
            # draw spring layout
            if ax is None:
                plt.figure()

            if spring_k is None:
                dist_spring_layout / np.sqrt(len(G))

            pos = nx.spring_layout(G, k=spring_k, iterations=iterations)
            nx.draw(
                G,
                pos=pos,
                width=width,
                node_size=node_sizes,
                connectionstyle="arc3, rad = 0.2",
                ax=ax,
                **draw_kwargs,
            )

        if draw_edge_label:
            edges_new = {}
            edges = nx.get_edge_attributes(G, "weight")
            # edges have to be recoded for drawing. Multigraph edges have the format: (n1, n2, edge_type): weight but
            # the drawing function only accepts (n1 n2): weight as input

            for (u, v, enum), weight in edges.items():
                if enum == draw_edge_label_type:
                    edges_new[(u, v)] = str(int(weight))
            GG = nx.Graph()
            GG.add_edges_from(edges_new)
            nx.draw_networkx_edge_labels(GG, pos, edge_labels=edges_new, label_pos=0.2)

        if filename is not None:
            plt.savefig(filename)

        if close_figure:
            plt.close()

    def get_adjacency_matrix_by_type(self, edge_type):
        assert edge_type in self.adjacency_dict["edge_name"], (
            f"Only {self.adjacency_dict['edge_name']} are available " f"but you provided {edge_type}"
        )
        edge_type_ix = self.adjacency_dict["edge_name"].index(edge_type)
        return self.adjacency_dict["A"][edge_type_ix]

    def get_adjacency_matrix(self):
        return nx.linalg.graphmatrix.adjacency_matrix(self.G).tocoo()

    def add_node_features_from_staypoints(
        self, staypoints, agg_dict={"started_at": list, "finished_at": list, "purpose": list}, add_duration=False
    ):
        """
        agg_dict is a dictionary passed on to pandas dataframe.agg()

        """
        if agg_dict is None and not add_duration:
            raise ValueError(f"Nothing to aggregate agg_dict is {agg_dict} and add_duration is {add_duration}")

        if agg_dict is None:
            agg_dict = {}
        sp = staypoints
        if add_duration:
            sp = sp.copy()
            sp["duration"] = sp["finished_at"] - sp["started_at"]
            agg_dict.update({"duration": sum})

        sp_grp_by_loc = sp.groupby("location_id").agg(agg_dict)

        for node_id, node_data in self.G.nodes(data=True):
            location_id = node_data["location_id"]
            # check if location id is in sp_grp_by_loc
            if location_id in sp_grp_by_loc.index:
                self.G.nodes[node_id].update(sp_grp_by_loc.loc[location_id].to_dict())

    def add_edge_features_from_trips(
        self,
        trips,
        edge_type_to_add="transition_counts",
        agg_dict={"started_at": list, "finished_at": list},
        add_duration=False,
        staypoints=None,
    ):
        # todo: Adding edge features from trips is really slow. Likely because we iterate all edges of each graph in
        #  a simple for loop.

        """
        agg_dict is a dictionary passed on to pandas dataframe.agg()

        """
        if agg_dict is None and not add_duration:
            raise ValueError(f"Nothing to aggregate agg_dict is {agg_dict} and add_duration is {add_duration}")

        if agg_dict is None:
            agg_dict = {}

        self.check_add_location_id_to_trips(trips=trips, staypoints=staypoints)

        if add_duration:
            trips = trips.copy()
            trips["duration"] = trips["finished_at"] - trips["started_at"]
            agg_dict.update({"duration": sum})

        trips_grp = trips.groupby(["origin_location_id", "destination_location_id"]).agg(agg_dict)

        for origin_node_id, destination_node_id, edge_type, edge_data in self.G.edges(data=True, keys=True):
            if edge_type != edge_type_to_add:
                continue

            origin_location_id = self.G.nodes[origin_node_id]["location_id"]
            destination_location_id = self.G.nodes[destination_node_id]["location_id"]

            if (origin_location_id, destination_location_id) in trips_grp.index:
                self.G.edges[origin_node_id, destination_node_id, edge_type].update(
                    trips_grp.loc[(origin_location_id, destination_location_id)].to_dict()
                )

    def get_node_feature_gdf(self):
        gdf = gpd.GeoDataFrame([self.G.nodes[node_id] for node_id in self.G.nodes()], geometry="center")
        gdf.set_index("location_id", inplace=True)

        return gdf

    def get_edge_feature_df(self):
        df = gpd.GeoDataFrame([self.G.edges[key] for key in self.G.edges(keys=True)])
        df.set_index(["origin_location_id", "destination_location_id"], inplace=True)

        return df


def _create_adjacency_matrix_from_transition_counts(counts):
    """
    Transform transition counts into a adjacency matrix per user.
    The input provides transition counts between locations of a user. These
    counts are transformed into a weighted adjacency matrix.
    Parameters
    ----------
    counts : DataFrame
        pandas DataFrame that has at least the columns ['user_id',
        'location_id', 'location_id_end', 'counts']. Counts represents the
        number of transitions between two locations.
    Returns
    -------
    adjacency_dict : dictionary
            A dictionary of adjacency matrices of type scipy.sparse.coo_matrix
    """

    row_ix = counts["origin_location_id"].values.astype("int")
    col_ix = counts["destination_location_id"].values.astype("int")
    values = counts["counts"].values.astype("float")

    if len(values) == 0:
        A = coo_matrix((0, 0))
        location_id_order = np.asarray([])

    else:
        # ix transformation to go from 0 to n
        org_ix = np.unique(np.concatenate((row_ix, col_ix)))
        new_ix = np.arange(0, len(org_ix))
        ix_tranformation = dict(zip(org_ix, new_ix))
        ix_backtranformation = dict(zip(new_ix, org_ix))

        row_ix = [ix_tranformation[row_ix_this] for row_ix_this in row_ix]
        row_ix = np.asarray(row_ix)
        col_ix = [ix_tranformation[col_ix_this] for col_ix_this in col_ix]
        col_ix = np.asarray(col_ix)

        # set shape and create sparse matrix
        max_ix = np.max([np.max(row_ix), np.max(col_ix)]) + 1
        shape = (max_ix, max_ix)

        A = coo_matrix((values, (row_ix, col_ix)), shape=shape)
        A.eliminate_zeros()
        location_id_order = org_ix

    return A, location_id_order, "transition_counts"


def _join_location_id(trips, staypoints):
    """Join location id from staypoints to trips
    Parameters
    ----------
    trips: Geodataframe or Dataframe
        trackintel trips
    sp: Dataframe
        trackintel staypoints

    Returns
    -------

    """
    origin_not_na = ~trips["origin_staypoint_id"].isna()
    dest_not_na = ~trips["destination_staypoint_id"].isna()

    trips.loc[origin_not_na, "origin_location_id"] = staypoints.loc[
        trips.loc[origin_not_na, "origin_staypoint_id"], "location_id"
    ].values
    trips.loc[dest_not_na, "destination_location_id"] = staypoints.loc[
        trips.loc[dest_not_na, "destination_staypoint_id"], "location_id"
    ].values
