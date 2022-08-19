import pandas as pd
import geopandas as gpd
import pytest
import trackintel as ti
from sqlalchemy import create_engine
import graph_trackintel as gti
from graph_trackintel.activity_graph import ActivityGraph
import numpy as np
import os
import pickle
import ntpath
import json
import datetime
from shapely.geometry import Point
from graph_trackintel.activity_graph import _join_location_id


@pytest.fixture
def single_user_graph():
    """Travel diary of a single user that visits 4 locations multiple times"""
    l1 = Point(8.5067847, 47.4)
    l2 = Point(8.5067847, 47.6)
    l3 = Point(8.5067847, 47.8)
    l4 = Point(8.5067847, 47.0)

    t0 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t1 = pd.Timestamp("1971-01-01 01:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 02:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-01 03:00:00", tz="utc")
    t4 = pd.Timestamp("1971-01-01 04:00:00", tz="utc")
    t5 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t6 = pd.Timestamp("1971-01-01 06:00:00", tz="utc")
    t7 = pd.Timestamp("1971-01-01 07:00:00", tz="utc")
    t8 = pd.Timestamp("1971-01-01 08:00:00", tz="utc")
    t9 = pd.Timestamp("1971-01-01 09:00:00", tz="utc")
    t10 = pd.Timestamp("1971-01-01 10:00:00", tz="utc")
    t11 = pd.Timestamp("1971-01-01 11:00:00", tz="utc")
    t12 = pd.Timestamp("1971-01-01 12:00:00", tz="utc")
    t13 = pd.Timestamp("1971-01-01 13:00:00", tz="utc")
    t14 = pd.Timestamp("1971-01-01 14:00:00", tz="utc")
    t15 = pd.Timestamp("1971-01-01 15:00:00", tz="utc")
    t16 = pd.Timestamp("1971-01-01 16:00:00", tz="utc")
    t17 = pd.Timestamp("1971-01-01 17:00:00", tz="utc")

    c0 = "a"
    c1 = "b"
    c2 = "c"
    c3 = "d"
    c4 = "e"
    c5 = "f"
    c6 = "g"
    c7 = "h"
    c8 = "i"

    # staypoints

    location_id = [1, 2, 3, 1, 2, 3, 1, 4, 1]
    started_at = [t0, t2, t4, t6, t8, t10, t12, t14, t16]
    finished_at = [t1, t3, t5, t7, t9, t11, t13, t15, t17]
    geometry = [l1, l2, l3, l1, l2, l3, l1, l4, l1]
    context = [c0, c1, c2, c3, c4, c5, c6, c7, c8]

    sp = gpd.GeoDataFrame(data=[location_id, started_at, finished_at, geometry, context]).transpose()
    sp.columns = ["location_id", "started_at", "finished_at", "geometry", "context"]
    sp = sp.set_geometry("geometry")
    sp.index.name = "id"
    sp["user_id"] = 1

    # locations
    user_id = [1, 1, 1, 1]
    center = [l1, l2, l3, l4]

    locs = gpd.GeoDataFrame(data=[user_id, center]).transpose()
    locs.columns = ["user_id", "center"]
    locs = locs.set_geometry("center")
    locs.index = np.arange(4) + 1
    locs.index.name = "id"

    # trips
    started_at = [t1, t3, t5, t7, t9, t11, t13, t15]
    finished_at = [t2, t4, t6, t8, t10, t12, t14, t16]
    origin_staypoint_id = [0, 1, 2, 3, 4, 5, 6, 7]
    destination_staypoint_id = [1, 2, 3, 4, 5, 6, 7, 8]

    trips = pd.DataFrame(data=[started_at, finished_at, origin_staypoint_id, destination_staypoint_id]).transpose()
    trips.columns = ["started_at", "finished_at", "origin_staypoint_id", "destination_staypoint_id"]
    trips.index.name = "id"
    trips["user_id"] = 1

    sp.as_staypoints
    trips.as_trips
    locs.as_locations

    return sp, trips, locs


class TestJoinLocationId:
    def test_error_without_fields(self, single_user_graph):
        sp, trips, locs = single_user_graph

        with pytest.raises(AssertionError):
            gti.activity_graph.ActivityGraph(locations=locs, trips=trips)

    def test_prior_join(self, single_user_graph):
        sp, trips, locs = single_user_graph
        _join_location_id(trips=trips, staypoints=sp)
        gti.activity_graph.ActivityGraph(locations=locs, trips=trips)

    def test_internal_join(self, single_user_graph):
        sp, trips, locs = single_user_graph
        gti.activity_graph.ActivityGraph(locations=locs, trips=trips, staypoints=sp)


class TestActivityGraph:
    def test_number_one(self, single_user_graph):
        sp, trips, locs = single_user_graph

    def test_validate_user(self, single_user_graph):
        """test if different scenarios of user mismatch get detected"""
        sp, trips, locs = single_user_graph
        _join_location_id(trips=trips, staypoints=sp)

        sp_ = sp.copy()
        trips_ = trips.copy()

        # sp and locs have the same user
        gti.activity_graph.ActivityGraph(staypoints=sp, locations=locs)

        # trips and locs have the same user
        gti.activity_graph.ActivityGraph(locations=locs, trips=trips)

        # sp and locs have a different user
        with pytest.raises(AssertionError):
            sp_["user_id"] = 2
            gti.activity_graph.ActivityGraph(locations=locs, staypoints=sp_)

        # sp have several users
        with pytest.raises(AssertionError):
            sp_["user_id"] = np.arange(len(sp_))
            gti.activity_graph.ActivityGraph(locations=locs, staypoints=sp_)

        # trips and locs have a different user
        with pytest.raises(AssertionError):
            trips_["user_id"] = 2
            gti.activity_graph.ActivityGraph(locations=locs, trips=trips_)

        # trips have several users
        with pytest.raises(AssertionError):
            trips_["user_id"] = np.arange(len(trips_))
            gti.activity_graph.ActivityGraph(locations=locs, trips=trips_)

    def test_adjacency_from_trips(self, single_user_graph):

        A_true = np.asmatrix([[0.0, 2.0, 0.0, 1.0], [0.0, 0.0, 2.0, 0.0], [2.0, 0.0, 0.0, 0.0], [1, 0.0, 0.0, 0.0]])

        sp, trips, locs = single_user_graph
        _join_location_id(trips=trips, staypoints=sp)

        AG = gti.activity_graph.ActivityGraph(locations=locs, trips=trips)

        A = AG.get_adjecency_matrix().todense()
        assert np.allclose(A_true, A)

    def test_adjacency_from_sp(self, single_user_graph):

        A_true = np.asmatrix([[0.0, 2.0, 0.0, 1.0], [0.0, 0.0, 2.0, 0.0], [2.0, 0.0, 0.0, 0.0], [1, 0.0, 0.0, 0.0]])

        sp, trips, locs = single_user_graph
        AG = gti.activity_graph.ActivityGraph(staypoints=sp, locations=locs)
        A = AG.get_adjecency_matrix().todense()
        assert np.allclose(A_true, A)


class TestAddNodeFeaturesFromStaypoints:
    def test_aggregation(self, single_user_graph):
        sp, trips, locs = single_user_graph
        AG = gti.activity_graph.ActivityGraph(staypoints=sp, locations=locs)
        agg_dict = {"started_at": [min, max, list], "finished_at": "max", "context": [list, "first"]}

        column_names = [
            ("started_at", "min"),
            ("started_at", "max"),
            ("started_at", "list"),
            ("finished_at", "max"),
            ("context", "list"),
            ("context", "first"),
        ]

        AG.add_node_features_from_staypoints(staypoints=sp, agg_dict=agg_dict)
        nodefeats_df = AG.get_node_feature_gdf()
        nodefeats_df = nodefeats_df[column_names]
        sp_agg_df = sp.groupby("location_id").agg(agg_dict)
        sp_agg_df.columns = column_names

        pd.testing.assert_frame_equal(nodefeats_df, sp_agg_df)

    def test_duration(self, single_user_graph):
        sp, trips, locs = single_user_graph

        AG = gti.activity_graph.ActivityGraph(staypoints=sp, locations=locs)
        agg_dict = {}

        AG.add_node_features_from_staypoints(staypoints=sp, agg_dict=agg_dict, add_duration=True)
        nodefeats_df = AG.get_node_feature_gdf()
        duration_from_graph = nodefeats_df["duration"]

        sp["duration"] = sp.finished_at - sp.started_at
        duration_true = sp.groupby("location_id")["duration"].sum()

        pd.testing.assert_series_equal(duration_from_graph, duration_true)

    def test_error_nothing_to_aggregate(self, single_user_graph):
        sp, trips, locs = single_user_graph

        AG = gti.activity_graph.ActivityGraph(staypoints=sp, locations=locs)

        with pytest.raises(ValueError):
            AG.add_node_features_from_staypoints(staypoints=sp, agg_dict={}, add_duration=False)


class TestGetNodeFeaturegdf:
    def test_no_added_features(self, single_user_graph):
        sp, trips, locs = single_user_graph

        AG = gti.activity_graph.ActivityGraph(locations=locs, staypoints=sp)

        node_feat_gdf = AG.get_node_feature_gdf()

        assert node_feat_gdf.index.name == "location_id"

        assert np.allclose(node_feat_gdf.index.values, locs.index.values)

        assert all(node_feat_gdf.center.geom_equals(locs.center))

    def test_with_node_features(self, single_user_graph):
        """Test if added columns appear as node features"""

        sp, trips, locs = single_user_graph
        AG = gti.activity_graph.ActivityGraph(locations=locs, staypoints=sp)

        agg_dict = {"started_at": min, "finished_at": max, "context": "first"}

        AG.add_node_features_from_staypoints(staypoints=sp, agg_dict=agg_dict, add_duration=True)
        node_feat_gdf = AG.get_node_feature_gdf()
        new_columns = ["started_at", "finished_at", "context", "duration"]

        assert all(x in node_feat_gdf.columns for x in new_columns)
