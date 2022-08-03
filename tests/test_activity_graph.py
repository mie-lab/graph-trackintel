import pandas as pd
import geopandas as gpd
import pytest
import trackintel as ti
from sqlalchemy import create_engine
from graph_trackintel.activity_graph import ActivityGraph
import numpy as np
import os
import pickle
import ntpath
import json
import datetime
from shapely.geometry import Point

CRS_WGS84 = "epsg:4326"

n = "fconn"  # number of neighbors for neighbor weights


@pytest.fixture
def example_staypoints():
    """Staypoints to load into the database."""
    l1 = Point(8.5067847, 47.4)
    l2 = Point(8.5067847, 47.6)
    l4 = Point(8.5067847, 47.8)
    l6 = Point(8.5067847, 47.0)

    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 01:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-01 02:00:00", tz="utc")
    t4 = pd.Timestamp("1971-01-01 03:00:00", tz="utc")
    t5 = pd.Timestamp("1971-01-01 04:00:00", tz="utc")
    t6 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t7 = pd.Timestamp("1971-01-01 06:00:00", tz="utc")
    t8 = pd.Timestamp("1971-01-01 07:00:00", tz="utc")

    s1 = "Home"
    s2 = "work"
    s4 = "sport"
    s5 = "park"
    s6 = "friend"

    c1 = "a"
    c2 = "b"
    c3 = "c"
    c4 = "d"
    c5 = "e"
    c6 = "f"
    c7 = "g"

    one_hour = datetime.timedelta(hours=1)

    list_dict = [
        {
            "user_id": 0,
            "started_at": t1,
            "finished_at": t2,
            "geometry": l1,
            "label": s1,
            "context": c1,
            "location_id": 1,
        },
        {
            "user_id": 0,
            "started_at": t2,
            "finished_at": t3,
            "geometry": l2,
            "label": s2,
            "context": c2,
            "location_id": 2,
        },
        {
            "user_id": 0,
            "started_at": t3,
            "finished_at": t4,
            "geometry": l2,
            "label": s2,
            "context": c3,
            "location_id": 2,
        },
        {
            "user_id": 0,
            "started_at": t4,
            "finished_at": t5,
            "geometry": l4,
            "label": s4,
            "context": c4,
            "location_id": 4,
        },
        {
            "user_id": 0,
            "started_at": t5,
            "finished_at": t6,
            "geometry": l2,
            "label": s5,
            "context": c5,
            "location_id": 2,
        },
        {
            "user_id": 0,
            "started_at": t6,
            "finished_at": t7,
            "geometry": l6,
            "label": s6,
            "context": c6,
            "location_id": 6,
        },
        {
            "user_id": 0,
            "started_at": t7,
            "finished_at": t2,
            "geometry": l1,
            "label": s1,
            "context": c7,
            "location_id": 1,
        },
    ]
    stps = gpd.GeoDataFrame(data=list_dict, geometry="geometry", crs="EPSG:4326")
    stps.index.name = "id"
    assert stps.as_staypoints
    return stps


@pytest.fixture
def example_locations():
    """Locations to load into the database."""
    l1 = Point(8.5067847, 47.4)
    l2 = Point(8.5067847, 47.6)
    l4 = Point(8.5067847, 47.8)
    l6 = Point(8.5067847, 47.0)

    list_dict = [
        {"id": 1, "user_id": 0, "center": l1},
        {"id": 2, "user_id": 0, "center": l2},
        {"id": 4, "user_id": 0, "center": l4},
        {"id": 6, "user_id": 0, "center": l6},
    ]
    locs = gpd.GeoDataFrame(data=list_dict, geometry="center", crs="EPSG:4326")
    locs.set_index("id", inplace=True)
    assert locs.as_locations
    return locs


class TestValidate_user:
    def test1(self):
        pass


# activity graph
class TestActivtyGraph:
    def test_create_activty_graph_example(self, example_staypoints, example_locations):
        """Test if adjecency matrix gets corretly reproduced"""
        A_true = np.asarray([[0, 1, 0, 0], [0, 1, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]])

        spts = example_staypoints
        locs = example_locations
        AG = ActivityGraph(spts, locs)
        # AG.plot(os.path.join(".", "tests"))
        A = np.asarray(AG.get_adjecency_matrix().todense())
        assert np.allclose(A, A_true)

    def test_plot_example(self, example_staypoints, example_locations):
        """create a plot of a sbb gc user"""
        sp = example_staypoints
        locs = example_locations
        AG = ActivityGraph(sp, locs)
        os.makedirs(os.path.join(".", "output_test_plots"), exist_ok=True)
        AG.plot(filename=os.path.join(".", "output_test_plots", "example_spring"), layout="spring")
        AG.plot(filename=os.path.join(".", "output_test_plots", "example_coordinate"), layout="coordinate")
