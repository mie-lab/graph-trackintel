import networkx as nx
from http.client import IncompleteRead
import smopy


def nx_coordinate_layout(G):
    """
    Return networkx graph layout based on geographic coordinates.
    Parameters
    ----------
    G : networkx graph
        A networkx graph that was generated based on trackintel locations.
        Nodes require the `center` attribute that holds a shapely point
        geometry
    Returns
    -------
    pos : dictionary
        dictionary with node_id as key that holds coordinates for each node
    """
    node_center = nx.get_node_attributes(G, "center")
    pos = {key: (geometry.x, geometry.y) for key, geometry in node_center.items()}

    return pos


def nx_coordinate_layout_smopy(G, smap):
    """ "transforms WGS84 coordinates to pixel coordinates of a smopy map"""
    node_center = nx.get_node_attributes(G, "center")
    pos = {key: (smap.to_pixels(geometry.y, geometry.x)) for key, geometry in node_center.items()}

    return pos


def draw_smopy_basemap(G, figsize=(8, 6), zoom=10, ax=None):
    """Draw a basemap with the extent given by graph G"""

    pos_wgs = nx_coordinate_layout(G)
    lon = [coords[0] for coords in pos_wgs.values()]
    lat = [coords[1] for coords in pos_wgs.values()]

    lon_min = min(lon)
    lon_max = max(lon)
    lat_min = min(lat)
    lat_max = max(lat)
    attempts = 0
    while attempts < 3:
        try:
            # smap = smopy.Map(lat_min, lon_min, lat_max, lon_max, tileserver="http://tile.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png", z=zoom)
            smap = smopy.Map(lat_min, lon_min, lat_max, lon_max, z=zoom)

            break
        except IncompleteRead as e:

            attempts += 1
            if attempts == 3:
                print(G.graph["user_id"], e)
                smap = smopy.Map(lat_min, lon_min, lat_max, lon_max)

    # map = smopy.Map((min_y, max_y-min_y, min_x, max_x-min_x), z=5)
    ax = smap.show_mpl(figsize=figsize, ax=ax)

    return ax, smap
