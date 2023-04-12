# ROADMAP

#### Folder structure:

  - Folder IO:
      - from_file
      - from_postgis
  - Folder graph preprocessing —> from tracking data to graph:
      - activity graph
      - graph_utils (only utilities for the preprocessing)
  -  Folder analysis:
      - graph_features (maybe divide into single-value features and _helper like degree - distribution)
      - derive_subgraphs (copy from current utils.py)
      - graph_matching (feature-based similarity comparison of two graphs)
      - graph_hypothesis_testing (from my network analysis paper)
  - Folder visualisation

#### Code structure

- ActivityGraph is the main class --> implement as in Trackintel that everythign is in separate folders and AG class gets important methods as wrappers
- The other functions are implemented such that the first argument is a graph, e.g. node_degree(AG, ...)
- Write wrapper functions in AG class for the methods that are in other files (e.g. IO files)

**Important: Examples**
