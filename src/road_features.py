import networkx as nx
import pandas as pd

print("Loading road network...")

G = nx.read_graphml("data/raw/roads.graphml")

print("Road network loaded!")
print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))

# Calculate how many roads connect to each node
node_degree = dict(G.degree())

roads = []

for u, v, data in G.edges(data=True):

    highway = data.get("highway", "unknown")

    if isinstance(highway, list):
        highway = highway[0]

    # -----------------------------
    # Road importance
    # -----------------------------

    importance = 1

    if highway in ["motorway", "trunk"]:
        importance = 5
    elif highway == "primary":
        importance = 4
    elif highway == "secondary":
        importance = 3
    elif highway == "tertiary":
        importance = 2

    # -----------------------------
    # Road connectivity
    # -----------------------------

    connectivity = node_degree.get(u, 0) + node_degree.get(v, 0)

    roads.append({
        "from_node": u,
        "to_node": v,
        "length_m": float(data.get("length", 0)),
        "highway": highway,
        "name": data.get("name", "unnamed"),
        "road_importance": importance,
        "connectivity": connectivity
    })

roads_df = pd.DataFrame(roads)

roads_df.to_csv(
    "data/processed/road_features.csv",
    index=False
)

print("\nRoad features created!")
print("Road segments:", len(roads_df))

print("\nImportance distribution:")
print(
    roads_df["road_importance"]
    .value_counts()
    .sort_index()
)

print("\nConnectivity statistics:")
print(
    roads_df["connectivity"].describe()
)

print("\nSample:")
print(roads_df.head())

print("\nGIS road feature extraction complete.")