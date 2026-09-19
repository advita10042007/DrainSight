import os
import pandas as pd
import networkx as nx

print("=== DrainSight GIS Foundation ===")

# -----------------------------
# 1. Check rainfall data
# -----------------------------
rainfall_path = "data/raw/rainfall.csv"

rainfall = pd.read_csv(rainfall_path)

print("\nRainfall data:")
print(rainfall.head())
print("Rows:", len(rainfall))

# -----------------------------
# 2. Load road network
# -----------------------------
roads_path = "data/raw/roads.graphml"

G = nx.read_graphml(roads_path)

print("\nRoad network:")
print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))

# -----------------------------
# 3. Pilot area
# -----------------------------
pilot_path = "data/raw/pilot_area.csv"

pilot = pd.read_csv(pilot_path)

print("\nPilot area:")
print(pilot)

# -----------------------------
# 4. Check files
# -----------------------------
print("\nRequired files:")

files = [
    "data/raw/pilot.osm",
    "data/raw/roads.graphml",
    "data/raw/rainfall.csv",
    "data/raw/pilot_area.csv"
]

for file in files:
    print(file, "->", "FOUND" if os.path.exists(file) else "MISSING")

print("\nGIS foundation check complete.")