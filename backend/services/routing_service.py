from pathlib import Path

import networkx as nx
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

GRAPH_PATH = BASE_DIR / "data" / "raw" / "roads.graphml"
ROAD_FEATURES_PATH = BASE_DIR / "data" / "processed" / "road_features.csv"


def load_road_graph():
    return nx.read_graphml(GRAPH_PATH)


def load_road_features():
    return pd.read_csv(ROAD_FEATURES_PATH)


def build_routing_graph():
    graph = load_road_graph()
    features = load_road_features()

    feature_lookup = {}

    for _, row in features.iterrows():
        key = (
            str(row["from_node"]),
            str(row["to_node"])
        )

        feature_lookup[key] = row

    for u, v, key, data in graph.edges(
        keys=True,
        data=True
    ):
        lookup = feature_lookup.get(
            (str(u), str(v))
        )

        if lookup is None:
            road_importance = 0
            connectivity = 0
        else:
            road_importance = float(
                lookup["road_importance"]
            )
            connectivity = float(
                lookup["connectivity"]
            )

        length = float(
            data.get("length", 0)
        )

        flood_risk_cost = (
            length
            * (
                1
                + road_importance / 10
                + connectivity / 20
            )
        )

        data["length_m"] = length
        data["road_importance"] = road_importance
        data["connectivity"] = connectivity
        data["flood_risk_cost"] = flood_risk_cost

    return graph


def find_nearest_node(graph, latitude, longitude):
    best_node = None
    best_distance = float("inf")

    for node, data in graph.nodes(data=True):
        try:
            node_lat = float(data["y"])
            node_lon = float(data["x"])
        except (KeyError, TypeError, ValueError):
            continue

        distance = (
            (node_lat - latitude) ** 2
            + (node_lon - longitude) ** 2
        )

        if distance < best_distance:
            best_distance = distance
            best_node = node

    if best_node is None:
        raise ValueError(
            "Could not find a valid graph node."
        )

    return best_node
def calculate_route(
    start_lat,
    start_lon,
    end_lat,
    end_lon
):
    graph = build_routing_graph()

    start_node = find_nearest_node(
        graph,
        start_lat,
        start_lon
    )

    end_node = find_nearest_node(
        graph,
        end_lat,
        end_lon
    )

    normal_route = nx.shortest_path(
        graph,
        start_node,
        end_node,
        weight="length_m"
    )

    flood_safe_route = nx.shortest_path(
        graph,
        start_node,
        end_node,
        weight="flood_risk_cost"
    )

    def route_distance(route):
        total = 0

        for u, v in zip(route[:-1], route[1:]):
            edge_data = graph.get_edge_data(u, v)

            if edge_data is None:
                continue

            edge = min(
                edge_data.values(),
                key=lambda item: float(
                    item.get("length_m", 0)
                )
            )

            total += float(
                edge.get("length_m", 0)
            )

        return round(total, 2)

    def route_coordinates(route):
        coordinates = []

        for node in route:
            data = graph.nodes[node]

            coordinates.append({
                "lat": float(data["y"]),
                "lon": float(data["x"])
            })

        return coordinates

    return {
        "start": {
            "lat": start_lat,
            "lon": start_lon
        },
        "end": {
            "lat": end_lat,
            "lon": end_lon
        },
        "normal_route": {
            "distance_m": route_distance(normal_route),
            "nodes": len(normal_route),
            "coordinates": route_coordinates(
                normal_route
            )
        },
        "flood_safe_route": {
            "distance_m": route_distance(flood_safe_route),
            "nodes": len(flood_safe_route),
            "coordinates": route_coordinates(
                flood_safe_route
            )
        },
        "assumption": (
            "Flood-safe routing uses a pilot graph cost "
            "based on road importance and connectivity. "
            "It is not a validated hydraulic flood model."
        )
    }