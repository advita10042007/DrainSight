import pandas as pd


HOTSPOTS = "data/processed/flood_hotspots_clean.csv"
FLOOD_EVENTS = "data/processed/flood_control_clean.csv"


def load_flood_data():

    hotspots = pd.read_csv(HOTSPOTS)
    flood_events = pd.read_csv(FLOOD_EVENTS)

    print("Flood hotspot records:", len(hotspots))
    print("Mapped hotspot records:",
          (hotspots["coordinate_status"] == "valid").sum())

    print("Historical flood-control records:", len(flood_events))

    return hotspots, flood_events


if __name__ == "__main__":

    hotspots, flood_events = load_flood_data()

    print("\nHotspot sample:")
    print(
        hotspots[
            [
                "hotspot_id",
                "sector",
                "cause_category",
                "mcg_status"
            ]
        ].head()
    )

    print("\nFlood event sample:")
    print(
        flood_events[
            [
                "event_id",
                "location",
                "sector",
                "cause_category",
                "exposure_type"
            ]
        ].head()
    )