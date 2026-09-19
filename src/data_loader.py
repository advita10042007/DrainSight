import pandas as pd


def load_data():
    road_features = pd.read_csv(
        "data/processed/road_features.csv"
    )

    flood_features = pd.read_csv(
        "data/processed/flood_risk_features.csv"
    )

    rainfall = pd.read_csv(
        "data/raw/rainfall.csv"
    )

    vision_metadata = pd.read_csv(
        "data/raw/vision_metadata.csv"
    )

    return {
        "road_features": road_features,
        "flood_features": flood_features,
        "rainfall": rainfall,
        "vision_metadata": vision_metadata
    }


if __name__ == "__main__":
    data = load_data()

    print("=== DrainSight Data Loader ===")

    for name, df in data.items():
        print(f"{name}: {len(df)} rows, {len(df.columns)} columns")