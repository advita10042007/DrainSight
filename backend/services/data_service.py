from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"


def load_risk_features():
    path = PROCESSED_DIR / "flood_risk_features.csv"

    if not path.exists():
        raise FileNotFoundError(f"Risk data not found: {path}")

    return pd.read_csv(path)


def load_hotspots():
    path = PROCESSED_DIR / "flood_hotspots_clean.csv"

    if not path.exists():
        raise FileNotFoundError(f"Hotspot data not found: {path}")

    return pd.read_csv(path)


def load_flood_events():
    path = PROCESSED_DIR / "flood_control_clean.csv"

    if not path.exists():
        raise FileNotFoundError(f"Flood event data not found: {path}")

    return pd.read_csv(path)


def load_road_features():
    path = PROCESSED_DIR / "road_features.csv"

    if not path.exists():
        raise FileNotFoundError(f"Road data not found: {path}")

    return pd.read_csv(path)


def load_rainfall():
    path = RAW_DIR / "rainfall.csv"

    if not path.exists():
        raise FileNotFoundError(f"Rainfall data not found: {path}")

    return pd.read_csv(path)