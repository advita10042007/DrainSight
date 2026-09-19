import re
import pandas as pd


def standardize_location(location):
    """
    Standardize sector names while preserving
    non-sector locations.
    """

    if pd.isna(location):
        return None

    location = str(location).strip()

    # Standardize locations explicitly written as sectors
    match = re.fullmatch(
        r"sector\s+(\d+)([A-Za-z]?)(?:\s+part\s*[-–]?\s*(.+))?",
        location,
        re.IGNORECASE
    )

    if match:
        number = int(match.group(1))
        letter = match.group(2).upper()
        part = match.group(3)

        result = f"Sector {number}{letter}"

        if part:
            part = re.sub(r"\s+", " ", part.strip())
            result += f" Part {part.upper()}"

        return result

    # Keep named locations unchanged
    return location


def standardize_flood_data(df):
    df = df.copy()
    df["sector_standardized"] = df["sector"].apply(
        standardize_location
    )
    return df


def standardize_vision_data(df):
    df = df.copy()
    df["location_standardized"] = df["location"].apply(
        standardize_location
    )
    return df


if __name__ == "__main__":

    flood = pd.read_csv(
        "data/processed/flood_risk_features.csv"
    )

    vision = pd.read_csv(
        "data/raw/vision_metadata.csv"
    )

    flood = standardize_flood_data(flood)
    vision = standardize_vision_data(vision)

    print("=== Location Standardization Test ===")

    print("\nFlood sectors:")
    print(
        flood[
            ["sector", "sector_standardized"]
        ].drop_duplicates().to_string(index=False)
    )

    print("\nVision locations:")
    print(
        vision[
            ["location", "location_standardized"]
        ].drop_duplicates().to_string(index=False)
    )