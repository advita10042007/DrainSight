import pandas as pd

from backend.services.data_service import (
    load_risk_features,
    load_road_features,
    load_rainfall,
)


def normalize_series(series):
    """
    Convert a numeric series to a 0-100 scale.
    """
    series = pd.to_numeric(series, errors="coerce").fillna(0)

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            [0.0] * len(series),
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
        * 100
    )


def calculate_sector_risk(rain_mm: float | None = None):
    """
    Calculate a transparent pilot flood-risk score
    using existing DrainSight datasets.

    rain_mm optionally overrides the rainfall scenario
    (e.g. 25 / 50 / 75 / 100 from the frontend simulator).
    When None, the total from rainfall.csv is used.
    """

    risk_df = load_risk_features().copy()

    # Historical flooding component
    risk_df["historical_component"] = normalize_series(
        risk_df["historical_risk_score"]
    )

    # Drain-condition component
    condition_columns = [
        "low_lying_flag",
        "blockage_silt_flag",
        "missing_drain_flag",
        "road_level_flag",
        "pumping_flag",
        "external_inflow_flag",
    ]

    existing_columns = [
        column
        for column in condition_columns
        if column in risk_df.columns
    ]

    if existing_columns:
        condition_sum = risk_df[existing_columns].sum(axis=1)
        risk_df["drain_condition_component"] = normalize_series(
            condition_sum
        )
    else:
        risk_df["drain_condition_component"] = 0

    # Rainfall component
    if rain_mm is not None:
        total_rainfall = float(rain_mm)
    else:
        rainfall_df = load_rainfall().copy()

        rainfall_df["rain_mm"] = pd.to_numeric(
            rainfall_df["rain_mm"],
            errors="coerce"
        ).fillna(0)

        total_rainfall = rainfall_df["rain_mm"].sum()

    rainfall_score = min(
        float(total_rainfall),
        100.0
    )

    risk_df["rainfall_component"] = rainfall_score

    # Road importance component
    roads_df = load_road_features().copy()

    if "road_importance" in roads_df.columns:
        road_score = roads_df["road_importance"].mean()

        if road_score > 0:
            road_score = min(
                float(road_score) / 5 * 100,
                100
            )
        else:
            road_score = 0
    else:
        road_score = 0

    risk_df["road_importance_component"] = road_score

    # Connectivity component
    if "connectivity" in roads_df.columns:
        connectivity_score = roads_df["connectivity"].mean()

        if connectivity_score > 0:
            connectivity_score = min(
                float(connectivity_score) / 20 * 100,
                100
            )
        else:
            connectivity_score = 0
    else:
        connectivity_score = 0

    risk_df["connectivity_component"] = connectivity_score

    # Weighted total
    risk_df["risk_score"] = (
        risk_df["historical_component"] * 0.30
        + risk_df["drain_condition_component"] * 0.25
        + risk_df["rainfall_component"] * 0.20
        + risk_df["road_importance_component"] * 0.15
        + risk_df["connectivity_component"] * 0.10
    )

    risk_df["risk_score"] = (
        risk_df["risk_score"]
        .clip(0, 100)
        .round(2)
    )

    # Risk category
    def classify_risk(score):
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Medium"
        return "Low"

    risk_df["risk_level"] = risk_df["risk_score"].apply(
        classify_risk
    )

    return risk_df


def get_ranked_risk(rain_mm: float | None = None):
    """
    Return sectors ranked from highest to lowest risk.
    """

    df = calculate_sector_risk(rain_mm=rain_mm)

    return df.sort_values(
        "risk_score",
        ascending=False
    ).reset_index(drop=True)