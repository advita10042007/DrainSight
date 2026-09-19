import pandas as pd


INPUT_FILE = "data/processed/flood_hotspots_clean.csv"
OUTPUT_FILE = "data/processed/flood_risk_features.csv"


print("Loading historical flood hotspots...")

df = pd.read_csv(INPUT_FILE)


# Keep only hotspots with valid coordinates
df = df[df["coordinate_status"] == "valid"].copy()


print("Valid flood hotspots:", len(df))


# --------------------------------------------------
# 1. Historical flood frequency
# --------------------------------------------------

sector_counts = (
    df.groupby("sector")
    .size()
    .reset_index(name="historical_hotspot_count")
)


# --------------------------------------------------
# 2. Cause-based risk indicators
# --------------------------------------------------

cause_features = (
    df.groupby("sector")[
        [
            "low_lying_flag",
            "blockage_silt_flag",
            "missing_drain_flag",
            "road_level_flag",
            "pumping_flag",
            "external_inflow_flag"
        ]
    ]
    .sum()
    .reset_index()
)


# --------------------------------------------------
# 3. Combine everything
# --------------------------------------------------

features = sector_counts.merge(
    cause_features,
    on="sector",
    how="left"
)


# --------------------------------------------------
# 4. Historical risk score
# --------------------------------------------------

features["historical_risk_score"] = (
    features["historical_hotspot_count"] * 2
    + features["low_lying_flag"] * 2
    + features["blockage_silt_flag"] * 2
    + features["missing_drain_flag"] * 2
    + features["road_level_flag"]
    + features["pumping_flag"]
    + features["external_inflow_flag"]
)


# --------------------------------------------------
# 5. Normalize to 0–100
# --------------------------------------------------

max_score = features["historical_risk_score"].max()

if max_score > 0:
    features["historical_risk_score"] = (
        features["historical_risk_score"]
        / max_score
        * 100
    )


# --------------------------------------------------
# 6. Save
# --------------------------------------------------

features.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nHistorical flood features created!")
print("Sectors:", len(features))
print("\nTop historical-risk sectors:")

print(
    features
    .sort_values(
        "historical_risk_score",
        ascending=False
    )
    .head(10)
)


print("\nSaved to:")
print(OUTPUT_FILE)