from backend.services.risk_service import calculate_sector_risk


def simulate_clear_drain(sector: str):
    df = calculate_sector_risk()

    matches = df[
        df["sector"].astype(str).str.lower()
        == sector.lower()
    ]

    if matches.empty:
        return None

    row = matches.iloc[0]

    before = float(row["risk_score"])

    original_condition = float(
        row["drain_condition_component"]
    )

    improved_condition = original_condition * 0.60

    after = (
        float(row["historical_component"]) * 0.30
        + improved_condition * 0.25
        + float(row["rainfall_component"]) * 0.20
        + float(row["road_importance_component"]) * 0.15
        + float(row["connectivity_component"]) * 0.10
    )

    after = round(max(0, min(100, after)), 2)

    return {
        "sector": row["sector"],
        "intervention": "Clear drain",
        "before_risk": round(before, 2),
        "after_risk": after,
        "risk_reduction": round(before - after, 2),
        "assumption": (
            "Pilot simulation assumes drain clearing "
            "reduces the drain-condition component by 40%."
        )
    }