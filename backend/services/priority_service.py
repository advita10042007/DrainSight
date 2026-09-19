from backend.services.risk_service import get_ranked_risk


def get_priority_list(limit=10):
    df = get_ranked_risk().head(limit).copy()

    results = []

    for index, row in df.iterrows():
        results.append({
            "rank": index + 1,
            "sector": row["sector"],
            "risk_score": float(row["risk_score"]),
            "risk_level": row["risk_level"],
            "historical_risk": round(
                float(row["historical_component"]), 2
            ),
            "drain_condition_risk": round(
                float(row["drain_condition_component"]), 2
            ),
            "rainfall_risk": round(
                float(row["rainfall_component"]), 2
            )
        })

    return results
