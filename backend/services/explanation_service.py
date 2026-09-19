def explain_risk(row):
    reasons = []

    if row.get("historical_component", 0) >= 60:
        reasons.append(
            "The location has a strong history of flooding."
        )

    if row.get("drain_condition_component", 0) >= 60:
        reasons.append(
            "Drain or terrain-related risk indicators are elevated."
        )

    if row.get("rainfall_component", 0) >= 60:
        reasons.append(
            "Recent rainfall is contributing significantly to risk."
        )

    if row.get("road_importance_component", 0) >= 60:
        reasons.append(
            "The location is associated with important road infrastructure."
        )

    if row.get("connectivity_component", 0) >= 60:
        reasons.append(
            "The surrounding road network has high connectivity."
        )

    if not reasons:
        reasons.append(
            "No single risk factor dominates the current pilot score."
        )

    return reasons