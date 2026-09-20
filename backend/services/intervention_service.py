from backend.services.risk_service import calculate_sector_risk


# Each solution scales risk components differently, so every option
# produces a distinct before/after result.
SOLUTION_DEFS = {
    "clear-drain": {
        "label": "Clear drains",
        "drain_factor": 0.60,
        "rain_factor": 1.00,
        "assumption": (
            "Pilot simulation assumes drain clearing "
            "reduces the drain-condition component by 40%."
        ),
    },
    "desilt-culvert": {
        "label": "Desilt culverts",
        "drain_factor": 0.45,
        "rain_factor": 1.00,
        "assumption": (
            "Pilot simulation assumes culvert desilting "
            "reduces the drain-condition component by 55%."
        ),
    },
    "deploy-pumps": {
        "label": "Deploy pumps",
        "drain_factor": 0.70,
        "rain_factor": 0.85,
        "assumption": (
            "Pilot simulation assumes temporary pumps cut the "
            "drain-condition component by 30% and offset 15% "
            "of the rainfall component."
        ),
    },
    "upgrade-channel": {
        "label": "Upgrade channel",
        "drain_factor": 0.30,
        "rain_factor": 0.95,
        "assumption": (
            "Pilot simulation assumes a channel capacity upgrade "
            "reduces the drain-condition component by 70% and "
            "offsets 5% of the rainfall component."
        ),
    },
    "retention-sump": {
        "label": "Upstream retention sump",
        "drain_factor": 0.90,
        "rain_factor": 0.65,
        "assumption": (
            "Pilot simulation assumes an upstream retention sump "
            "throttles the inflow surge, offsetting 35% of the "
            "rainfall component with minor drain relief."
        ),
    },
    "flood-barriers": {
        "label": "Inflatable flood barriers",
        "drain_factor": 0.95,
        "rain_factor": 0.75,
        "assumption": (
            "Pilot simulation assumes perimeter barriers deflect "
            "sheet flow, offsetting 25% of rainfall-driven risk "
            "without changing drain condition."
        ),
    },
}


def observed_blockage_pct(sector: str, risk_level: str = "Medium") -> tuple[float | None, bool]:
    """Blockage observed in the sector's node photo (None if unavailable)."""
    try:
        from backend.services.vision_service import get_node_photo

        photo = get_node_photo(sector, risk_level)
        if photo is None:
            return None, False
        return float(photo["analysis"]["blockage_estimate_pct"]), True
    except Exception:
        return None, False


def simulate_intervention(
    sector: str,
    solution: str = "clear-drain",
    rain_mm: float | None = None,
):
    if solution not in SOLUTION_DEFS:
        return None

    definition = SOLUTION_DEFS[solution]
    df = calculate_sector_risk(rain_mm=rain_mm)

    matches = df[
        df["sector"].astype(str).str.lower()
        == sector.lower()
    ]

    if matches.empty:
        return None

    row = matches.iloc[0]
    before = float(row["risk_score"])

    # Tune the simulation to what the node photo actually shows:
    # a fully-blocked drain gains more from an intervention than
    # an already-clear one.
    blockage, vision_adjusted = observed_blockage_pct(
        str(row["sector"]), str(row.get("risk_level", "Medium"))
    )
    severity = (blockage / 100.0) if blockage is not None else 0.6
    tune = 0.4 + 0.6 * max(0.0, min(1.0, severity))

    def tuned(base_factor: float) -> float:
        improvement = (1.0 - base_factor) * tune
        return max(0.0, 1.0 - improvement)

    drain_factor = tuned(definition["drain_factor"])
    rain_factor = tuned(definition["rain_factor"])

    improved_condition = float(row["drain_condition_component"]) * drain_factor
    improved_rainfall = float(row["rainfall_component"]) * rain_factor

    components_before = {
        "historical": round(float(row["historical_component"]), 2),
        "drain_condition": round(float(row["drain_condition_component"]), 2),
        "rainfall": round(float(row["rainfall_component"]), 2),
        "road_importance": round(float(row["road_importance_component"]), 2),
        "connectivity": round(float(row["connectivity_component"]), 2),
    }
    components_after = {
        "historical": components_before["historical"],
        "drain_condition": round(improved_condition, 2),
        "rainfall": round(improved_rainfall, 2),
        "road_importance": components_before["road_importance"],
        "connectivity": components_before["connectivity"],
    }

    after = (
        float(row["historical_component"]) * 0.30
        + improved_condition * 0.25
        + improved_rainfall * 0.20
        + float(row["road_importance_component"]) * 0.15
        + float(row["connectivity_component"]) * 0.10
    )

    after = round(max(0, min(100, after)), 2)

    assumption = definition["assumption"]
    if vision_adjusted:
        assumption += (
            f" Effect tuned to the node's observed blockage "
            f"({blockage:.1f}% from live photo analysis)."
        )

    return {
        "sector": row["sector"],
        "solution": solution,
        "intervention": definition["label"],
        "before_risk": round(before, 2),
        "after_risk": after,
        "risk_reduction": round(before - after, 2),
        "components_before": components_before,
        "components_after": components_after,
        "observed_blockage_pct": blockage,
        "vision_adjusted": vision_adjusted,
        "assumption": assumption,
    }


def simulate_clear_drain(sector: str, rain_mm: float | None = None):
    return simulate_intervention(sector, "clear-drain", rain_mm)
