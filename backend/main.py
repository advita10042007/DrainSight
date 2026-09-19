
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from backend.services.data_service import (
    load_risk_features,
    load_hotspots,
    load_flood_events,
    load_road_features,
    load_rainfall,
)

from backend.services.risk_service import get_ranked_risk
from backend.services.explanation_service import explain_risk
from backend.services.intervention_service import simulate_clear_drain

app = FastAPI(
    title="DrainSight API",
    description="Backend API for DrainSight flood-risk monitoring",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "DrainSight backend is running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "DrainSight backend"
    }


@app.get("/api/risk")
def get_risk():
    try:
        df = load_risk_features()

        records = df.to_dict(orient="records")

        return {
            "count": len(records),
            "data": records
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/api/hotspots")
def get_hotspots():
    try:
        df = load_hotspots()

        # Convert missing values to JSON-safe null values
        records = df.astype(object).where(pd.notna(df), None).to_dict(
            orient="records"
        )

        return {
            "count": len(records),
            "data": records
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/api/flood-events")
def get_flood_events():
    try:
        df = load_flood_events()

        # Convert missing values to JSON-safe null values
        records = df.astype(object).where(pd.notna(df), None).to_dict(
            orient="records"
        )

        return {
            "count": len(records),
            "data": records
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/api/roads")
def get_roads():
    try:
        df = load_road_features()

        records = df.to_dict(orient="records")

        return {
            "count": len(records),
            "data": records
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/api/rainfall")
def get_rainfall():
    try:
        df = load_rainfall()

        records = df.to_dict(orient="records")

        return {
            "count": len(records),
            "data": records
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.get("/api/ranked-risk")
def ranked_risk():
    try:
        df = get_ranked_risk()

        records = (
            df.astype(object)
            .where(pd.notna(df), None)
            .to_dict(orient="records")
        )

        return {
            "count": len(records),
            "data": records
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/api/risk/{sector}")
def risk_explanation(sector: str):
    try:
        df = get_ranked_risk()

        matches = df[
            df["sector"].astype(str).str.lower()
            == sector.lower()
        ]

        if matches.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Sector '{sector}' not found"
            )

        row = matches.iloc[0]

        return {
            "sector": row["sector"],
            "risk_score": float(row["risk_score"]),
            "risk_level": row["risk_level"],
            "reasons": explain_risk(row)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/intervention/clear-drain/{sector}")
def clear_drain_simulation(sector: str):
    try:
        result = simulate_clear_drain(sector)

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Sector '{sector}' not found"
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    