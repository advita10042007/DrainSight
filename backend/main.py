
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import tempfile
from pathlib import Path

from backend.services.data_service import (
    load_risk_features,
    load_hotspots,
    load_flood_events,
    load_road_features,
    load_rainfall,
)

from backend.services.risk_service import get_ranked_risk
from backend.services.explanation_service import explain_risk
from backend.services.intervention_service import (
    simulate_clear_drain,
    simulate_intervention,
    SOLUTION_DEFS,
)
from backend.services.vision_service import get_node_photo
from backend.services.priority_service import get_priority_list
from backend.services.routing_service import calculate_route



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


BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"


@app.get("/api")
def api_root():
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
def ranked_risk(rain_mm: float | None = Query(default=None, ge=0, le=500)):
    try:
        df = get_ranked_risk(rain_mm=rain_mm)

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

def normalize_sector_input(sector):
    sector = str(sector).strip()

    # If input is just a number, convert it to "Sector X"
    if sector.isdigit():
        return f"Sector {int(sector)}"

    # If input is already "Sector X", standardize spacing/case
    if sector.lower().startswith("sector "):
        number = sector.split()[-1]
        if number.isdigit():
            return f"Sector {int(number)}"

    return sector

@app.get("/api/risk/{sector}")
def risk_explanation(sector: str, rain_mm: float | None = Query(default=None, ge=0, le=500)):
    try:
        df = get_ranked_risk(rain_mm=rain_mm)

        normalized_sector = normalize_sector_input(sector)

        matches = df[
            df["sector"].astype(str).str.lower()
            == normalized_sector.lower()
        ]

        if matches.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Sector '{sector}' not found"
            )

        row = matches.iloc[0].to_dict()

        reasons = explain_risk(row)

        return {
            "sector": row["sector"],
            "risk_score": float(row["risk_score"]),
            "risk_level": row["risk_level"],
            "rainfall_component": float(row.get("rainfall_component", 0)),
            "historical_component": float(row.get("historical_component", 0)),
            "drain_condition_component": float(row.get("drain_condition_component", 0)),
            "road_importance_component": float(row.get("road_importance_component", 0)),
            "connectivity_component": float(row.get("connectivity_component", 0)),
            "reasons": reasons
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/api/vision/{sector}")
def node_photo(sector: str):
    try:
        normalized_sector = normalize_sector_input(sector)

        ranked = get_ranked_risk()
        matches = ranked[
            ranked["sector"].astype(str).str.lower()
            == normalized_sector.lower()
        ]
        risk_level = (
            str(matches.iloc[0]["risk_level"])
            if not matches.empty else "Medium"
        )

        photo = get_node_photo(normalized_sector, risk_level)

        if photo is None:
            raise HTTPException(
                status_code=404,
                detail=f"No photo found for sector '{sector}'"
            )

        return photo

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


def read_frame_exif(image_path: Path) -> dict:
    """Extract capture time + GPS from a frame's EXIF data, if embedded."""
    exif_data: dict = {"captured_at": None, "gps": None}
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS, TAGS

        with Image.open(image_path) as img:
            raw = img.getexif() or {}
            tags = {TAGS.get(k, k): v for k, v in raw.items()}

            if tags.get("DateTimeOriginal"):
                exif_data["captured_at"] = str(tags["DateTimeOriginal"])

            gps_raw = None
            if hasattr(raw, "get_ifd"):
                try:
                    gps_raw = raw.get_ifd(0x8825)
                except Exception:
                    gps_raw = None
            if not gps_raw:
                gps_raw = tags.get("GPSInfo")
            if gps_raw and not isinstance(gps_raw, int):
                gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}

                def to_degrees(value):
                    try:
                        d, m, s = (float(x) for x in value)
                        return d + m / 60.0 + s / 3600.0
                    except (TypeError, ValueError):
                        return None

                lat = to_degrees(gps.get("GPSLatitude"))
                lon = to_degrees(gps.get("GPSLongitude"))
                if lat is not None and lon is not None:
                    if str(gps.get("GPSLatitudeRef", "")).upper() == "S":
                        lat = -lat
                    if str(gps.get("GPSLongitudeRef", "")).upper() == "W":
                        lon = -lon
                    exif_data["gps"] = {
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                    }
    except Exception:
        pass
    return exif_data


@app.post("/api/vision/analyse")
async def analyse_upload(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename or "upload").suffix.lower()
        is_image = (file.content_type or "").startswith("image/") or suffix in {
            ".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".gif", ".tiff",
        }
        if not is_image:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must be an image."
            )

        if not suffix:
            suffix = ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        try:
            from backend.services.vision_service import classify_upload

            result = classify_upload(tmp_path)
            result["exif"] = read_frame_exif(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        result["filename"] = file.filename
        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/api/solutions")
def list_solutions():
    return {
        "count": len(SOLUTION_DEFS),
        "data": [
            {"key": key, **value}
            for key, value in SOLUTION_DEFS.items()
        ],
    }


@app.get("/api/intervention/{solution}/{sector}")
def run_solution_simulation(
    solution: str,
    sector: str,
    rain_mm: float | None = Query(default=None, ge=0, le=500),
):
    try:
        if solution not in SOLUTION_DEFS:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Unknown solution '{solution}'. "
                    f"Available: {', '.join(sorted(SOLUTION_DEFS))}"
                )
            )

        normalized_sector = normalize_sector_input(sector)

        result = simulate_intervention(
            normalized_sector, solution, rain_mm=rain_mm
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Sector '{sector}' not found"
            )

        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/api/priorities")
def priorities(
    limit: int = Query(default=10, ge=1, le=100),
    rain_mm: float | None = Query(default=None, ge=0, le=500),
):
    try:
        results = get_priority_list(limit, rain_mm=rain_mm)

        return {
            "count": len(results),
            "data": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/api/routes")
def routes(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float
):
    try:
        result = calculate_route(
            start_lat=start_lat,
            start_lon=start_lon,
            end_lat=end_lat,
            end_lon=end_lon
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---- Frontend serving (same-origin integration) ----
# Serves frontend/index.html at "/" so the dashboard and API
# share one origin (no CORS / hardcoded localhost needed).
if FRONTEND_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="frontend-assets",
    )


IMAGES_DIR = BASE_DIR / "data" / "images"

if IMAGES_DIR.exists():
    app.mount(
        "/images",
        StaticFiles(directory=str(IMAGES_DIR)),
        name="node-images",
    )


@app.get("/", include_in_schema=False)
def serve_frontend():
    if FRONTEND_INDEX.exists():
        return FileResponse(
            str(FRONTEND_INDEX),
            headers={"Cache-Control": "no-store"},
        )
    return {
        "message": "DrainSight backend is running"
    }