"""Vision service: map sectors to drain photos and analyse them in real time.

Image database: data/raw/vision_metadata.csv + data/images/<label>/...
Each row has an id, filename, human label (blocked / partially_blocked /
standing_water / clear), location text and confidence.

Matching is honest about provenance:
- "exact"      — the sector number appears in the photo's location/filename
- "nearby"     — the photo is from a known corridor of that sector
- "representative" — no photo for the sector; closest severity fallback
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.vision_features import extract_vision_features


BASE_DIR = Path(__file__).resolve().parents[2]
IMAGES_DIR = BASE_DIR / "data" / "images"
METADATA_PATH = BASE_DIR / "data" / "raw" / "vision_metadata.csv"

# Approximate corridor mapping for named locations (used only as
# "nearby" references, never presented as exact node photos).
CORRIDOR_SECTORS = {
    "palam vihar": "Sector 3",
    "hero honda": "Sector 10",
    "badshapur": "Sector 4",
    "badashapur": "Sector 4",
    "tulip": "Sector 69",
    "vatika": "Sector 49",
    "sohna": "Sector 48",
    "sohana": "Sector 48",
    "cyber": "Sector 18",
    "dlf": "Sector 18",
    "nh48": "Sector 10",
    "nh8": "Sector 33",
}

# Human-label severity baselines (percent blocked).
LABEL_BASELINE = {
    "blocked": 80.0,
    "partially_blocked": 45.0,
    "standing_water": 55.0,
    "clear": 8.0,
}

# Fallback severity per risk level when no sector photo exists.
RISK_FALLBACK_LABEL = {
    "High": "blocked",
    "Medium": "partially_blocked",
    "Low": "clear",
}


@lru_cache(maxsize=1)
def load_vision_metadata() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Vision metadata not found: {METADATA_PATH}")
    return pd.read_csv(METADATA_PATH)


def normalize_sector(sector: str) -> str:
    text = str(sector).strip()
    if text.isdigit():
        return f"Sector {int(text)}"
    match = re.match(r"^sector\s+0*(\d+)$", text, re.IGNORECASE)
    if match:
        return f"Sector {int(match.group(1))}"
    return text


def sector_number(sector: str) -> str | None:
    match = re.search(r"(\d+)", normalize_sector(sector))
    return match.group(1).lstrip("0") or "0" if match else None


def resolve_image_file(filename: str) -> Path | None:
    """Find the actual file for a metadata filename (extension may vary)."""
    for label_dir in ("blocked", "partially_blocked", "standing_water", "clear"):
        directory = IMAGES_DIR / label_dir
        if not directory.exists():
            continue
        exact = directory / filename
        if exact.is_file():
            return exact
        for candidate in directory.iterdir():
            if candidate.stem == filename:
                return candidate
    return None


def find_photo_row(sector: str, risk_level: str = "Medium"):
    """Return (row_dict, match_type) for the best photo for a sector."""
    df = load_vision_metadata()
    number = sector_number(sector)

    # 1. Exact: sector number in location or filename.
    if number:
        pattern = re.compile(rf"(?:sector\D*0*{number}\b|0*{number}\b)", re.IGNORECASE)
        for _, row in df.iterrows():
            haystack = f"{row.get('location', '')} {row.get('filename', '')}"
            if re.search(rf"sector\D*0*{number}\b", haystack, re.IGNORECASE):
                return row.to_dict(), "exact"
        # bare number only counts if paired with sector-like context
        _ = pattern

    # 2. Nearby: known corridor keyword.
    for _, row in df.iterrows():
        location = str(row.get("location", "")).lower()
        for keyword, mapped in CORRIDOR_SECTORS.items():
            if keyword in location and normalize_sector(mapped).lower() == normalize_sector(sector).lower():
                return row.to_dict(), "nearby"

    # 3. Representative fallback: same severity, deterministic pick.
    wanted = RISK_FALLBACK_LABEL.get(risk_level, "partially_blocked")
    matches = df[df["label"] == wanted].sort_values("image_id")
    if not matches.empty:
        return matches.iloc[0].to_dict(), "representative"
    first = df.sort_values("image_id").iloc[0].to_dict()
    return first, "representative"


def analyse_photo(image_path: Path, label: str) -> dict:
    """Analyse a database node photo in real time.

    The photo's field-verified human label anchors severity (these
    labels were confirmed on site), while pixel measurements are
    reported as supporting evidence and nudge the estimate by ±5.
    Sky is masked out (top third) so clouds don't read as water.
    """
    features = extract_vision_features(str(image_path))
    water_pct = round(_masked_water_pct(image_path) * 100, 1)
    baseline = LABEL_BASELINE.get(label, 40.0)
    nudge = 5.0 if water_pct > 25.0 else (-5.0 if water_pct < 5.0 else 0.0)
    blockage = round(max(0.0, min(100.0, baseline + nudge)), 1)
    return {
        "water_coverage_pct": water_pct,
        "edge_density": round(float(features.get("edge_density", 0.0)), 4),
        "mean_brightness": round(float(features.get("mean_brightness", 0.0)), 1),
        "image_width": int(features.get("image_width", 0)),
        "image_height": int(features.get("image_height", 0)),
        "blockage_estimate_pct": blockage,
        "severity_source": "field-verified label + pixel evidence",
    }


def _masked_water_pct(image_path: Path) -> float:
    """Blue-water fraction over the lower two thirds of the frame."""
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        return 0.0
    low = image[int(image.shape[0] * 0.33):]
    hsv = cv2.cvtColor(low, cv2.COLOR_BGR2HSV)
    lower = np.array([80, 30, 30])
    upper = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    total = mask.shape[0] * mask.shape[1]
    return float(np.sum(mask > 0) / total) if total else 0.0


def classify_upload(image_path: Path) -> dict:
    """Classify a user-uploaded frame with the real-time pipeline.

    No field label exists for uploads, so severity comes from pixels
    alone: dark debris texture plus sky-masked dark-water fraction
    over the lower two thirds of the frame. Calibrated directionally
    on the 26-photo database (no blocked drain scores as clear);
    uploads are always reported as unverified with capped confidence.
    """
    import cv2

    features = extract_vision_features(str(image_path))
    image = cv2.imread(str(image_path))
    low = image[int(image.shape[0] * 0.33):]
    hsv = cv2.cvtColor(low, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    dark_water = float((((H >= 80) & (H <= 140) & (S >= 30) & (V >= 30) & (V < 150))).mean())
    dark_ratio = float((V < 70).mean())

    water_pct = round(dark_water * 100, 1)
    score = 100.0 * (0.5 * min(1.0, dark_ratio * 3.0) + 0.5 * min(1.0, dark_water * 6.0))

    if score >= 50.0:
        label, confidence = "blocked", "Medium (unverified frame)"
    elif score >= 32.0:
        label, confidence = "partially_blocked", "Medium (unverified frame)"
    elif score >= 15.0:
        label, confidence = "standing_water", "Low (unverified frame)"
    else:
        label, confidence = "clear", "Low (unverified frame)"

    return {
        "label": label,
        "water_coverage_pct": water_pct,
        "edge_density": round(float(features.get("edge_density", 0.0)), 4),
        "mean_brightness": round(float(features.get("mean_brightness", 0.0)), 1),
        "image_width": int(features.get("image_width", 0)),
        "image_height": int(features.get("image_height", 0)),
        "blockage_estimate_pct": round(max(3.0, min(95.0, score)), 1),
        "confidence": confidence,
        "severity_source": "pixel estimate only — unverified frame",
    }


def get_node_photo(sector: str, risk_level: str = "Medium") -> dict | None:
    row, match_type = find_photo_row(sector, risk_level)
    if not row:
        return None
    image_path = resolve_image_file(str(row.get("filename", "")))
    if image_path is None:
        return None
    label = str(row.get("label", "unknown"))
    analysis = analyse_photo(image_path, label)
    relative = image_path.relative_to(IMAGES_DIR).as_posix()
    return {
        "sector": normalize_sector(sector),
        "image_id": str(row.get("image_id", "")),
        "image_url": f"/images/{relative}",
        "label": label,
        "location": str(row.get("location", "")).strip(),
        "description": str(row.get("description", "")).strip(),
        "confidence": str(row.get("confidence", "")).strip(),
        "source": str(row.get("source", "")).strip(),
        "match_type": match_type,
        "analysis": analysis,
    }
