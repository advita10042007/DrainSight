"""
Vision-based feature extraction for DrainSight.

This module provides lightweight computer-vision utilities for extracting
building, road, and water-related features from an image.

The functions are intentionally modular so the backend can call them later.
"""

from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk.

    Returns:
        OpenCV image as a NumPy array.

    Raises:
        FileNotFoundError: If the image does not exist or cannot be read.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    return image


def get_image_features(image_path: str) -> Dict[str, Any]:
    """
    Extract basic visual/geospatial proxy features from an image.

    These are baseline features for the prototype. More advanced ML
    detection can be plugged into this module later without changing
    the backend interface.
    """

    image = load_image(image_path)

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Basic image statistics
    mean_brightness = float(np.mean(gray))
    brightness_std = float(np.std(gray))

    # Edge density can act as a simple proxy for structural complexity.
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    return {
        "image_width": width,
        "image_height": height,
        "mean_brightness": mean_brightness,
        "brightness_std": brightness_std,
        "edge_density": edge_density,
    }


def detect_water_regions(image_path: str) -> Dict[str, Any]:
    """
    Estimate water-like regions using HSV colour thresholds.

    This is a baseline heuristic for the prototype and should not be
    interpreted as a trained water-segmentation model.
    """

    image = load_image(image_path)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Broad blue/cyan range.
    lower = np.array([80, 30, 30])
    upper = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    water_pixels = int(np.sum(mask > 0))
    total_pixels = mask.shape[0] * mask.shape[1]

    water_ratio = water_pixels / total_pixels if total_pixels else 0.0

    return {
        "water_pixel_count": water_pixels,
        "water_ratio": float(water_ratio),
    }


def extract_vision_features(image_path: str) -> Dict[str, Any]:
    """
    Main entry point for the vision feature pipeline.

    Returns a single dictionary so the backend can consume the result
    without knowing how individual features were calculated.
    """

    features = get_image_features(image_path)
    water_features = detect_water_regions(image_path)

    features.update(water_features)

    return features


if __name__ == "__main__":
    print("vision_features.py loaded successfully.")
