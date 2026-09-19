import pandas as pd
from pathlib import Path

# Paths
METADATA_FILE = Path("data/raw/vision_metadata.csv")
IMAGE_DIR = Path("data/images")

print("=== DrainSight Vision Dataset Validation ===\n")

# Load metadata
df = pd.read_csv(METADATA_FILE)

print("Metadata rows:", len(df))
print("Metadata columns:")
print(list(df.columns))

# Count images
image_files = [
    p for p in IMAGE_DIR.rglob("*")
    if p.is_file()
]

print("\nActual image files:", len(image_files))

# Show labels
print("\nLabels found:")
print(df["label"].value_counts())

# Check duplicate IDs
duplicate_ids = df[df["image_id"].duplicated(keep=False)]

print("\nDuplicate image IDs:", len(duplicate_ids))

if len(duplicate_ids) > 0:
    print(duplicate_ids[["image_id", "filename"]])

# Check duplicate filenames
duplicate_filenames = df[df["filename"].duplicated(keep=False)]

print("\nDuplicate metadata filenames:", len(duplicate_filenames))

if len(duplicate_filenames) > 0:
    print(duplicate_filenames[["filename", "image_id"]])

# Check expected labels
expected_labels = {
    "blocked",
    "partially_blocked",
    "standing_water",
    "clear"
}

invalid_labels = set(df["label"].dropna()) - expected_labels

print("\nInvalid labels:", invalid_labels)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

print("\n=== Validation complete ===")

# Check metadata filenames against actual image filenames

actual_filenames = {
    p.stem
    for p in image_files
}

metadata_filenames = set(
    df["filename"]
    .astype(str)
    .str.strip()
)

missing_images = metadata_filenames - actual_filenames

print("\nMetadata filenames with no matching image:")
print(missing_images)

print("\nNumber of unmatched metadata filenames:", len(missing_images))