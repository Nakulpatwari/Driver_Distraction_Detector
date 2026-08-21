from pathlib import Path
import pandas as pd
from PIL import Image


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "state_farm"
TRAIN_DIR = DATA_DIR / "imgs" / "train"
DRIVER_CSV = DATA_DIR / "driver_imgs_list.csv"


# --------------------------------------------------
# 1. Basic dataset information
# --------------------------------------------------

print("=" * 60)
print("DRIVER DISTRACTION DATASET AUDIT")
print("=" * 60)

print(f"\nDataset directory: {DATA_DIR}")
print(f"Train directory:   {TRAIN_DIR}")


# --------------------------------------------------
# 2. Class distribution
# --------------------------------------------------

classes = sorted(
    [directory.name for directory in TRAIN_DIR.iterdir() if directory.is_dir()]
)

print("\nClasses:")
print(classes)

class_counts = {}

for class_name in classes:
    class_dir = TRAIN_DIR / class_name
    images = list(class_dir.glob("*.jpg"))
    class_counts[class_name] = len(images)

class_distribution = pd.Series(class_counts)

print("\nImages per class:")
print(class_distribution)

print(f"\nTotal training images: {class_distribution.sum()}")


# --------------------------------------------------
# 3. Driver metadata
# --------------------------------------------------

print("\n" + "=" * 60)
print("DRIVER METADATA")
print("=" * 60)

driver_df = pd.read_csv(DRIVER_CSV)

print("\nCSV columns:")
print(driver_df.columns.tolist())

print("\nFirst 5 rows:")
print(driver_df.head())

print(f"\nUnique drivers: {driver_df['subject'].nunique()}")

print("\nImages per driver:")
print(driver_df["subject"].value_counts().sort_index())


# --------------------------------------------------
# 4. Drivers per class
# --------------------------------------------------

print("\nUnique drivers per class:")

drivers_per_class = (
    driver_df.groupby("classname")["subject"]
    .nunique()
    .sort_index()
)

print(drivers_per_class)


# --------------------------------------------------
# 5. Image dimensions
# --------------------------------------------------

print("\n" + "=" * 60)
print("IMAGE DIMENSIONS")
print("=" * 60)

sample_images = []

for class_name in classes:
    class_dir = TRAIN_DIR / class_name

    first_image = next(class_dir.glob("*.jpg"), None)

    if first_image:
        with Image.open(first_image) as img:
            sample_images.append({
                "class": class_name,
                "image": first_image.name,
                "width": img.width,
                "height": img.height,
                "mode": img.mode
            })

dimensions_df = pd.DataFrame(sample_images)

print(dimensions_df.to_string(index=False))


# --------------------------------------------------
# 6. Corrupted image check
# --------------------------------------------------

print("\n" + "=" * 60)
print("CHECKING FOR CORRUPTED IMAGES")
print("=" * 60)

# Instead of opening every image, perform a quick
# file-level check first.

all_images = list(TRAIN_DIR.rglob("*.jpg"))

print(f"\nTotal image files found: {len(all_images)}")

corrupted_images = []

# Quick verification using PIL
for i, image_path in enumerate(all_images):

    try:
        with Image.open(image_path) as img:
            img.load()

    except Exception:
        corrupted_images.append(str(image_path))

    # Progress update every 1000 images
    if (i + 1) % 1000 == 0:
        print(f"Checked {i + 1}/{len(all_images)} images")

print(f"\nCorrupted images found: {len(corrupted_images)}")

if corrupted_images:
    print("\nCorrupted files:")
    for image in corrupted_images[:20]:
        print(image)
# --------------------------------------------------
# 7. Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)

print(f"Total images : {class_distribution.sum()}")
print(f"Total classes: {len(classes)}")
print(f"Total drivers: {driver_df['subject'].nunique()}")
print(f"Corrupt images: {len(corrupted_images)}")