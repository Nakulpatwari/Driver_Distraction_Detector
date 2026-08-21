from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "state_farm"
TRAIN_DIR = DATA_DIR / "imgs" / "train"

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Class names
# ============================================================

CLASS_NAMES = {
    "c0": "Safe driving",
    "c1": "Texting - right",
    "c2": "Talking phone - right",
    "c3": "Texting - left",
    "c4": "Talking phone - left",
    "c5": "Operating radio",
    "c6": "Drinking",
    "c7": "Reaching behind",
    "c8": "Hair / makeup",
    "c9": "Talking to passenger",
}


# ============================================================
# Load split data
# ============================================================

train_df = pd.read_csv(
    SPLIT_DIR / "train.csv"
)

validation_df = pd.read_csv(
    SPLIT_DIR / "validation.csv"
)

test_df = pd.read_csv(
    SPLIT_DIR / "test.csv"
)


# ============================================================
# 1. Class distribution
# ============================================================

class_counts = (
    train_df["classname"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=class_counts.index,
    y=class_counts.values
)

plt.title("Training Set Class Distribution")
plt.xlabel("Driver Behavior")
plt.ylabel("Number of Images")

plt.xticks(
    range(len(class_counts)),
    [
        CLASS_NAMES[c]
        for c in class_counts.index
    ],
    rotation=35,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "class_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# 2. Sample images from every class
# ============================================================

fig, axes = plt.subplots(
    2,
    5,
    figsize=(18, 8)
)

for index, class_code in enumerate(CLASS_NAMES):

    class_images = list(
        (TRAIN_DIR / class_code).glob("*.jpg")
    )

    image_path = class_images[0]

    image = Image.open(image_path)

    ax = axes[index // 5][index % 5]

    ax.imshow(image)
    ax.set_title(
        f"{class_code}\n{CLASS_NAMES[class_code]}"
    )

    ax.axis("off")


plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "class_examples.png",
    dpi=150
)

plt.close()


# ============================================================
# 3. Driver distribution
# ============================================================

driver_counts = (
    train_df["subject"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=driver_counts.index,
    y=driver_counts.values
)

plt.title("Training Images per Driver")
plt.xlabel("Driver ID")
plt.ylabel("Number of Images")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "driver_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# 4. Split summary
# ============================================================

summary = pd.DataFrame(
    {
        "split": [
            "train",
            "validation",
            "test"
        ],
        "images": [
            len(train_df),
            len(validation_df),
            len(test_df)
        ],
        "drivers": [
            train_df["subject"].nunique(),
            validation_df["subject"].nunique(),
            test_df["subject"].nunique()
        ]
    }
)

summary.to_csv(
    PROJECT_ROOT / "outputs" / "reports" / "dataset_summary.csv",
    index=False
)


# ============================================================
# Print report
# ============================================================

print("=" * 60)
print("EDA COMPLETE")
print("=" * 60)

print("\nDataset split:")

print(summary.to_string(index=False))

print("\nTraining class distribution:")

print(class_counts)

print("\nFigures generated:")

print("outputs/figures/class_distribution.png")
print("outputs/figures/class_examples.png")
print("outputs/figures/driver_distribution.png")

print("\n✓ EDA completed successfully.")