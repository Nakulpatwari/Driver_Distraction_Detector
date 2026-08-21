from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "state_farm"
TRAIN_DIR = DATA_DIR / "imgs" / "train"
DRIVER_CSV = DATA_DIR / "driver_imgs_list.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

df = pd.read_csv(DRIVER_CSV)

print("=" * 60)
print("CREATING DRIVER-AWARE DATA SPLIT")
print("=" * 60)

print(f"\nRaw metadata rows: {len(df)}")
print(f"Unique image names: {df['img'].nunique()}")

# --------------------------------------------------
# Remove duplicate metadata records
# --------------------------------------------------

duplicate_count = df.duplicated(subset=["img"]).sum()

print(f"Duplicate image records: {duplicate_count}")

df = df.drop_duplicates(subset=["img"]).copy()

print(f"Metadata rows after deduplication: {len(df)}")

# --------------------------------------------------
# Verify metadata consistency
# --------------------------------------------------

actual_images = list(TRAIN_DIR.rglob("*.jpg"))

print(f"Actual image files: {len(actual_images)}")

assert len(df) == len(df["img"].unique()), \
    "Duplicate image records still exist."

assert len(df) == len(actual_images), \
    "Metadata count does not match actual image count."

print("\n✓ Metadata matches the actual image dataset.")

print(f"\nTotal images: {len(df)}")
print(f"Total drivers: {df['subject'].nunique()}")
print(f"Total classes: {df['classname'].nunique()}")
# --------------------------------------------------
# First split:
# 85% development / 15% test
# --------------------------------------------------

splitter_test = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42
)

train_val_idx, test_idx = next(
    splitter_test.split(
        df,
        groups=df["subject"]
    )
)

train_val_df = df.iloc[train_val_idx].copy()
test_df = df.iloc[test_idx].copy()


# --------------------------------------------------
# Second split:
# 70% train / 15% validation / 15% test
# --------------------------------------------------

splitter_val = GroupShuffleSplit(
    n_splits=1,
    test_size=0.1765,
    random_state=42
)

train_idx, val_idx = next(
    splitter_val.split(
        train_val_df,
        groups=train_val_df["subject"]
    )
)

train_df = train_val_df.iloc[train_idx].copy()
val_df = train_val_df.iloc[val_idx].copy()


# --------------------------------------------------
# Save splits
# --------------------------------------------------

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


# --------------------------------------------------
# Display split information
# --------------------------------------------------

print("\n" + "=" * 60)
print("SPLIT SUMMARY")
print("=" * 60)

for name, split in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    print(f"\n{name}")
    print("-" * 40)

    print(f"Images : {len(split)}")
    print(f"Drivers: {split['subject'].nunique()}")

    print(
        "Drivers:",
        sorted(split["subject"].unique())
    )

    print("\nClass distribution:")
    print(split["classname"].value_counts().sort_index())


# --------------------------------------------------
# Verify no driver leakage
# --------------------------------------------------

train_drivers = set(train_df["subject"])
val_drivers = set(val_df["subject"])
test_drivers = set(test_df["subject"])

print("\n" + "=" * 60)
print("LEAKAGE CHECK")
print("=" * 60)

print(
    "Train ∩ Validation:",
    train_drivers.intersection(val_drivers)
)

print(
    "Train ∩ Test:",
    train_drivers.intersection(test_drivers)
)

print(
    "Validation ∩ Test:",
    val_drivers.intersection(test_drivers)
)

if (
    train_drivers.isdisjoint(val_drivers)
    and
    train_drivers.isdisjoint(test_drivers)
    and
    val_drivers.isdisjoint(test_drivers)
):

    print("\n✓ No driver leakage detected.")

else:

    raise RuntimeError(
        "Driver leakage detected!"
    )

print("\nSplits saved to:")
print(OUTPUT_DIR)