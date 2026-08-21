from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 10
SEED = 42


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "state_farm"
TRAIN_DIR = DATA_DIR / "imgs" / "train"

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"


# ============================================================
# Class mapping
# ============================================================

CLASS_NAMES = [
    "c0",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
    "c7",
    "c8",
    "c9",
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


# ============================================================
# Image loading
# ============================================================

def load_image(image_path, label):
    """
    Load an image and resize it.

    Output pixel range:
        [0, 255]
    """

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    return image, label


# ============================================================
# Data augmentation
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomRotation(
            factor=0.05
        ),

        layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10
        ),

        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        layers.RandomContrast(
            factor=0.10
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# Create dataset
# ============================================================

def create_dataset(
    dataframe,
    training=False,
    preprocessing="baseline"
):
    """
    Create a TensorFlow dataset.

    preprocessing options:

        baseline
            Converts pixels from [0,255] -> [0,1]

        mobilenet
            Converts pixels from [0,255] -> [-1,1]
            using MobileNetV2 preprocessing.
    """

    image_paths = [
        str(
            TRAIN_DIR / row["classname"] / row["img"]
        )
        for _, row in dataframe.iterrows()
    ]

    labels = [
        CLASS_TO_INDEX[class_name]
        for class_name in dataframe["classname"]
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            labels
        )
    )

    # --------------------------------------------------------
    # Shuffle training data
    # --------------------------------------------------------

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    if training:

        dataset = dataset.map(
            lambda image, label: (
                data_augmentation(
                    image,
                    training=True
                ),
                label
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # --------------------------------------------------------
    # Model-specific preprocessing
    # --------------------------------------------------------

    if preprocessing == "baseline":

        # Baseline CNN receives [0,1]
        dataset = dataset.map(
            lambda image, label: (
                image / 255.0,
                label
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    elif preprocessing == "mobilenet":

        # MobileNetV2 expects [0,255] and converts
        # it internally to approximately [-1,1].
        dataset = dataset.map(
            lambda image, label: (
                tf.keras.applications.mobilenet_v2.preprocess_input(
                    image
                ),
                label
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    else:

        raise ValueError(
            f"Unknown preprocessing mode: {preprocessing}"
        )

    # --------------------------------------------------------
    # Batch + prefetch
    # --------------------------------------------------------

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# Load datasets
# ============================================================

def load_datasets(
    preprocessing="baseline"
):

    train_df = pd.read_csv(
        SPLIT_DIR / "train.csv"
    )

    validation_df = pd.read_csv(
        SPLIT_DIR / "validation.csv"
    )

    test_df = pd.read_csv(
        SPLIT_DIR / "test.csv"
    )

    train_dataset = create_dataset(
        train_df,
        training=True,
        preprocessing=preprocessing
    )

    validation_dataset = create_dataset(
        validation_df,
        training=False,
        preprocessing=preprocessing
    )

    test_dataset = create_dataset(
        test_df,
        training=False,
        preprocessing=preprocessing
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )


# ============================================================
# Test pipeline
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING DATA PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Baseline preprocessing
    # --------------------------------------------------------

    print("\nTesting baseline preprocessing...")

    train_ds, val_ds, test_ds = load_datasets(
        preprocessing="baseline"
    )

    images, labels = next(
        iter(train_ds)
    )

    print("\nBaseline batch shape:")
    print(images.shape)

    print("Baseline pixel range:")

    print(
        float(tf.reduce_min(images)),
        "to",
        float(tf.reduce_max(images))
    )

    # --------------------------------------------------------
    # MobileNet preprocessing
    # --------------------------------------------------------

    print("\nTesting MobileNetV2 preprocessing...")

    train_ds, val_ds, test_ds = load_datasets(
        preprocessing="mobilenet"
    )

    images, labels = next(
        iter(train_ds)
    )

    print("\nMobileNet batch shape:")
    print(images.shape)

    print("MobileNet pixel range:")

    print(
        float(tf.reduce_min(images)),
        "to",
        float(tf.reduce_max(images))
    )

    print("\nNumber of classes:")
    print(NUM_CLASSES)

    print("\n✓ Data pipeline working correctly.")