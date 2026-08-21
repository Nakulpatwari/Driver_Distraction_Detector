from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mobilenet_finetuned_best.keras"
)


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


CLASS_LABELS = {
    "c0": "Safe driving",
    "c1": "Texting - right hand",
    "c2": "Talking on phone - right hand",
    "c3": "Texting - left hand",
    "c4": "Talking on phone - left hand",
    "c5": "Operating radio",
    "c6": "Drinking",
    "c7": "Reaching behind",
    "c8": "Hair / makeup",
    "c9": "Talking to passenger",
}


def load_model():
    """
    Load the trained MobileNetV2 model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("✓ Model loaded.")

    return model


def preprocess_frames(frames):
    """
    Prepare OpenCV-extracted RGB frames
    for the trained MobileNetV2 model.

    Input:
        frames -> list of RGB numpy arrays

    Output:
        numpy array of shape:
        (N, 224, 224, 3)
    """

    processed = []

    for frame in frames:

        # Resize to training resolution
        frame = tf.image.resize(
            frame,
            (224, 224)
        )

        # Convert to float32
        frame = tf.cast(
            frame,
            tf.float32
        )

        # MobileNetV2 preprocessing
        # Converts [0, 255] → [-1, 1]
        frame = tf.keras.applications.mobilenet_v2.preprocess_input(
            frame
        )

        processed.append(
            frame.numpy()
        )

    return np.array(
        processed,
        dtype=np.float32
    )


def predict_frames(model, frames):
    """
    Generate predictions for all sampled frames.

    Returns:
        probabilities
        predicted_indices
    """

    batch = preprocess_frames(
        frames
    )

    probabilities = model.predict(
        batch,
        verbose=0
    )

    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    return (
        probabilities,
        predicted_indices
    )


def decode_prediction(index):
    """
    Convert class index into human-readable label.
    """

    class_name = CLASS_NAMES[index]

    return (
        class_name,
        CLASS_LABELS[class_name]
    )