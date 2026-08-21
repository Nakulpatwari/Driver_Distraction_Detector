from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.preprocessing.data_loader import load_datasets


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "baseline_cnn_best.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "metrics"
)

FIGURE_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Class names
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


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("BASELINE MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load best saved model
    # --------------------------------------------------------

    print("\nLoading best baseline model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("✓ Best model loaded.")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    _, validation_ds, test_ds = load_datasets(
        preprocessing="baseline"
    )

    print("✓ Datasets loaded.")

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BEST MODEL - VALIDATION")
    print("=" * 60)

    val_loss, val_accuracy = model.evaluate(
        validation_ds,
        verbose=1
    )

    print(
        f"\nValidation loss: {val_loss:.4f}"
    )

    print(
        f"Validation accuracy: {val_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BEST MODEL - TEST")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1
    )

    print(
        f"\nTest loss: {test_loss:.4f}"
    )

    print(
        f"Test accuracy: {test_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating test predictions...")

    y_true = []
    y_pred = []

    for images, labels in test_ds:

        predictions = model.predict(
            images,
            verbose=0
        )

        predicted_classes = np.argmax(
            predictions,
            axis=1
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            predicted_classes
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --------------------------------------------------------
    # Macro F1
    # --------------------------------------------------------

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    print(
        f"\nTest Macro F1: {macro_f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(report)

    # Save report

    report_path = (
        OUTPUT_DIR /
        "baseline_classification_report.txt"
    )

    with open(
        report_path,
        "w"
    ) as file:

        file.write(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_path = (
        OUTPUT_DIR /
        "baseline_confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_path
    )

    print(
        f"\nConfusion matrix saved to:"
    )

    print(cm_path)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        {
            "metric": [
                "validation_accuracy",
                "test_accuracy",
                "test_macro_f1"
            ],
            "value": [
                val_accuracy,
                test_accuracy,
                macro_f1
            ]
        }
    )

    summary.to_csv(
        OUTPUT_DIR /
        "baseline_results.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()