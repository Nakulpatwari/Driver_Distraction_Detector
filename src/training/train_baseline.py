from pathlib import Path

import tensorflow as tf

from src.preprocessing.data_loader import load_datasets
from src.models.baseline_cnn import build_baseline_cnn


# ============================================================
# Configuration
# ============================================================

EPOCHS = 15
LEARNING_RATE = 1e-3

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Main training pipeline
# ============================================================

def main():

    print("=" * 60)
    print("BASELINE CNN TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_ds, val_ds, test_ds = load_datasets(
    	preprocessing="baseline"
    )    
    print("✓ Datasets loaded.")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding baseline CNN...")

    model = build_baseline_cnn()

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    print("\nModel compiled successfully.")

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    checkpoint_path = (
        MODEL_DIR /
        "baseline_cnn_best.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_model_path = (
        MODEL_DIR /
        "baseline_cnn_final.keras"
    )

    model.save(
        final_model_path
    )

    # --------------------------------------------------------
    # Evaluate validation performance
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION PERFORMANCE")
    print("=" * 60)

    val_loss, val_accuracy = model.evaluate(
        val_ds,
        verbose=1
    )

    print(
        f"\nValidation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    import json

    history_path = (
        OUTPUT_DIR /
        "baseline_training_history.json"
    )

    with open(
        history_path,
        "w"
    ) as file:

        json.dump(
            history.history,
            file
        )

    print("\nTraining history saved.")

    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"\nBest model: {checkpoint_path}"
    )

    print(
        f"Final model: {final_model_path}"
    )


if __name__ == "__main__":
    main()