from pathlib import Path

import json
import tensorflow as tf

from src.preprocessing.data_loader import load_datasets


# ============================================================
# Configuration
# ============================================================

EPOCHS = 6
LEARNING_RATE = 1e-5
UNFREEZE_LAYERS = 30

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
# Main
# ============================================================

def main():

    print("=" * 60)
    print("MOBILENETV2 FINE-TUNING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_ds, val_ds, test_ds = load_datasets(
        preprocessing="mobilenet"
    )

    print("✓ Datasets loaded.")

    # --------------------------------------------------------
    # Load frozen model
    # --------------------------------------------------------

    frozen_model_path = (
        MODEL_DIR /
        "mobilenet_frozen_best.keras"
    )

    print("\nLoading frozen MobileNetV2...")

    model = tf.keras.models.load_model(
        frozen_model_path
    )

    print("✓ Model loaded.")

    # --------------------------------------------------------
    # Find MobileNetV2 backbone
    # --------------------------------------------------------

    base_model = None

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            base_model = layer
            break

    if base_model is None:

        raise RuntimeError(
            "Could not find MobileNetV2 backbone."
        )

    print(
        f"\nBackbone found: {base_model.name}"
    )

    print(
        f"Total backbone layers: {len(base_model.layers)}"
    )

    # --------------------------------------------------------
    # Freeze entire backbone first
    # --------------------------------------------------------

    base_model.trainable = True

    # Freeze all layers
    for layer in base_model.layers:

        layer.trainable = False

    # --------------------------------------------------------
    # Unfreeze final layers
    # --------------------------------------------------------

    for layer in base_model.layers[-UNFREEZE_LAYERS:]:

        # Keep BatchNorm frozen
        if not isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = True

    # --------------------------------------------------------
    # Print trainable statistics
    # --------------------------------------------------------

    trainable_layers = [
        layer
        for layer in base_model.layers
        if layer.trainable
    ]

    print(
        f"\nTrainable backbone layers: "
        f"{len(trainable_layers)}"
    )

    print(
        f"Frozen backbone layers: "
        f"{len(base_model.layers) - len(trainable_layers)}"
    )

    # --------------------------------------------------------
    # Compile with small learning rate
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

    print("\n✓ Model compiled.")

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    best_model_path = (
        MODEL_DIR /
        "mobilenet_finetuned_best.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            filepath=str(
                best_model_path
            ),

            monitor="val_accuracy",

            mode="max",

            save_best_only=True,

            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=2,

            restore_best_weights=True,

            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=1,

            min_lr=1e-7,

            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Fine-tune
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STARTING FINE-TUNING")
    print("=" * 60)

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
        "mobilenet_finetuned_final.keras"
    )

    model.save(
        final_model_path
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINE-TUNED VALIDATION PERFORMANCE")
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
    # Save history
    # --------------------------------------------------------

    history_path = (
        OUTPUT_DIR /
        "mobilenet_finetuned_history.json"
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
    print("FINE-TUNING COMPLETE")
    print("=" * 60)

    print(
        f"\nBest model: {best_model_path}"
    )

    print(
        f"Final model: {final_model_path}"
    )


if __name__ == "__main__":
    main()