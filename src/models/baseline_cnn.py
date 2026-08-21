import tensorflow as tf
from tensorflow.keras import layers, models


def build_baseline_cnn(
    input_shape=(224, 224, 3),
    num_classes=10
):

    model = models.Sequential(
        [

            layers.Input(
                shape=input_shape
            ),

            # Block 1
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same"
            ),

            layers.MaxPooling2D(
                (2, 2)
            ),

            # Block 2
            layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same"
            ),

            layers.MaxPooling2D(
                (2, 2)
            ),

            # Block 3
            layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same"
            ),

            layers.MaxPooling2D(
                (2, 2)
            ),

            # Feature aggregation
            layers.GlobalAveragePooling2D(),

            layers.Dropout(
                0.3
            ),

            layers.Dense(
                128,
                activation="relu"
            ),

            layers.Dropout(
                0.3
            ),

            # Classification
            layers.Dense(
                num_classes,
                activation="softmax"
            )
        ],

        name="baseline_cnn"
    )

    return model


if __name__ == "__main__":

    model = build_baseline_cnn()

    model.summary()