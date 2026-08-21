import tensorflow as tf
from tensorflow.keras import layers, models


def build_mobilenet(
    input_shape=(224, 224, 3),
    num_classes=10
):

    # --------------------------------------------------------
    # Pretrained MobileNetV2 backbone
    # --------------------------------------------------------

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers initially
    base_model.trainable = False

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    inputs = layers.Input(
        shape=input_shape
    )

    x = base_model(
        inputs,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(
        0.3
    )(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.3
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="mobilenetv2_driver_distraction"
    )

    return model


if __name__ == "__main__":

    model = build_mobilenet()

    model.summary()