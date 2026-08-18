import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import IMG_HEIGHT, IMG_WIDTH, CHANNELS, NUM_CLASSES

def build_baseline_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), num_classes=NUM_CLASSES):
    """
    Builds a deliberately simple CNN baseline.
    Architecture: Input -> Conv2D -> ReLU -> MaxPooling -> Conv2D -> ReLU -> MaxPooling ->
                  Conv2D -> ReLU -> MaxPooling -> GlobalAveragePooling -> Dense -> Dropout -> Dense -> Softmax
    """
    inputs = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    # The baseline should prove to be limited, encouraging overfitting analysis
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="Baseline_CNN")
    return model

def build_improved_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), num_classes=NUM_CLASSES, fine_tune=False):
    """
    Builds an improved model using MobileNetV2 and ImageNet pretrained weights.
    Stage 1: Frozen backbone, trains only the top classification head.
    Stage 2: Unfreezes upper layers and applies a substantially smaller learning rate (fine_tune=True).
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Default behavior: freeze the backbone (Stage 1)
    base_model.trainable = False
    
    # Stage 2: unfreeze a reasonable number of upper layers
    if fine_tune:
        base_model.trainable = True
        fine_tune_at = 100
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    
    # Note: Images have already been preprocessed natively in MobileNetV2 / ResNet style in data.py
    # We pass training=False to the base_model to keep BatchNorm layers in inference mode, ensuring safe fine-tuning.
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="Improved_MobileNetV2")
    return model
