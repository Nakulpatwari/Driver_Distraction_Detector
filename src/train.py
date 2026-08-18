import argparse
import tensorflow as tf
from pathlib import Path
import json

from src.config import EPOCHS, MODELS_DIR, OUTPUTS_DIR
from src.data import load_metadata, create_driver_split, load_dataset
from src.model import build_baseline_cnn, build_improved_model

def train_model(model_type='baseline', fine_tune=False):
    """
    Trains the specified model type (baseline or improved) using the driver-split dataset.
    """
    df = load_metadata()
    train_df, val_df = create_driver_split(df)
    
    train_ds = load_dataset(train_df, shuffle=True)
    val_ds = load_dataset(val_df, shuffle=False)
    
    if model_type == 'baseline':
        model = build_baseline_cnn()
        model_save_path = MODELS_DIR / "baseline_cnn.keras"
        lr = 0.001
    elif model_type == 'improved':
        model = build_improved_model(fine_tune=fine_tune)
        model_save_path = MODELS_DIR / "best_model.keras"
        lr = 1e-5 if fine_tune else 0.001
    else:
        raise ValueError("Invalid model type")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(model_save_path), save_best_only=True)
    ]
    
    print(f"Training {model_type} model with architecture:")
    model.summary()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Save training history for later analysis
    history_file = OUTPUTS_DIR / model_type / "history.json"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, 'w') as f:
        json.dump(history.history, f)
        
    print(f"Model successfully trained and saved at {model_save_path}")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline or improved model.")
    parser.add_argument('--model', type=str, choices=['baseline', 'improved'], default='baseline', help="Type of model to train")
    parser.add_argument('--fine_tune', action='store_true', help="Use small learning rate and unfreeze layers for Stage 2 of transfer learning (improved model only).")
    args = parser.parse_args()
    
    try:
        train_model(model_type=args.model, fine_tune=args.fine_tune)
    except FileNotFoundError as e:
        print(f"Error: {e}")
