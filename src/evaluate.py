import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from src.config import MODELS_DIR, OUTPUTS_DIR, CLASS_MAP
from src.data import load_metadata, create_driver_split, load_dataset

CLASS_NAMES = [CLASS_MAP[f'c{i}'] for i in range(10)]

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plots and saves a classification confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()

def plot_training_curves(history_dict, output_dir):
    """Plots and saves the training accuracy and loss curves if history exists."""
    if not history_dict:
        return
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_dict.get('accuracy', []), label='Train Accuracy')
    plt.plot(history_dict.get('val_accuracy', []), label='Val Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_dict.get('loss', []), label='Train Loss')
    plt.plot(history_dict.get('val_loss', []), label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png')
    plt.close()

def evaluate_model(model_type='baseline'):
    """
    Evaluates the specified model against the driver-aware validation set.
    Outputs metrics JSON, classification report, and confusion matrices to output dir.
    """
    df = load_metadata()
    _, val_df = create_driver_split(df)
    val_ds = load_dataset(val_df, shuffle=False)
    
    model_name = "baseline_cnn.keras" if model_type == 'baseline' else "best_model.keras"
    model_path = MODELS_DIR / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
        
    print(f"Loading {model_type} model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Evaluating on Validation Data...")
    y_true = []
    for _, labels in val_ds:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)
        
    y_pred_probs = model.predict(val_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    out_dir = OUTPUTS_DIR / model_type
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate and Save classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    with open(out_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
        
    print("\nClassification Report:\n", report)
    
    # Calculate overarching metrics
    acc = np.mean(y_true == y_pred)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': float(acc),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1)
    }
    
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Generate Visualizations
    plot_confusion_matrix(y_true, y_pred, out_dir)
    
    history_file = out_dir / 'history.json'
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        plot_training_curves(history, out_dir)
        
    print(f"Evaluation artifacts saved successfully to {out_dir}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline or improved model.")
    parser.add_argument('--model', type=str, choices=['baseline', 'improved'], default='baseline', help="Model to evaluate")
    args = parser.parse_args()
    
    try:
        evaluate_model(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
