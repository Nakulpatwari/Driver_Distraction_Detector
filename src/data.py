import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
from src.config import CSV_PATH, TRAIN_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

def load_metadata(csv_path=CSV_PATH):
    """Loads the dataset metadata from the CSV file."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Metadata file not found at {csv_path}. Please add the dataset according to data/README.md.")
    df = pd.read_csv(csv_path)
    # Ensure correct mapping if classes are integer or strings
    if df['classname'].dtype == object and df['classname'].str.startswith('c').all():
        pass # Expected format
    return df

def create_driver_split(df, test_size=0.2, random_state=42):
    """
    Creates a driver-aware train/validation split using GroupShuffleSplit.
    This ensures images from the same driver don't appear in both train and validation sets,
    preventing data leakage.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df['subject']))
    return df.iloc[train_idx], df.iloc[val_idx]

def _parse_function(filename, label):
    """Reads an image from a file, decodes it, and preprocesses it identically to the notebook handling."""
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    
    # Notebook preprocessing method
    image = tf.keras.applications.imagenet_utils.preprocess_input(image)
    image = image / 255.0
    
    return image, label

def load_dataset(df, img_dir=TRAIN_DIR, batch_size=BATCH_SIZE, shuffle=True):
    """
    Creates a tf.data.Dataset from a DataFrame over driver images.
    Robustly handles images organized either flatly (notebook style) or in class subfolders (Kaggle style).
    """
    if not Path(img_dir).exists():
        raise FileNotFoundError(f"Image directory not found at {img_dir}. Please follow data/README.md instructions.")
        
    filepaths = []
    for _, row in df.iterrows():
        class_folder_path = Path(img_dir) / row['classname'] / row['img']
        flat_path = Path(img_dir) / row['img']
        if class_folder_path.exists():
            filepaths.append(str(class_folder_path))
        else:
            filepaths.append(str(flat_path))
            
    if 'classname' in df.columns:
        labels = df['classname'].astype(str).str.replace('c', '').astype(int).values
        dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    else:
        # Inference context where we just have images without labels
        dataset = tf.data.Dataset.from_tensor_slices((filepaths, np.zeros(len(filepaths))))
        
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

if __name__ == "__main__":
    try:
        df = load_metadata()
        train_df, val_df = create_driver_split(df)
        print(f"Total images: {len(df)}")
        print(f"Training images: {len(train_df)} ({len(train_df['subject'].unique())} drivers)")
        print(f"Validation images: {len(val_df)} ({len(val_df['subject'].unique())} drivers)")
    except Exception as e:
        print(f"Setup warning: {e}")
