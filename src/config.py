import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IMGS_DIR = DATA_DIR / "imgs"
TRAIN_DIR = IMGS_DIR / "train"
TEST_DIR = IMGS_DIR / "test"
CSV_PATH = DATA_DIR / "driver_imgs_list.csv"

MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Hyperparameters
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 10

# Classes mapping
CLASS_MAP = {
    'c0': 'Safe Driving',
    'c1': 'Texting - Right',
    'c2': 'Talking on Phone - Right',
    'c3': 'Texting - Left',
    'c4': 'Talking on Phone - Left',
    'c5': 'Operating Radio',
    'c6': 'Drinking',
    'c7': 'Reaching Behind',
    'c8': 'Hair and Makeup',
    'c9': 'Talking to Passenger'
}

NUM_CLASSES = len(CLASS_MAP)
