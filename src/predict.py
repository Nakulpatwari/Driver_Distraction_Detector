import tensorflow as tf
import numpy as np
from pathlib import Path
from src.config import CLASS_MAP, IMG_HEIGHT, IMG_WIDTH, MODELS_DIR

# Map integer index to class string and human-readable names
IDX_TO_CLASS = {int(k.replace('c', '')): v for k, v in CLASS_MAP.items()}

class DriverDistractionPredictor:
    def __init__(self, model_path=MODELS_DIR / "best_model.keras"):
        """
        Initializes the predictor with the trained model.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train a model first.")
        self.model = tf.keras.models.load_model(model_path)
        
    def _preprocess(self, image_path):
        """ 
        Identical preprocessing to data.py training pipeline to prevent concept drift.
        """
        image_string = tf.io.read_file(str(image_path))
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        
        image = tf.keras.applications.imagenet_utils.preprocess_input(image)
        image = image / 255.0
        
        # Add batch dimension
        return tf.expand_dims(image, 0)

    def predict(self, image_path, uncertainty_threshold=0.5):
        """
        Runs inference and returns dict with class prediction and confidence.
        Includes a low-confidence flag if softmax probability doesn't exceed threshold.
        """
        image_tensor = self._preprocess(image_path)
        preds = self.model.predict(image_tensor, verbose=0)[0]
        
        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id])
        
        result = {
            'class_id': f"c{class_id}",
            'class_name': IDX_TO_CLASS.get(class_id, "Unknown"),
            'confidence': confidence,
            'uncertain': confidence < uncertainty_threshold
        }
        return result
