import streamlit as st
from PIL import Image
import os
import sys

# Ensure src module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import DriverDistractionPredictor
from src.config import MODELS_DIR

st.set_page_config(page_title="Driver Distraction Detector", layout="centered", page_icon="🚗")

@st.cache_resource
def load_predictor():
    """ Load and cache the classifier so we don't repeat this each render loop """
    try:
        return DriverDistractionPredictor(model_path=MODELS_DIR / "best_model.keras")
    except FileNotFoundError:
        return None

st.title("🚗 Driver Distraction Detector")
st.markdown("Upload a dashboard camera image of a driver, and the model will predict what they are doing.")

predictor = load_predictor()

if predictor is None:
    st.error("Model not found. Please train the `improved` model first and ensure it is saved in `models/best_model.keras`.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load and Preview Image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Save temp file for prediction
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # Perform Inference
        # We ensure visual processing block only runs when calculating
        with st.spinner("Running inference..."):
            result = predictor.predict(temp_path)
            
        os.remove(temp_path)
        
        # Output Metrics
        st.subheader(f"Prediction: **{result['class_name']}**")
        st.write(f"Confidence: {result['confidence']:.2%}")
        
        # Handle Uncertainty Flags
        if result['uncertain']:
            st.warning("⚠️ Low confidence warning: The model is not highly certain about this prediction.")
        else:
            st.success("✅ Prediction made with high confidence.")
