# 🚗 Driver Distraction Detector

A deep learning based driver distraction detection system that analyzes driving videos and identifies potentially unsafe driver activities.

The system uses a fine-tuned MobileNetV2 model to classify driver behavior at the frame level and aggregates the predictions to generate a video-level distraction summary.

## Features

- 🎥 Driving video upload
- 🖼️ Frame-level activity classification
- 🧠 MobileNetV2 transfer learning
- 📊 Confidence scores for predictions
- 🚨 Automatic distraction detection
- 📈 Distraction percentage calculation
- ⏱️ Activity timeline
- 🌐 Interactive Streamlit interface
- 🎞️ Annotated video generation

## Driver Activities

The model recognizes 10 activities:

| Class | Activity |
|---|---|
| c0 | Safe driving |
| c1 | Texting - right hand |
| c2 | Talking on phone - right hand |
| c3 | Texting - left hand |
| c4 | Talking on phone - left hand |
| c5 | Operating radio |
| c6 | Drinking |
| c7 | Reaching behind |
| c8 | Hair / makeup |
| c9 | Talking to passenger |

## Architecture

```text
Driving Video
      ↓
OpenCV Video Processing
      ↓
Uniform Frame Sampling
      ↓
Image Preprocessing
      ↓
Fine-Tuned MobileNetV2
      ↓
Frame-Level Predictions
      ↓
Activity Aggregation
      ↓
Video-Level Distraction Analysis
      ↓
Streamlit Dashboard
```

Model

The project uses MobileNetV2 with transfer learning.

The model was initially trained with the convolutional base frozen and subsequently fine-tuned to adapt the pretrained features to driver behavior classification.

Evaluation

The fine-tuned model achieved approximately:

Validation Accuracy: 59.74%
Test Accuracy: 57.11%
Test Macro F1: 55.27%

Performance varies across individual driver activity classes.

Example

For an uploaded driving video, the application produces results such as:

Safe driving: 2 frames
Talking on phone - right hand: 13 frames


Dominant Activity:
Talking on phone - right hand


Distraction Detected: YES


Distraction Percentage:

86.67%


Project Structure


Driver_Distraction_Detector/
│
├── app/
│   └── app.py
│
├── data/
│
├── models/
│
├── outputs/
│
├── src/
│   ├── analysis/
│   ├── evaluation/
│   ├── explainability/
│   └── inference/
│
├── requirements.txt
├── README.md
└── .gitignore

Technologies

Python

TensorFlow / Keras

MobileNetV2

OpenCV

NumPy

Scikit-learn

Streamlit
