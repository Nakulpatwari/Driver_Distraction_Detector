import tempfile
from pathlib import Path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import streamlit as st

from src.inference.video_analyzer import analyze_video


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Driver Distraction Detector",
    page_icon="🚗",
    layout="wide",
)


# ============================================================
# Title
# ============================================================

st.title("🚗 Driver Distraction Detector")

st.write(
    "Upload a driving video to detect potentially "
    "distracted driving activities."
)


# ============================================================
# Video Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a driving video",
    type=["mp4", "avi", "mov"],
)


# ============================================================
# Analyze Video
# ============================================================

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button(
        "Analyze Video",
        type="primary",
    ):

        # Create temporary file
        suffix = Path(
            uploaded_file.name
        ).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_video_path = Path(
                temp_file.name
            )

        # Run inference
        with st.spinner(
            "Analyzing video... Please wait."
        ):

            try:

                results = analyze_video(
                    temp_video_path,
                    num_frames=15,
                )

                st.success(
                    "Video analysis completed!"
                )

                # ------------------------------------------------
                # Video information
                # ------------------------------------------------

                video_info = results[
                    "video_info"
                ]

                st.subheader(
                    "Video Information"
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Duration",
                    f"{video_info['duration']:.2f} s",
                )

                col2.metric(
                    "FPS",
                    f"{video_info['fps']:.2f}",
                )

                col3.metric(
                    "Frames",
                    video_info["frame_count"],
                )

                col4.metric(
                    "Resolution",
                    (
                        f"{video_info['width']} × "
                        f"{video_info['height']}"
                    ),
                )

                # ------------------------------------------------
                # Frame predictions
                # ------------------------------------------------

                st.subheader(
                    "Frame-Level Predictions"
                )

                for result in results[
                    "frame_results"
                ]:

                    confidence = (
                        result["confidence"]
                        * 100
                    )

                    st.write(
                        f"**Frame "
                        f"{result['sample_number']:02d}** | "
                        f"{result['timestamp']:.2f}s | "
                        f"{result['label']} | "
                        f"{confidence:.2f}%"
                    )
                                # ====================================================
                # VIDEO-LEVEL SUMMARY
                # ====================================================

                st.subheader(
                    "Video-Level Summary"
                )

                frame_results = results[
                    "frame_results"
                ]

                # Count predicted activities
                activity_counts = {}

                for result in frame_results:

                    label = result["label"]

                    if label not in activity_counts:
                        activity_counts[label] = 0

                    activity_counts[label] += 1

                st.write("### Activity Counts")

                for activity, count in activity_counts.items():

                    st.write(
                        f"**{activity}**: {count} frames"
                    )

                # Find dominant activity
                dominant_activity = max(
                    activity_counts,
                    key=activity_counts.get
                )

                st.write(
                    f"**Dominant Activity:** "
                    f"{dominant_activity}"
                )

                # Calculate distraction percentage
                safe_frames = activity_counts.get(
                    "Safe driving",
                    0
                )

                total_frames = len(
                    frame_results
                )

                distracted_frames = (
                    total_frames - safe_frames
                )

                distraction_percentage = (
                    distracted_frames
                    / total_frames
                    * 100
                    if total_frames > 0
                    else 0
                )

                # ====================================================
                # DISTRACTION STATUS
                # ====================================================

                st.write("### Driver Status")

                if distracted_frames > 0:

                    st.error(
                        "⚠️ DISTRACTION DETECTED"
                    )

                else:

                    st.success(
                        "✅ DRIVER APPEARS SAFE"
                    )

                st.metric(
                    "Distraction Percentage",
                    f"{distraction_percentage:.2f}%"
                )
                # ====================================================
                # ACTIVITY TIMELINE
                # ====================================================

                st.subheader("Activity Timeline")

                timeline = []

                current_label = None
                start_time = None
                previous_time = None

                for result in frame_results:

                    label = result["label"]
                    timestamp = result["timestamp"]

                    if current_label is None:
                        current_label = label
                        start_time = timestamp

                    elif label != current_label:

                        timeline.append(
                            {
                                "start": start_time,
                                "end": previous_time,
                                "activity": current_label,
                            }
                        )

                        current_label = label
                        start_time = timestamp

                    previous_time = timestamp

                # Add final segment
                if current_label is not None:

                    timeline.append(
                        {
                            "start": start_time,
                            "end": previous_time,
                            "activity": current_label,
                        }
                    )

                for segment in timeline:

                    st.write(
                        f"**{segment['start']:.2f}s → "
                        f"{segment['end']:.2f}s** | "
                        f"{segment['activity']}"
                    )

            except Exception as e:

                st.error(
                    f"Error during analysis: {e}"
                )