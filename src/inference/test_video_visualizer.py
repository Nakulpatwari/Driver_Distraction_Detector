from pathlib import Path

import cv2
import numpy as np

from src.inference.video_processor import get_video_info
from src.inference.predictor import (
    load_model,
    preprocess_frames,
    decode_prediction,
)
from src.inference.video_visualizer import annotate_frame


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / "test_video.mp4"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videos"

OUTPUT_VIDEO = (
    OUTPUT_DIR
    / "annotated_test_video.mp4"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("VIDEO VISUALIZER TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Check input video
    # --------------------------------------------------------

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    # --------------------------------------------------------
    # 2. Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 3. Get video information
    # --------------------------------------------------------

    video_info = get_video_info(
        VIDEO_PATH
    )

    print("\nVideo information:")
    print(
        f"Duration: {video_info['duration']:.2f} seconds"
    )
    print(
        f"FPS: {video_info['fps']:.2f}"
    )
    print(
        f"Frames: {video_info['frame_count']}"
    )
    print(
        f"Resolution: "
        f"{video_info['width']} x "
        f"{video_info['height']}"
    )

    # --------------------------------------------------------
    # 4. Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise ValueError(
            "Could not open video."
        )

    # --------------------------------------------------------
    # 5. Load trained model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # 6. Prepare video writer
    # --------------------------------------------------------

    fps = video_info["fps"]

    width = video_info["width"]
    height = video_info["height"]

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise ValueError(
            "Could not create output video."
        )

    # --------------------------------------------------------
    # 7. Process video frame-by-frame
    # --------------------------------------------------------

    frame_number = 0

    print("\nProcessing video...")

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # OpenCV gives BGR.
        # Convert to RGB for TensorFlow.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # Prepare frame for MobileNetV2.
        batch = preprocess_frames(
            [rgb_frame]
        )

        # Model prediction.
        probabilities = model.predict(
            batch,
            verbose=0,
        )

        predicted_index = int(
            np.argmax(
                probabilities[0]
            )
        )

        confidence = float(
            probabilities[0][predicted_index]
        )

        # Decode prediction.
        class_name, label = decode_prediction(
            predicted_index
        )

        # ----------------------------------------------------
        # 8. Annotate frame
        # ----------------------------------------------------

        annotated_frame = annotate_frame(
            frame.copy(),
            label,
            confidence,
        )

        # Add timestamp.
        timestamp = (
            frame_number / fps
            if fps > 0
            else 0
        )

        cv2.putText(
            annotated_frame,
            f"Time: {timestamp:.2f}s",
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # ----------------------------------------------------
        # 9. Write frame
        # ----------------------------------------------------

        writer.write(
            annotated_frame
        )

        # Show progress every 30 frames.
        if frame_number % 30 == 0:

            print(
                f"Processed "
                f"{frame_number}/"
                f"{video_info['frame_count']} frames"
            )

    # --------------------------------------------------------
    # 10. Cleanup
    # --------------------------------------------------------

    cap.release()
    writer.release()

    print("\n" + "=" * 60)
    print("VIDEO VISUALIZATION COMPLETE")
    print("=" * 60)

    print(
        f"\nOutput video:\n{OUTPUT_VIDEO}"
    )

    print(
        f"\nTotal frames processed: "
        f"{frame_number}"
    )


if __name__ == "__main__":
    main()