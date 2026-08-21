from pathlib import Path

from src.inference.video_processor import (
    sample_frames,
)

from src.inference.predictor import (
    load_model,
    predict_frames,
    decode_prediction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / "test_video.mp4"
)


def main():

    print("=" * 60)
    print("VIDEO MODEL INFERENCE TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Extract frames
    # --------------------------------------------------------

    print("\nExtracting video frames...")

    frames, frame_indices = sample_frames(
        VIDEO_PATH,
        num_frames=15
    )

    print(
        f"✓ Extracted {len(frames)} frames."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("\nRunning model inference...")

    probabilities, predictions = predict_frames(
        model,
        frames
    )

    print("✓ Predictions generated.")

    # --------------------------------------------------------
    # Display predictions
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FRAME-LEVEL PREDICTIONS")
    print("=" * 60)

    for i, (frame_index, prediction) in enumerate(
        zip(frame_indices, predictions)
    ):

        class_name, label = decode_prediction(
            int(prediction)
        )

        confidence = probabilities[
            i,
            prediction
        ]

        print(
            f"Frame {i + 1:02d} "
            f"(video frame {frame_index:04d}) "
            f"→ {class_name} | "
            f"{label} | "
            f"{confidence:.2%}"
        )

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()