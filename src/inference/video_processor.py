from pathlib import Path

import cv2
import numpy as np


def get_video_info(video_path):
    """
    Extract basic information from a video.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def sample_frames(
    video_path,
    num_frames=15,
):
    """
    Uniformly sample frames from a video.

    Returns:
        frames: list of RGB numpy arrays
        frame_indices: corresponding frame numbers
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames == 0:
        cap.release()

        raise ValueError(
            "Video contains no frames."
        )

    # Generate evenly spaced frame indices.
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int,
    )

    frames = []

    for index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index),
        )

        success, frame = cap.read()

        if not success:
            continue

        # OpenCV uses BGR.
        # TensorFlow/Keras models expect RGB.
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(
            "Could not extract frames from video."
        )

    return frames, frame_indices[:len(frames)]