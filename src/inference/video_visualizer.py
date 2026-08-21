from pathlib import Path

import cv2


CLASS_COLORS = {
    "Safe driving": (0, 255, 0),
}


def get_status(label):
    """
    Determine whether the predicted activity
    is safe or distracted.
    """

    if label == "Safe driving":
        return "SAFE"

    return "DISTRACTED"


def annotate_frame(
    frame,
    label,
    confidence,
):
    """
    Add prediction information to a video frame.
    """

    status = get_status(label)

    if status == "SAFE":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    # Background rectangle
    cv2.rectangle(
        frame,
        (20, 20),
        (600, 125),
        (0, 0, 0),
        -1,
    )

    # Title
    cv2.putText(
        frame,
        "DRIVER DISTRACTION DETECTOR",
        (35, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # Activity
    cv2.putText(
        frame,
        f"Activity: {label}",
        (35, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    # Confidence
    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.2f}%",
        (35, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    # Status
    cv2.putText(
        frame,
        status,
        (frame.shape[1] - 180, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    return frame