from pathlib import Path

from src.inference.video_processor import (
    get_video_info,
    sample_frames,
)

from src.inference.predictor import (
    load_model,
    predict_frames,
    decode_prediction,
)


def analyze_video(video_path, num_frames=15):
    """
    Run frame-level driver distraction inference
    on a video.
    """

    video_path = Path(video_path)

    # 1. Get video information
    video_info = get_video_info(video_path)

    # 2. Sample frames
    frames, frame_indices = sample_frames(
        video_path,
        num_frames=num_frames,
    )

    # 3. Load trained model
    model = load_model()

    # 4. Generate predictions
    probabilities, predicted_indices = predict_frames(
        model,
        frames,
    )

    # 5. Create frame-level results
    frame_results = []

    for i, (frame_index, predicted_index) in enumerate(
        zip(frame_indices, predicted_indices)
    ):

        class_name, label = decode_prediction(
            int(predicted_index)
        )

        confidence = float(
            probabilities[i][predicted_index]
        )

        # Convert frame number to timestamp
        if video_info["fps"] > 0:
            timestamp = (
                float(frame_index)
                / video_info["fps"]
            )
        else:
            timestamp = 0.0

        frame_result = {
            "sample_number": i + 1,
            "frame_index": int(frame_index),
            "timestamp": timestamp,
            "class_name": class_name,
            "label": label,
            "confidence": confidence,
        }

        frame_results.append(frame_result)

        # 6. Create video-level summary
    activity_counts = {}

    for result in frame_results:
        label = result["label"]

        activity_counts[label] = (
            activity_counts.get(label, 0) + 1
        )
        # 7. Build activity timeline
    timeline = []

    if frame_results:

        current_activity = frame_results[0]["label"]
        start_time = frame_results[0]["timestamp"]

        for result in frame_results[1:]:

            activity = result["label"]

            if activity != current_activity:

                timeline.append({
                    "activity": current_activity,
                    "start_time": start_time,
                    "end_time": result["timestamp"],
                })

                current_activity = activity
                start_time = result["timestamp"]

        # Add the final activity segment
        timeline.append({
            "activity": current_activity,
            "start_time": start_time,
            "end_time": frame_results[-1]["timestamp"],
        })

    # Find the most frequently predicted activity
    dominant_activity = max(
        activity_counts,
        key=activity_counts.get
    )

    # Safe driving is c0.
    # Any other class is considered a distraction.
    distraction_frames = sum(
        count
        for label, count in activity_counts.items()
        if label != "Safe driving"
    )

    distraction_detected = distraction_frames > 0

    # Estimate the proportion of the video classified
    # as distracted based on sampled frames.
    if len(frame_results) > 0:
        distraction_percentage = (
            distraction_frames
            / len(frame_results)
            * 100
        )
    else:
        distraction_percentage = 0.0

    return {
        "video_info": video_info,
        "num_frames_analyzed": len(frame_results),
        "frame_results": frame_results,

        # Video-level summary
        "activity_counts": activity_counts,
        "dominant_activity": dominant_activity,
        "distraction_detected": distraction_detected,
        "distraction_percentage": distraction_percentage,
        "timeline": timeline,
    }