from pathlib import Path

from src.inference.video_analyzer import analyze_video


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / "test_video.mp4"
)


def main():

    print("=" * 60)
    print("VIDEO ANALYZER TEST")
    print("=" * 60)

    print("\nAnalyzing video:")
    print(VIDEO_PATH)

    results = analyze_video(
        VIDEO_PATH,
        num_frames=15,
    )

    video_info = results["video_info"]

    print("\n" + "=" * 60)
    print("VIDEO INFORMATION")
    print("=" * 60)

    print(
        f"Duration      : {video_info['duration']:.2f} seconds"
    )

    print(
        f"FPS           : {video_info['fps']:.2f}"
    )

    print(
        f"Total frames  : {video_info['frame_count']}"
    )

    print(
        f"Resolution    : "
        f"{video_info['width']} x {video_info['height']}"
    )

    print(
        f"Frames sampled: "
        f"{results['num_frames_analyzed']}"
    )

    print("\n" + "=" * 60)
    print("FRAME-LEVEL RESULTS")
    print("=" * 60)

    for result in results["frame_results"]:

        print(
            f"Frame {result['sample_number']:02d} | "
            f"{result['timestamp']:5.2f}s | "
            f"{result['class_name']} | "
            f"{result['label']} | "
            f"{result['confidence']:.2%}"
        )
        print("\n" + "=" * 60)
    print("VIDEO-LEVEL SUMMARY")
    print("=" * 60)

    print("\nActivity counts:")

    for activity, count in results["activity_counts"].items():
        print(f"{activity:<30} {count} frames")

    print(
        f"\nDominant activity : "
        f"{results['dominant_activity']}"
    )

    print(
        f"Distraction detected : "
        f"{'YES' if results['distraction_detected'] else 'NO'}"
    )

    print(
        f"Distraction percentage : "
        f"{results['distraction_percentage']:.2f}%"
    )
    print("\n" + "=" * 60)
    print("ACTIVITY TIMELINE")
    print("=" * 60)

    for segment in results["timeline"]:
        print(
            f"{segment['start_time']:.2f}s"
            f" -> "
            f"{segment['end_time']:.2f}s"
            f" | "
            f"{segment['activity']}"
        )

if __name__ == "__main__":
    main()