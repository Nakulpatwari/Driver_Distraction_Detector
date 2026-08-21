from pathlib import Path

from src.inference.video_processor import (
    get_video_info,
    sample_frames,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    PROJECT_ROOT /
    "data" /
    "videos" /
    "test_video.mp4"
)


def main():

    print("=" * 60)
    print("VIDEO PROCESSING TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    info = get_video_info(
        VIDEO_PATH
    )

    print("\nVIDEO INFORMATION")
    print("-" * 40)

    print(
        f"FPS          : {info['fps']:.2f}"
    )

    print(
        f"Frame count  : {info['frame_count']}"
    )

    print(
        f"Resolution   : "
        f"{info['width']} x {info['height']}"
    )

    print(
        f"Duration     : "
        f"{info['duration']:.2f} seconds"
    )

    # --------------------------------------------------------
    # Sample frames
    # --------------------------------------------------------

    print("\nSAMPLING FRAMES")
    print("-" * 40)

    frames, indices = sample_frames(
        VIDEO_PATH,
        num_frames=15,
    )

    print(
        f"Frames requested : 15"
    )

    print(
        f"Frames extracted : {len(frames)}"
    )

    print(
        f"Frame indices    : {indices.tolist()}"
    )

    # --------------------------------------------------------
    # Frame shape
    # --------------------------------------------------------

    print("\nFRAME INFORMATION")
    print("-" * 40)

    print(
        f"First frame shape : "
        f"{frames[0].shape}"
    )

    print(
        f"First frame dtype : "
        f"{frames[0].dtype}"
    )

    print(
        f"Pixel range      : "
        f"{frames[0].min()} - "
        f"{frames[0].max()}"
    )

    print("\n✓ VIDEO PROCESSING WORKS")


if __name__ == "__main__":
    main()