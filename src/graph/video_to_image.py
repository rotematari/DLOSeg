import cv2
import os

def extract_frames(video_path: str, output_dir: str, img_ext: str = 'png'):
    """
    Extracts all frames from a video file and saves them as images.
    
    Args:
        video_path:   Path to the input video (e.g. 'input.webm')
        output_dir:   Directory where frames will be written.
        img_ext:      Image extension/format (e.g. 'png', 'jpg').
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Compose filename: frame_000000.png, frame_000001.png, ...
        fname = os.path.join(output_dir, f"frame_{frame_idx:06d}.{img_ext}")
        cv2.imwrite(fname, frame)
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames into '{output_dir}'")

if __name__ == "__main__":
    # Example usage:
    extract_frames('/home/admina/Videos/Screencasts/spline_video.webm', 'frames_output')
