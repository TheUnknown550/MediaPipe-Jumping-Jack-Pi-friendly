"""
Lightweight TFLite runner for pose_landmarker_lite.tflite using MediaPipe Tasks.
No reliance on the deprecated mp.solutions.* API.

Run:
    python run_pose_tflite.py
Press 'q' to exit.
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp

# Pose connection topology (copied from MediaPipe Pose v1)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # left eye/ear
    (0, 4), (4, 5), (5, 6), (6, 8),  # right eye/ear
    (9, 10),  # mouth
    (11, 12),  # shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # left arm/hand
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # right arm/hand
    (11, 23), (12, 24), (23, 24),  # torso/hips
    (23, 25), (25, 27), (27, 29), (29, 31),  # left leg/foot
    (24, 26), (26, 28), (28, 30), (30, 32),  # right leg/foot
    (27, 31), (28, 32),  # feet cross links
]


def draw_landmarks(frame, landmarks):
    """Draw simple skeleton lines and keypoints."""
    h, w, _ = frame.shape

    # Draw connections
    for i, j in POSE_CONNECTIONS:
        if i >= len(landmarks) or j >= len(landmarks):
            continue
        li, lj = landmarks[i], landmarks[j]
        if li.visibility < 0.5 or lj.visibility < 0.5:
            continue
        pt1 = (int(li.x * w), int(li.y * h))
        pt2 = (int(lj.x * w), int(lj.y * h))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Draw keypoints
    for l in landmarks:
        if l.visibility < 0.5:
            continue
        pt = (int(l.x * w), int(l.y * h))
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


def main():
    model_path = Path("pose_landmarker_lite.tflite")
    if not model_path.exists():
        raise FileNotFoundError(
            "pose_landmarker_lite.tflite not found. "
            "Place it next to this script or update model_path."
        )

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                draw_landmarks(frame, result.pose_landmarks[0])

            now = time.time()
            fps = 1 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Pose TFLite Runner", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
