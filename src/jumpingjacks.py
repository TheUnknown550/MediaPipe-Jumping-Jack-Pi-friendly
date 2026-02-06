from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

# -----------------------------
# 1) SETUP - Pose Landmarker
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run from repo root or adjust the path.")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
)

# MediaPipe drawing helpers (for skeleton overlay)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# -----------------------------
# 2) HELPER - Angle Calculation
# -----------------------------
def get_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


# -----------------------------
# 3) MAIN LOOP
# -----------------------------
cap = cv2.VideoCapture(0)

# Optional: reduce resolution for speed on Pi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0
stage = "down"
prev_time = time.time()
last_latency_ms = None  # last measured end-to-end detection latency
leg_ratio_hist = []  # keep a short history to see if legs actually move

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # OpenCV gives BGR; MediaPipe expects RGB for SRGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        # Detect landmarks and measure latency
        detect_start = time.perf_counter()
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        last_latency_ms = (time.perf_counter() - detect_start) * 1000

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # ---- DRAW SKELETON OVERLAY ----
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    landmark_pb2.NormalizedLandmark(
                        x=l.x, y=l.y, z=l.z, visibility=getattr(l, "visibility", 0.0)
                    )
                    for l in landmarks
                ]
            )
            mp_drawing.draw_landmarks(
                frame,
                pose_landmarks_proto,
                mp_pose.POSE_CONNECTIONS
            )

            # ---- JUMPING JACK COUNT LOGIC ----
            # Indices: shoulders (11,12), elbows (13,14), hips (23,24)
            l_sh, r_sh = landmarks[11], landmarks[12]
            l_el, r_el = landmarks[13], landmarks[14]
            l_hi, r_hi = landmarks[23], landmarks[24]
            l_ank, r_ank = landmarks[27], landmarks[28]  # ankles for leg spread

            # "Armpit" angles: hip -> shoulder -> elbow
            l_angle = get_angle(l_hi, l_sh, l_el)
            r_angle = get_angle(r_hi, r_sh, r_el)

            # ---- LEG SPREAD RATIO ----
            # Use ankle distance relative to shoulder width to decide if legs are open.
            shoulder_w = np.linalg.norm(
                np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y])
            )
            ankle_w = np.linalg.norm(
                np.array([l_ank.x, l_ank.y]) - np.array([r_ank.x, r_ank.y])
            )
            leg_ratio = ankle_w / (shoulder_w + 1e-6)
            leg_open = leg_ratio > 0.8  # heuristic threshold: >80% of shoulder width

            # Track recent leg ratios to detect movement variability
            leg_ratio_hist.append(leg_ratio)
            if len(leg_ratio_hist) > 30:  # roughly 1s window at 30 fps
                leg_ratio_hist.pop(0)
            legs_moving = (max(leg_ratio_hist) - min(leg_ratio_hist)) > 0.15

            # Count only when arms are up AND legs are open; finish when both reset
            if l_angle > 140 and r_angle > 140 and leg_open:
                stage = "up"

            if l_angle < 50 and r_angle < 50 and not leg_open and stage == "up":
                stage = "down"
                count += 1
                print(f"Jumping Jack Count: {count}")

        # HUD & FPS
        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now

        latency_text = f"{int(last_latency_ms)}ms" if last_latency_ms is not None else "--"

        cv2.putText(
            frame,
            f"REPS: {count}  FPS: {int(fps)}  LAT: {latency_text}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Legs: {'OPEN' if result.pose_landmarks and leg_open else 'CLOSED'} ({leg_ratio:.2f}x)",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 200, 255) if leg_open else (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Legs moving: {'YES' if legs_moving else 'NO'}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 200, 0) if legs_moving else (0, 0, 255),
            2,
        )

        cv2.imshow("MediaPipe Task  s Pi Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
