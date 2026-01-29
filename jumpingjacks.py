import cv2
import mediapipe as mp
import time
import numpy as np

from mediapipe.framework.formats import landmark_pb2

# -----------------------------
# 1) SETUP - Pose Landmarker
# -----------------------------
model_path = "pose_landmarker_lite.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
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

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # OpenCV gives BGR; MediaPipe expects RGB for SRGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        # Detect landmarks
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

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

            # "Armpit" angles: hip -> shoulder -> elbow
            l_angle = get_angle(l_hi, l_sh, l_el)
            r_angle = get_angle(r_hi, r_sh, r_el)

            if l_angle > 140 and r_angle > 140:
                stage = "up"

            if l_angle < 50 and r_angle < 50 and stage == "up":
                stage = "down"
                count += 1
                print(f"Jumping Jack Count: {count}")

        # HUD & FPS
        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.putText(
            frame,
            f"REPS: {count}  FPS: {int(fps)}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("MediaPipe Tasks Pi Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
