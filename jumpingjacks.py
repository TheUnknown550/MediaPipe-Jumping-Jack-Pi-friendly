import cv2
import mediapipe as mp
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. SETUP - Configure the Pose Landmarker
model_path = 'pose_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Callback function to handle the results (Asynchronous mode for max speed)
def print_result(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global current_landmarks
    if result.pose_landmarks:
        current_landmarks = result.pose_landmarks[0]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO, # Optimized for webcam streams
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5
)

# 2. HELPER - Angle Calculation
def get_angle(a, b, c):
    a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# 3. MAIN LOOP
cap = cv2.VideoCapture(0)
count = 0
stage = "down"
prev_time = time.time()

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Convert the frame to MediaPipe's Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(time.time() * 1000)
        
        # Detect landmarks
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            
            # Keypoints for Jumping Jacks
            # Shoulder (11,12), Hip (23,24), Elbow (13,14)
            l_sh, r_sh = landmarks[11], landmarks[12]
            l_hi, r_hi = landmarks[23], landmarks[24]
            l_el, r_el = landmarks[13], landmarks[14]

            # Calculate Armpit Angles
            l_angle = get_angle(l_hi, l_sh, l_el)
            r_angle = get_angle(r_hi, r_sh, r_el)

            # Logic State Machine
            if l_angle > 140 and r_angle > 140:
                stage = "up"
            if l_angle < 50 and r_angle < 50 and stage == "up":
                stage = "down"
                count += 1
                print(f"Jumping Jack Count: {count}")

        # HUD & FPS
        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        cv2.putText(frame, f"REPS: {count}  FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('MediaPipe Tasks Pi Counter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()