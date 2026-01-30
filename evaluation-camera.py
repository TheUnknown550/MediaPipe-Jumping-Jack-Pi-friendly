"""
Live webcam evaluation for jumping-jack counting with MediaPipe Pose.

- Runs the light pose_landmarker_lite model on your webcam feed in real time.
- Overlays the counter and simple state on the video window.
- Press "q" or ESC to stop; you'll then be asked for the ground-truth rep count.
- A short metrics summary and CSV snapshot are saved per session.
"""

from __future__ import annotations

import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
# Paths & configuration
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / f"mp_webcam_{RUN_ID}"
EXPORT_FOLDER = OUTPUT_ROOT / "videos"
TABLES_FOLDER = OUTPUT_ROOT / "tables"

MODEL_PATHS: Dict[str, Path] = {
    "Pose Landmarker Lite (task)": "pose_landmarker_lite.task",
}

# Simple pose topology for drawing
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (27, 31), (28, 32),
]


# -----------------------------
# Helpers
# -----------------------------
def _safe_filename_component(text: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "_", text)
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_.")
    return safe or "model"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def draw_landmarks(frame: np.ndarray, landmarks) -> None:
    h, w, _ = frame.shape
    for i, j in POSE_CONNECTIONS:
        if i >= len(landmarks) or j >= len(landmarks):
            continue
        li, lj = landmarks[i], landmarks[j]
        pt1 = (int(li.x * w), int(li.y * h))
        pt2 = (int(lj.x * w), int(lj.y * h))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    for l in landmarks:
        pt = (int(l.x * w), int(l.y * h))
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


# -----------------------------
# Counter
# -----------------------------
class JumpingJackCounter:
    def __init__(self, window: int = 7):
        self.state = "UNKNOWN"
        self.count = 0
        self.open_hist = deque(maxlen=window)
        self.closed_hist = deque(maxlen=window)

    def update(self, open_score: float, closed_score: float):
        self.open_hist.append(open_score)
        self.closed_hist.append(closed_score)

        o = np.mean(self.open_hist)
        c = np.mean(self.closed_hist)

        prev_state = self.state
        if o > 0.6 and o > c + 0.1:
            self.state = "OPEN"
        elif c > 0.6 and c > o + 0.1:
            self.state = "CLOSED"

        if prev_state == "CLOSED" and self.state == "OPEN":
            self.count += 1

        return o, c, self.state, self.count


# -----------------------------
# Core evaluation helpers
# -----------------------------
def build_landmarker(model_path: Path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    return PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
    )


def _calculate_metrics(predictions, total_frames, total_infer_time, model_path: Path):
    metrics = {
        "mae": 0.0,
        "rmse": 0.0,
        "samples": len(predictions),
        "fps": 0.0,
        "latency_ms": 0.0,
        "size_mb": model_path.stat().st_size / 1e6 if model_path.exists() else 0.0,
        "params": None,
        "map": np.nan,
        "map50_95": np.nan,
        "confusion": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }

    correct_predictions = 0
    errors = []
    signed_errors = []

    for _, data in predictions.items():
        correct = data["correct"]
        pred = data["predicted"]

        errors.append(abs(correct - pred))
        signed_errors.append(pred - correct)

        if correct == pred:
            correct_predictions += 1


    metrics["mae"] = float(np.mean(errors)) if errors else 0.0
    metrics["rmse"] = float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else 0.0
    #metrics["bias"] = float(np.mean(signed_errors)) if signed_errors else 0.0

    
    if total_frames > 0 and total_infer_time > 0:
        metrics["fps"] = total_frames / total_infer_time
        metrics["latency_ms"] = (total_infer_time / total_frames) * 1000

    #metrics["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": 0}
    return metrics


def run_webcam_session(model_name: str, model_path: Path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam (device 0).")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps_setting = cap.get(cv2.CAP_PROP_FPS) or 30.0

    EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    out_file = EXPORT_FOLDER / f"webcam_{_safe_filename_component(model_name)}_{RUN_ID}.mp4"
    writer = None

    counter = JumpingJackCounter()
    total_infer_time = 0.0
    frames = 0
    start_time = time.time()

    with build_landmarker(model_path) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame from camera; stopping.")
                break

            h, w = frame.shape[:2]
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_file), fourcc, fps_setting, (w, h))
                if not writer.isOpened():
                    print(f"[WARN] Could not open writer at {out_file}; continuing without export.")
                    writer = None

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(time.time() * 1000)

            t0 = time.perf_counter()
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            total_infer_time += time.perf_counter() - t0

            overlay = frame.copy()
            open_s = closed_s = 0.0

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                draw_landmarks(overlay, lms)
                try:
                    coords = np.array([(lm.x * w, lm.y * h) for lm in lms])
                    l_sho, r_sho = coords[11], coords[12]
                    l_ank, r_ank = coords[27], coords[28]
                    l_wri, r_wri = coords[15], coords[16]

                    sho_w = dist(l_sho, r_sho)
                    ank_w = dist(l_ank, r_ank)
                    legs_ratio = ank_w / (sho_w + 1e-6)

                    wrists_y = (l_wri[1] + r_wri[1]) / 2
                    shoulders_y = (l_sho[1] + r_sho[1]) / 2
                    arms_up = clamp01((shoulders_y - wrists_y) / 100)

                    open_s = 0.5 * clamp01(legs_ratio - 0.8) + 0.5 * arms_up
                    closed_s = 1.0 - open_s
                except Exception:
                    pass

            o, c, state, count = counter.update(open_s, closed_s)
            fps_live = frames / max(time.time() - start_time, 1e-6)

            cv2.putText(overlay, f"Count: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(overlay, f"State: {state}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(overlay, f"Model: {model_name}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"FPS: {fps_live:0.1f}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, 'Press "q" or ESC to stop', (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if writer:
                writer.write(overlay)

            cv2.imshow("Jumping Jack Counter (press q to stop)", overlay)
            frames += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    if writer:
        writer.release()
        print(f"[OK] Saved annotated session to {out_file}")

    return {
        "count": counter.count,
        "frames": frames,
        "infer_time": total_infer_time,
        "video_path": str(out_file) if writer else None,
    }


def prompt_ground_truth() -> int:
    while True:
        user_input = input("How many jumping jacks did you actually complete? ")
        try:
            return int(user_input)
        except ValueError:
            print("Please enter a whole number.")


def print_session_results(model_name: str, metrics: dict, predicted: int, ground_truth: int, video_path: str | None):
    print("\n" + "=" * 60)
    print(f"{'Webcam Evaluation Summary':^60}")
    print("=" * 60)
    print(f"Model           : {model_name}")
    print(f"Predicted count : {predicted}")
    print(f"Ground truth    : {ground_truth}")
    print(f"Absolute error  : {abs(predicted - ground_truth)}")
    #print(f"Bias (pred-gt)  : {metrics['bias']:.2f}")
    print(f"MAE / RMSE      : {metrics['mae']:.2f} / {metrics['rmse']:.2f}")
    print(f"Precision / Recall / F1 (%): {metrics['precision']:.1f} / {metrics['recall']:.1f} / {metrics['f1']:.1f}")
    print(f"Throughput FPS  : {metrics['fps']:.2f}")
    print(f"Latency (ms)    : {metrics['latency_ms']:.2f}")
    print(f"Model size (MB) : {metrics['size_mb']:.2f}")
    if video_path:
        print(f"Annotated video : {video_path}")
    print("=" * 60)


def save_session_csv(model_name: str, metrics: dict, predicted: int, ground_truth: int):
    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_FOLDER / f"session_{_safe_filename_component(model_name)}_{RUN_ID}.csv"
    rows = [
        ["model", "ground_truth", "predicted", "error", "mae", "rmse", "bias", "fps", "latency_ms", "size_mb"],
        [
            model_name,
            ground_truth,
            predicted,
            abs(predicted - ground_truth),
            f"{metrics['mae']:.3f}",
            f"{metrics['rmse']:.3f}",
            f"{metrics['bias']:.3f}",
            f"{metrics['fps']:.3f}",
            f"{metrics['latency_ms']:.3f}",
            f"{metrics['size_mb']:.3f}",
        ],
    ]
    with open(csv_path, "w", newline="") as f:
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")
    print(f"[OK] Session metrics saved to {csv_path}")


def pick_first_available_model() -> Tuple[str, Path]:
    for name, path in MODEL_PATHS.items():
        if path.exists():
            return name, path
    raise FileNotFoundError("No pose model files found. Expected pose_landmarker_lite.task or pose_landmarker_lite.tflite in the project root.")


def main():
    print("=" * 60)
    print("MediaPipe Jumping Jack - Webcam Evaluation")
    print("=" * 60)

    try:
        model_name, model_path = pick_first_available_model()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    print(f"Using model: {model_name} -> {model_path}")
    print('Instructions: keep doing jumping jacks; press "q" or ESC to stop.')

    session_stats = run_webcam_session(model_name, model_path)
    if not session_stats or session_stats["frames"] == 0:
        print("[ERROR] No frames processed; exiting.")
        return

    ground_truth = prompt_ground_truth()
    predictions = {
        "webcam": {
            "correct": ground_truth,
            "predicted": session_stats["count"],
            "frames": session_stats["frames"],
            "infer_time": session_stats["infer_time"],
        }
    }

    metrics = _calculate_metrics(
        predictions=predictions,
        total_frames=session_stats["frames"],
        total_infer_time=session_stats["infer_time"],
        model_path=model_path,
    )

    print_session_results(model_name, metrics, session_stats["count"], ground_truth, session_stats.get("video_path"))
    save_session_csv(model_name, metrics, session_stats["count"], ground_truth)

    print("\nArtifacts saved to:")
    print(f"  Videos : {EXPORT_FOLDER}")
    print(f"  Tables : {TABLES_FOLDER}")
    print(f"  Run ID : {RUN_ID}")


if __name__ == "__main__":
    main()
