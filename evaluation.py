"""
MediaPipe Pose evaluation for jumping-jack counting.

Replicates the metrics and artifact exports used in optimization/evaluate_model/evaluate_model.py,
but runs with the lightweight pose_landmarker_lite TFLite/task models.

Artifacts per run:
- videos/: annotated mp4s per model
- tables/: ground truth + per-model predictions + CSV summaries
- figures/: bar charts and confusion matrices
"""

from __future__ import annotations

import csv
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd

# -----------------------------
# Paths & configuration
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
GROUND_TRUTH_TXT = PROJECT_ROOT / "evaluation_requirement" / "ground_truth.txt"
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / f"mp_evaluation_{RUN_ID}"
EXPORT_FOLDER = OUTPUT_ROOT / "videos"
FIGURES_FOLDER = OUTPUT_ROOT / "figures"
TABLES_FOLDER = OUTPUT_ROOT / "tables"

# Model registry (task + raw tflite)
MODEL_PATHS: Dict[str, Path] = {
    "Pose Landmarker Lite (task)": PROJECT_ROOT / "activity-mediapipe" / "pose_landmarker_lite.task",
}

# Pose topology for quick drawing (same as run_pose_tflite.py)
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
    """Sanitize strings for filesystem-safe filename components (Windows friendly)."""
    safe = re.sub(r'[<>:"/\\|?*]', "_", text)
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_.")
    return safe or "model"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def draw_landmarks(frame: np.ndarray, landmarks) -> None:
    """Draw simple skeleton lines and keypoints."""
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
    """Detects jumping jack state transitions and counts reps."""

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
# Evaluator
# -----------------------------
class MediapipeEvaluator:
    def __init__(self, ground_truth_txt: Path):
        self.ground_truth_txt = Path(ground_truth_txt)
        self.ground_truth = self._load_ground_truth()
        self.results: Dict[str, dict] = {}

        EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)
        FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
        TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
        self._export_ground_truth_txt()

    def _load_ground_truth(self) -> Dict[str, float]:
        data = {}
        if not self.ground_truth_txt.exists():
            print(f"[ERROR] Ground truth file missing: {self.ground_truth_txt}")
            return data
        with open(self.ground_truth_txt, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[row["Video"]] = float(row["Counts"])
        return data

    def get_video_files(self) -> List[str]:
        return sorted(self.ground_truth.keys())

    def evaluate_model(self, model_name: str, model_path: Path):
        if not model_path.exists():
            print(f"[ERROR] Model not found: {model_path}")
            return

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

        predictions = {}
        total_frames = 0
        total_infer_time = 0.0

        for video_id in self.get_video_files():
            gt_count = self.ground_truth[video_id]
            video_path = DATA_FOLDER / f"{video_id}.mp4"
            if not video_path.exists():
                print(f"[WARN] Skipping {video_path} (not found)")
                continue

            with PoseLandmarker.create_from_options(options) as landmarker:
                print(f"\nProcessing {video_id}.mp4 with {model_name} (expected {gt_count})")
                stats = self._process_video(video_path, landmarker, model_name, video_id)

            predictions[video_id] = {
                "correct": gt_count,
                "predicted": stats["count"],
                "frames": stats["frames"],
                "infer_time": stats["infer_time"],
            }
            total_frames += stats["frames"]
            total_infer_time += stats["infer_time"]

        if not predictions:
            print("No predictions recorded for this model.")
            return

        metrics = self._calculate_metrics(
            predictions=predictions,
            total_frames=total_frames,
            total_infer_time=total_infer_time,
            model_path=model_path,
        )
        self.results[model_name] = {"predictions": predictions, "metrics": metrics}
        self._save_predictions_txt(model_name, predictions)
        return metrics

    def _process_video(self, video_path: Path, landmarker, model_name: str, video_id: str):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"    [ERROR] Could not open {video_path}")
            return {"count": 0, "frames": 0, "infer_time": 0.0}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        model_folder = EXPORT_FOLDER / _safe_filename_component(model_name)
        model_folder.mkdir(parents=True, exist_ok=True)
        outfile = model_folder / f"{video_id}_{_safe_filename_component(model_name)}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(outfile), fourcc, fps, (w, h))
        if not writer.isOpened():
            print(f"    [WARN] Could not open writer at {outfile}; continuing without export.")
            writer = None

        counter = JumpingJackCounter()
        frame_id = 0
        total_infer_time = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(frame_id * (1000 / fps))

            t0 = time.perf_counter()
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            total_infer_time += (time.perf_counter() - t0)

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

            cv2.putText(overlay, f"Count: {count}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(overlay, f"State: {state}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(overlay, f"Model: {model_name}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Frame: {frame_id}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if writer:
                writer.write(overlay)
            frame_id += 1

        cap.release()
        if writer:
            writer.release()
            print(f"    [OK] Saved annotated video to {outfile}")
        else:
            print(f"    [WARN] Skipped video export for {video_path.name}")

        return {"count": counter.count, "frames": frame_id, "infer_time": total_infer_time}

    def _calculate_metrics(self, predictions, total_frames, total_infer_time, model_path: Path):
        """
        Simplified metric set aligned with evaluation-camera.py:
        MAE / RMSE plus throughput and model size. Accuracy/precision/recall are omitted.
        """
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

        errors = []
        for _, data in predictions.items():
            correct = data["correct"]
            pred = data["predicted"]
            errors.append(abs(correct - pred))

        metrics["mae"] = float(np.mean(errors)) if errors else 0.0
        metrics["rmse"] = float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else 0.0

        if total_frames > 0 and total_infer_time > 0:
            metrics["fps"] = total_frames / total_infer_time
            metrics["latency_ms"] = (total_infer_time / total_frames) * 1000

        return metrics

    def _export_ground_truth_txt(self):
        if not self.ground_truth:
            return
        TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
        dest = TABLES_FOLDER / "ground_truth.txt"
        with open(dest, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Video", "Counts"])
            for vid, cnt in sorted(self.ground_truth.items()):
                writer.writerow([vid, int(cnt)])
        print(f"[OK] Ground truth snapshot saved to {dest}")

    def _save_predictions_txt(self, model_name: str, predictions):
        path = TABLES_FOLDER / f"{_safe_filename_component(model_name)}_predictions.txt"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Video", "Counts"])
            for vid, data in sorted(predictions.items()):
                writer.writerow([vid, int(data["predicted"])])
        print(f"[OK] Predictions saved to {path}")

    def print_results(self):
        if not self.results:
            print("No results to display")
            return

        header = (
            f"{'Model':<32}"
            f"{'FPS':>8}"
            f"{'Size(MB)':>10}"
            f"{'Latency(ms)':>13}"
            f"{'MAE':>7}"
            f"{'RMSE':>7}"
        )
        print("\n" + "=" * len(header))
        print(f"{'MEDIAPIPE MODEL COMPARISON':^{len(header)}}")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for model_name, data in self.results.items():
            m = data["metrics"]
            print(
                f"{model_name:<32}"
                f"{m['fps']:>8.1f}"
                f"{m['size_mb']:>10.2f}"
                f"{m['latency_ms']:>13.2f}"
                f"{m['mae']:>7.2f}"
                f"{m['rmse']:>7.2f}"
            )
        print("=" * len(header))

    def save_results_csv(self, output_file="evaluation_results.csv"):
        output_path = TABLES_FOLDER / output_file
        rows = []
        for model_name, data in self.results.items():
            metrics = data["metrics"]
            for vid, pred_data in data["predictions"].items():
                rows.append({
                    "model": model_name,
                    "video": vid,
                    "correct_count": pred_data["correct"],
                    "predicted_count": pred_data["predicted"],
                    "error": abs(pred_data["correct"] - pred_data["predicted"]),
                    "frames": pred_data.get("frames", 0),
                    "infer_time_s": pred_data.get("infer_time", 0.0),
                })
            rows.append({
                "model": model_name,
                "video": "SUMMARY",
                "correct_count": "",
                "predicted_count": "",
                "error": "",
                "frames": "",
                "infer_time_s": "",
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "fps": metrics["fps"],
                "latency_ms": metrics["latency_ms"],
                "size_mb": metrics["size_mb"],
                "params": metrics["params"],
            })
        if rows:
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(f"[OK] Results saved to {output_path}")

    def save_metrics_summary_csv(self, output_file="metrics_summary.csv"):
        output_path = TABLES_FOLDER / output_file
        rows = []
        for model_name, data in self.results.items():
            m = data["metrics"]
            rows.append({
                "model": model_name,
                "samples": m["samples"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "fps": m["fps"],
                "latency_ms": m["latency_ms"],
                "size_mb": m["size_mb"],
                "params": m["params"],
            })
        if rows:
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(f"[OK] Metrics summary saved to {output_path}")

    def plot_results(self):
        if not self.results:
            print("No results to plot")
            return

        model_names = list(self.results.keys())
        metrics_data = {name: data["metrics"] for name, data in self.results.items()}
        x = np.arange(len(model_names))
        width = 0.35

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Jumping Jack Counter - Simplified Metrics", fontsize=16, fontweight="bold")

        maes = [metrics_data[name]["mae"] for name in model_names]
        rmses = [metrics_data[name]["rmse"] for name in model_names]
        axes[0].bar(x - width / 2, maes, width, label="MAE", color="orange")
        axes[0].bar(x + width / 2, rmses, width, label="RMSE", color="red")
        axes[0].set_ylabel("Error (count)")
        axes[0].set_title("MAE vs RMSE")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names)
        axes[0].legend()

        fps_vals = [metrics_data[name]["fps"] for name in model_names]
        latency = [metrics_data[name]["latency_ms"] for name in model_names]
        axes[1].bar(x - width / 2, fps_vals, width, label="FPS", color="steelblue")
        axes[1].bar(x + width / 2, latency, width, label="Latency (ms)", color="mediumseagreen")
        axes[1].set_title("Throughput & Latency")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names)
        axes[1].legend()

        sizes = [metrics_data[name]["size_mb"] for name in model_names]
        axes[2].bar(model_names, sizes, color="plum")
        axes[2].set_ylabel("Size (MB)")
        axes[2].set_title("Model Size")

        plt.tight_layout()
        summary_path = FIGURES_FOLDER / "evaluation_results.png"
        plt.savefig(summary_path, dpi=300)
        print(f"[OK] Summary plot saved to {summary_path}")
        plt.close("all")


def main():
    print("=" * 60)
    print("MediaPipe Jumping Jack Evaluation")
    print("=" * 60)

    evaluator = MediapipeEvaluator(GROUND_TRUTH_TXT)

    for model_name, model_path in MODEL_PATHS.items():
        evaluator.evaluate_model(model_name, model_path)

    evaluator.print_results()
    evaluator.save_results_csv()
    evaluator.save_metrics_summary_csv()
    evaluator.plot_results()

    print("\nArtifacts saved to:")
    print(f"  Videos   : {EXPORT_FOLDER}")
    print(f"  Figures  : {FIGURES_FOLDER}")
    print(f"  Tables   : {TABLES_FOLDER}")
    print(f"  Run ID   : {RUN_ID}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
