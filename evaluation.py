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
    "Pose Landmarker Lite (tflite)": PROJECT_ROOT / "activity-mediapipe" / "pose_landmarker_lite.tflite",
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
        metrics = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "tpr": 0,
            "fpr": 0,
            "mae": 0,
            "rmse": 0,
            "bias": 0,
            "samples": len(predictions),
            "fps": 0,
            "latency_ms": 0,
            "size_mb": model_path.stat().st_size / 1e6 if model_path.exists() else 0,
            "params": None,
            "map": np.nan,
            "map50_95": np.nan,
            "confusion": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        }

        correct_predictions = 0
        tp = fp = fn = 0
        errors = []
        signed_errors = []

        for _, data in predictions.items():
            correct = data["correct"]
            pred = data["predicted"]

            errors.append(abs(correct - pred))
            signed_errors.append(pred - correct)

            if correct == pred:
                correct_predictions += 1

            if pred > 0:
                tp += min(correct, pred)
                if pred > correct:
                    fp += pred - correct
            if pred < correct:
                fn += correct - pred

        metrics["accuracy"] = (correct_predictions / len(predictions)) * 100 if predictions else 0
        metrics["mae"] = float(np.mean(errors)) if errors else 0
        metrics["rmse"] = float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else 0
        metrics["bias"] = float(np.mean(signed_errors)) if signed_errors else 0

        if (tp + fp) > 0:
            metrics["precision"] = (tp / (tp + fp)) * 100
        if (tp + fn) > 0:
            metrics["recall"] = (tp / (tp + fn)) * 100
        if (metrics["precision"] + metrics["recall"]) > 0:
            p = metrics["precision"] / 100
            r = metrics["recall"] / 100
            metrics["f1"] = (2 * p * r / (p + r)) * 100

        metrics["tpr"] = metrics["recall"]
        if (fp + fn) > 0:
            metrics["fpr"] = (fp / (fp + fn)) * 100

        if total_frames > 0 and total_infer_time > 0:
            metrics["fps"] = total_frames / total_infer_time
            metrics["latency_ms"] = (total_infer_time / total_frames) * 1000

        metrics["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": 0}
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
            f"{'Params':>12}"
            f"{'mAP':>8}"
            f"{'mAP50-95':>10}"
            f"{'Prec%':>8}"
            f"{'Rec%':>8}"
            f"{'F1%':>8}"
            f"{'MAE':>7}"
            f"{'RMSE':>7}"
            f"{'Bias':>7}"
        )
        print("\n" + "=" * len(header))
        print(f"{'MEDIAPIPE MODEL COMPARISON':^{len(header)}}")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for model_name, data in self.results.items():
            m = data["metrics"]
            params_m = (m["params"] / 1e6) if m["params"] else 0
            print(
                f"{model_name:<32}"
                f"{m['fps']:>8.1f}"
                f"{m['size_mb']:>10.2f}"
                f"{m['latency_ms']:>13.2f}"
                f"{params_m:>12.2f}"
                f"{(m['map'] if not np.isnan(m['map']) else 0):>8.1f}"
                f"{(m['map50_95'] if not np.isnan(m['map50_95']) else 0):>10.1f}"
                f"{m['precision']:>8.1f}"
                f"{m['recall']:>8.1f}"
                f"{m['f1']:>8.1f}"
                f"{m['mae']:>7.2f}"
                f"{m['rmse']:>7.2f}"
                f"{m['bias']:>7.2f}"
            )
            conf = m["confusion"]
            print(f"    Confusion (TP/FP/FN/TN): {conf['tp']}/{conf['fp']}/{conf['fn']}/{conf['tn']}")
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
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "bias": metrics["bias"],
                "fps": metrics["fps"],
                "latency_ms": metrics["latency_ms"],
                "size_mb": metrics["size_mb"],
                "params": metrics["params"],
                "tp": metrics["confusion"]["tp"],
                "fp": metrics["confusion"]["fp"],
                "fn": metrics["confusion"]["fn"],
                "tn": metrics["confusion"]["tn"],
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
                "bias": m["bias"],
                "accuracy_pct": m["accuracy"],
                "precision_pct": m["precision"],
                "recall_pct": m["recall"],
                "f1_pct": m["f1"],
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
        width = 0.25

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Jumping Jack Counter - MediaPipe Models", fontsize=16, fontweight="bold")

        accuracies = [metrics_data[name]["accuracy"] for name in model_names]
        axes[0, 0].bar(model_names, accuracies, color="skyblue")
        axes[0, 0].set_ylabel("Accuracy (%)")
        axes[0, 0].set_title("Accuracy")
        axes[0, 0].set_ylim([0, 100])

        precisions = [metrics_data[name]["precision"] for name in model_names]
        recalls = [metrics_data[name]["recall"] for name in model_names]
        f1s = [metrics_data[name]["f1"] for name in model_names]
        axes[0, 1].bar(x - width, precisions, width, label="Precision", color="lightgreen")
        axes[0, 1].bar(x, recalls, width, label="Recall", color="lightcoral")
        axes[0, 1].bar(x + width, f1s, width, label="F1", color="gold")
        axes[0, 1].set_ylabel("Score (%)")
        axes[0, 1].set_title("Precision / Recall / F1")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names)
        axes[0, 1].legend()
        axes[0, 1].set_ylim([0, 100])

        maes = [metrics_data[name]["mae"] for name in model_names]
        rmses = [metrics_data[name]["rmse"] for name in model_names]
        axes[0, 2].bar(x - width / 2, maes, width, label="MAE", color="orange")
        axes[0, 2].bar(x + width / 2, rmses, width, label="RMSE", color="red")
        axes[0, 2].set_ylabel("Error (count)")
        axes[0, 2].set_title("MAE vs RMSE")
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(model_names)
        axes[0, 2].legend()

        fps_vals = [metrics_data[name]["fps"] for name in model_names]
        latency = [metrics_data[name]["latency_ms"] for name in model_names]
        axes[1, 0].bar(x - width / 2, fps_vals, width, label="FPS", color="steelblue")
        axes[1, 0].bar(x + width / 2, latency, width, label="Latency (ms)", color="mediumseagreen")
        axes[1, 0].set_title("Throughput & Latency")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names)
        axes[1, 0].legend()

        sizes = [metrics_data[name]["size_mb"] for name in model_names]
        axes[1, 1].bar(model_names, sizes, color="plum")
        axes[1, 1].set_ylabel("Size (MB)")
        axes[1, 1].set_title("Model Size")

        params = [((metrics_data[name]["params"] or 0) / 1e6) for name in model_names]
        axes[1, 2].bar(model_names, params, color="lightgray")
        axes[1, 2].set_ylabel("Parameters (M)")
        axes[1, 2].set_title("Parameter Count")

        plt.tight_layout()
        summary_path = FIGURES_FOLDER / "evaluation_results.png"
        plt.savefig(summary_path, dpi=300)
        print(f"[OK] Summary plot saved to {summary_path}")

        fig_cm, ax_cm = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))
        if len(model_names) == 1:
            ax_cm = [ax_cm]
        for idx, name in enumerate(model_names):
            conf = metrics_data[name]["confusion"]
            matrix = np.array([[conf["tp"], conf["fp"]], [conf["fn"], conf["tn"]]])
            im = ax_cm[idx].imshow(matrix, cmap="Blues")
            ax_cm[idx].set_title(f"Confusion: {name}")
            ax_cm[idx].set_xticks([0, 1])
            ax_cm[idx].set_yticks([0, 1])
            ax_cm[idx].set_xticklabels(["Pred Pos", "Pred Neg"])
            ax_cm[idx].set_yticklabels(["Actual Pos", "Actual Neg"])
            for (i, j), val in np.ndenumerate(matrix):
                ax_cm[idx].text(j, i, f"{int(val)}", ha="center", va="center", color="black", fontsize=12)
        plt.tight_layout()
        cm_path = FIGURES_FOLDER / "confusion_matrices.png"
        plt.savefig(cm_path, dpi=300)
        print(f"[OK] Confusion matrices saved to {cm_path}")
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
