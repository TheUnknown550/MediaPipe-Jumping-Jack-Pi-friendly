# MediaPipe Jumping Jack (Raspberry Pi friendly)

Lightweight MediaPipe Pose demos for counting jumping jacks, plus tooling to evaluate and quantize the pose_landmarker_lite model. Works with the bundled sample videos or a live webcam feed.

## Repo layout
```
.
|- src/                       # Python entry points
|  |- evaluation.py           # Batch evaluation on sample videos
|  |- evaluation_camera.py    # Webcam session with CSV snapshot
|  |- jumpingjacks.py         # Simple live counter overlay
|  |- run_pose_tflite.py      # Minimal TFLite pose runner
|  |- quantize_fp16.py        # Float16 converter for SavedModel -> TFLite
|  |- quantize_int8.py        # Int8 converter with representative dataset
|- models/                    # Pose models live here (lite task + tflite)
|- data/                      # Sample videos, ground truth CSV, evaluation PNG/CSV
|  |- exported_videos/        # Annotated clips from earlier runs
|  |- exported_videos_backup/ # Unused copy kept for reference
|- notebooks/                 # Jupyter notebooks + ground truth/prediction text
|- docs/setup-notes.md        # pyenv + Mediapipe setup steps for Linux
|- requirements.txt           # Runtime deps (TensorFlow only needed for quantization)
|- .gitignore
|- .gitattributes
```

## Quick start
1) Python 3.12 recommended (see `docs/setup-notes.md` for pyenv/venv hints).
2) Install deps:
```bash
pip install -r requirements.txt
```
3) Ensure the pose models exist in `models/` (already included):
   - `pose_landmarker_lite.task`
   - `pose_landmarker_lite.tflite`

## Running the scripts
- **Offline evaluation on sample set**
  ```bash
  python src/evaluation.py
  ```
  Uses `data/1.mp4`–`7.mp4` with ground truth from `notebooks/ground_truth.txt`.
  Artifacts land in `outputs/mp_evaluation_<timestamp>/` (videos, figures, tables).

- **Webcam evaluation with metrics export**
  ```bash
  python src/evaluation_camera.py
  ```
  Counts reps live, then prompts for the actual count to compute MAE/RMSE and writes a CSV under `outputs/mp_webcam_<timestamp>/tables/`.

- **Lightweight live counter overlay**
  ```bash
  python src/jumpingjacks.py
  ```
  Shows FPS/latency and increments reps when arms/legs complete a jumping-jack cycle. Press `q` to exit.

- **TFLite pose demo (no counting logic)**
  ```bash
  python src/run_pose_tflite.py
  ```
  Draws the MediaPipe skeleton on webcam frames using the lite TFLite model.

- **Model quantization (optional)**
  - Float16: `python src/quantize_fp16.py --saved_model path/to/saved_model`
  - Int8: `python src/quantize_int8.py --saved_model path/to/saved_model --rep_data data/representative_frames`
  Outputs are saved to `models/` by default.

## Data and notebooks
- Sample ground-truth counts: `notebooks/ground_truth.txt`
- Sample predictions used in the original report: `notebooks/predictions.txt`
- Visualizations/analysis notebooks: `Evaluation.ipynb`, `Metrics_Plot.ipynb`
- Existing evaluation artifacts (CSV + PNG) live under `data/` for reference.

## Notes
- Run commands from the repo root; scripts resolve paths relative to their own file location.
- Generated artifacts (`outputs/`) are ignored by git to keep the repo tidy.
