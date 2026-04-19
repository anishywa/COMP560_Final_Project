# COMP560 Project: Face Recognition

## Task

Build a system that determines whether two face templates belong to the same person. Given template pairs, produce a similarity score for each pair.

## Datasets

| Dataset | Images | Templates | Pairs | Image Size |
|---------|--------|-----------|-------|------------|
| dataset_a | ~228K | ~12K | ~8M | 112x112 |
| dataset_b | ~469K | ~23K | ~15.7M | 112x112 |

Data format (Parquet):
- `test.parquet`: image metadata (image_path, template_id, media_id, landmarks, detection_score)
- `pairs.parquet`: verification pairs (template_id_1, template_id_2, label)

Each template consists of one or more face images. The evaluation protocol aggregates image-level features into template-level features, then computes cosine similarity between template pairs. See the baseline script for details.

## Submission Format

Submit a CSV file per dataset with columns:

```csv
template_id_1,template_id_2,score
1,11065,0.732
1,11066,0.215
...
```

- `template_id_1`, `template_id_2`: template pair IDs (must match pairs.parquet)
- `score`: similarity score (higher = more likely same identity)

## Evaluation

Open `evaluate.ipynb`, set `STUDENT_ID`, `PREDICTION_PATH`, and `DATASETS` in the Configuration cell, then run all cells.

## Baseline

Open `models/resnet_baseline.ipynb`, set `DATASET_ROOT` and `OUTPUT` in the Configuration cell, and run all cells.

## Training Example

Open `train_example.ipynb`. Set `MODE = "train"` (or `"predict"`) and configure the variables in the Configuration cell, then run all cells.

## Metrics

- **TAR@FAR=1e-6, 1e-5, 1e-4, 1e-3**: True Accept Rate at various False Accept Rates
- **AUC**: Area Under the ROC Curve

## Grading

- 40% Performance (TAR@FAR metrics)
- 30% Efficiency (model design, embedding dimension)
- 30% Report

## Training (Advanced)

Open `train.ipynb`. In the Configuration cell, set:
- `PHASE`: `"1"`, `"2"`, or `"both"` (default)
- `SCHEDULE`: `"step"` or `"cosine"`
- `EMBEDDING_DIM`: `128`, `256`, or `512`
- `DEBUG`: `True` for a fast 500-image, 2-epoch smoke test

Run all cells. Checkpoints are saved to `SAVE_DIR` (`./checkpoints` by default).

## TAR@FAR Evaluation

Open `eval_tar.ipynb`. Set `CHECKPOINT_PATHS` (a list), `DATASET_ROOT`, and optionally `OUTPUT_CSV` and `PLOT_PATH`, then run all cells. Reports TAR at FAR={1e-4, 1e-5, 1e-6}, throughput, and peak GPU memory.

## Ablation

Open `ablation.ipynb`. Set `DATA_ROOT`, `DATASET_ROOT`, and `DEBUG` in the Configuration cell, then run all cells. Runs the full 3×3 matrix ({phase 1, phase 2, both} × {D=128, 256, 512}) and displays a summary table saved to `results/ablation/summary.csv`.

## Directory Structure

```
project-fr/
├── datasets/
│   ├── dataset_a/
│   │   ├── images/          # Face images
│   │   ├── test.parquet     # Image metadata
│   │   └── pairs.parquet    # Verification pairs
│   └── dataset_b/
│       ├── images/
│       ├── test.parquet
│       └── pairs.parquet
├── models/
│   ├── resnet_baseline.ipynb  # Baseline prediction generator
│   └── face_encoder.ipynb     # FaceEncoder model definition + sanity check
├── train.ipynb                # Two-phase ArcFace training
├── eval_tar.ipynb             # TAR@FAR evaluation + ROC plot
├── ablation.ipynb             # Ablation matrix driver
├── evaluate.ipynb             # Evaluation script
├── train_example.ipynb        # Training example
└── results/                   # Output directory
```
