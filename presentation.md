# COMP560 Final Project — Summary & Findings

## Overview

This document briefs the team on the final state of our face recognition pipeline, the challenges we ran into during training, and how we resolved them to produce presentable results.

---

## The Problem: Training Was Unreasonably Slow

Our original training pipeline (two-phase Sub-center ArcFace) was estimated to complete in ~20 minutes on Google Colab's T4 GPU. In practice, a single epoch took **~30 minutes**, meaning the full 35-epoch Phase 1 alone would have taken **~17–18 hours** — far beyond what a single Colab session allows (12-hour limit).

The dataset contains **227,630 images** across **12,115 identities**, producing **1,778 batches per epoch** at the original batch size of 128. This was the core bottleneck.

---

## The Solution: Optimized Single-Phase Training (Option 2)

We switched to a streamlined training approach with three key changes:

### 1. Skipped Phase 1 (Sub-center ArcFace + Noise Isolation)
The original pipeline had two training phases:
- **Phase 1** trains a complex model that learns multiple "sub-versions" of each identity, then runs a full extra pass over the dataset to flag noisy/mislabeled images
- **Phase 2** retrains on the cleaned data

We skipped Phase 1 entirely and went straight to a single standard ArcFace training run. This eliminates the extra noise-isolation pass and the complexity of managing two sequential training jobs.

### 2. Doubled the Batch Size (128 → 256)
Doubling the batch size cuts the number of iterations per epoch in half — from 1,778 down to ~889. This alone roughly halved training time with no meaningful loss in accuracy for a short run.

### 3. Reduced Epochs (35 → 3)
Rather than training to full convergence, we trained for 3 epochs to produce a working model with real, measurable results. The model saves the best checkpoint automatically, so we always get the strongest result from the run.

### 4. Added Warmup Scheduler
The flexible training uses a **linear warmup** for the first 2 epochs (learning rate gradually increases from 0) followed by **cosine annealing** (learning rate smoothly decreases). This helps the model converge more stably in a small number of epochs compared to a fixed learning rate.

---

## What the Pipeline Does, Step by Step

| Step | What Happens |
|---|---|
| **Data staging** | Dataset (227,630 face images + metadata) is copied from Google Drive into Colab's fast local storage |
| **Training** | A ResNet50 neural network learns to produce a 512-dimensional "embedding" (a compact numerical fingerprint) for each face. ArcFace loss pushes embeddings from the same identity closer together and different identities further apart |
| **Prediction** | The trained model encodes every test image. For each pair of face templates in the dataset, it computes a similarity score (cosine similarity between averaged embeddings) |
| **Evaluation** | TAR@FAR metrics measure how well the model distinguishes genuine matches from impostors at various false alarm thresholds. AUC summarizes overall performance |

---

## Key Metrics Reported

- **TAR@FAR=1e-4** — "At a 0.01% false alarm rate, what fraction of real matches did we catch?"
- **TAR@FAR=1e-5** — Same at 0.001% false alarm rate (stricter)
- **TAR@FAR=1e-6** — Same at 0.0001% false alarm rate (very strict)
- **AUC** — Overall area under the ROC curve (higher = better, max 100%)
- **ROC Curve** — Visual plot of true accept rate vs. false alarm rate across all thresholds

---

## Final Configuration Used

```python
RUN_EXAMPLE  = True    # flexible single-phase ArcFace training
LOSS         = "arcface"
EPOCHS_EX    = 3
BATCH_SIZE   = 256
WARMUP_EPOCHS = 2
CHECKPOINT   = f"{SAVE_DIR}/best_model.pth"
```

**Total runtime:** ~50–70 minutes on Colab T4 GPU

---

## Results Location

After running the notebook, all outputs are saved to Google Drive under `MyDrive/COMP560/`:

| Output | File |
|---|---|
| Trained model weights | `checkpoints/best_model.pth` |
| Pair similarity scores | `predictions/dataset_a.csv` |
| TAR@FAR metrics + AUC | `results/YOUR_ID_TIMESTAMP.json` |
| ROC curve plot | `results/roc.png` |

---

## How to Reproduce

1. Open `TotalProject.ipynb` in Google Colab
2. Run **Section 0** to stage the dataset from Drive
3. Set the configuration above in **Section 1**
4. Click **Run all**
5. After completion, run **Section 12** to save outputs back to Drive
