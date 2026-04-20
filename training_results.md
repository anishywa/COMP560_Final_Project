# COMP560 Final Project — Training Results

## Training Configuration

| Parameter | Value |
|---|---|
| Architecture | ResNet50 + ArcFace |
| Dataset | 227,630 images, 12,115 identities |
| Epochs | 3 |
| Batch size | 256 |
| Batches per epoch | 889 |
| Optimizer | AdamW |
| Scheduler | Linear warmup (2 epochs) + cosine annealing |

---

## Training Log

| Epoch | Loss | Time | Speed | Best Model Saved |
|---|---|---|---|---|
| 1 / 3 | 22.6751 | 28:53 | 1.95 s/batch | Yes |
| 2 / 3 | 18.7612 | 28:00 | 1.89 s/batch | Yes |
| 3 / 3 | 16.1323 | 27:23 | 1.85 s/batch | Yes |

**Total training time: ~84 minutes**
**Final best loss: 16.1323**

---

## Takeaways

### 1. The model improved consistently across all 3 epochs
Loss dropped every single epoch — from 22.68 down to 16.13, a **29% reduction** over the full run. The best model was saved after every epoch, meaning the final checkpoint (`best_model.pth`) represents the strongest point in training.

### 2. The loss curve shows healthy learning
A decreasing ArcFace loss means the model is successfully learning to push embeddings from different identities further apart while pulling same-identity embeddings closer together. The consistent improvement with no spikes or plateaus indicates stable training.

### 3. Speed improved slightly each epoch
Batch processing time decreased from 1.95 s/batch → 1.85 s/batch across epochs. This is consistent with the warmup scheduler — in the early steps the learning rate is ramping up, and as training stabilizes the GPU utilization becomes more efficient.

### 4. Training time was ~84 minutes total
Each epoch took approximately 27–29 minutes on a T4 GPU (889 batches at ~1.9 s/batch). This confirms the optimized configuration (batch size 256, 3 epochs) successfully brought total runtime within one Colab session.

### 5. Loss has not fully converged
A loss of 16.13 after 3 epochs is still relatively high — ArcFace loss on a well-trained model typically converges to single digits. This means the model has learned meaningful representations but has not reached its full potential. Additional epochs would continue to push the loss down and improve TAR@FAR scores.

---

## What the Loss Value Means

ArcFace loss is a classification-style loss that penalizes the model when same-identity faces are not close together in embedding space. Lower loss = better separation between identities.

- **22.68 (epoch 1)** — Model is beginning to organize face embeddings but many identities are still confused
- **18.76 (epoch 2)** — Clearer separation forming between identities
- **16.13 (epoch 3)** — Model has learned a meaningful embedding space; genuine pairs score noticeably higher than impostor pairs

This trajectory directly explains the TAR@FAR results — with more epochs, the loss would continue dropping and TAR@FAR scores would rise substantially.

---

## Connection to TAR@FAR Results

The training loss and evaluation metrics tell a consistent story:

| Training Loss | TAR@FAR=1e-4 | Interpretation |
|---|---|---|
| 16.13 (3 epochs) | 11.78% | Early-stage model, learning but not converged |
| ~8–10 (est. 10 epochs) | ~30–50% | Partially converged |
| ~3–5 (est. 35 epochs) | ~70–90% | Fully converged, production-quality |
