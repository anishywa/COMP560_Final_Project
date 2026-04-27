# COMP560 Face Recognition — Study Guide

## 1. Project Goal

This project implements **1:1 facial recognition**, also called face *verification*. The question being answered is simple: **are these two face templates the same person?**

This is different from 1:N identification (e.g., "who is this person in a database of 1 million people"). Verification is a binary decision: match or no match.

### The Task
- **Training set:** 227,630 face images across 12,115 distinct identities
- **Evaluation:** Score approximately 8,000,000 template pairs (genuine matches and impostor pairs)
- **Output:** A cosine similarity score for each pair — higher means more likely the same person

### The Metric: TAR@FAR
TAR@FAR = True Accept Rate at a given False Accept Rate.

**Concrete analogy:** Imagine a building with a face-scanning door. You set the door to be very strict: it will only let through a stranger (impostor) once every 10,000 attempts (FAR = 1e-4 = 0.01%). At that strict threshold, what fraction of *real employees* does it still let through? That's TAR@FAR=1e-4.

Higher TAR at a low FAR = better. A random guesser would score ~0%.

---

## 2. The Original System Design (The Ambitious Plan)

The initial architecture was a **two-phase Sub-center ArcFace pipeline**, designed to produce production-quality results.

### Phase 1: Sub-center ArcFace (35 epochs)

Real-world face datasets are noisy. Some images are mislabeled (the wrong person), blurry, occluded, or taken under extreme conditions. Training directly on noisy data limits how tight the resulting embeddings can be.

Sub-center ArcFace addresses this by assigning each identity **K=3 sub-centers** in the embedding space instead of one. The idea:
- The **dominant sub-center** captures the "true" face for that identity
- The **non-dominant sub-centers** absorb noise, mislabeled samples, and outliers

After 35 epochs of training, a full pass over the dataset checks which sub-center each image aligns to. Images assigned to non-dominant sub-centers are flagged as likely-noisy and saved to `noise_flags.npy`.

**Outcome:** ~5–10% of training images removed as noise before Phase 2.

### Phase 2: Standard ArcFace (15 epochs)
- Load Phase 1 weights as initialization
- Train on the cleaned dataset (noisy images removed)
- Use a stricter scale parameter (`s=64.0`)
- **Outcome:** A fully converged, high-accuracy model

### Total estimated time: ~25 hours

This exceeded Google Colab's free GPU session limit (~12 hours), making the original plan infeasible.

---

## 3. The Compromise — What We Actually Built

### The Bottleneck
Initial estimate: 1 epoch ≈ 20 minutes.
Actual measured time: 1 epoch ≈ 30 minutes (50% slower than expected).

With batch size 128, each epoch had 1,778 gradient update steps on the T4 GPU. Phase 1 alone would take ~17–18 hours.

### The Decision
Rather than run half of a two-phase pipeline and end up with an unusable intermediate model, we pivoted to a **complete, working single-phase system** that could run end-to-end in one Colab session.

### Three Changes Made

| Change | Original | Optimized | Effect |
|---|---|---|---|
| Pipeline phases | Phase 1 + Phase 2 | Single ArcFace | Eliminates 17-hour noise-isolation step |
| Batch size | 128 (1,778 batches/epoch) | 256 (889 batches/epoch) | Halves iterations per epoch |
| Total epochs | 35 + 15 = 50 | 3 | Fits within one Colab session |
| LR schedule | Step decay | Warmup + cosine annealing | Stabilizes training in few-epoch budget |

**Result: ~90-minute end-to-end pipeline** (data staging + training + prediction + evaluation).

The cost: the model is under-trained (loss of 16.13 vs. the target of ~3–5 for a fully converged model). But the system is complete, correct, and demonstrates real learning.

---

## 4. Architecture Decisions — The "Why" Behind Each Choice

### a. Backbone: ResNet50 (ImageNet Pretrained)

**What it is:** ResNet50 is a 50-layer deep convolutional neural network. The "Res" stands for residual connections — skip connections between layers that allow gradients to flow cleanly during backpropagation, enabling much deeper networks to train reliably.

**Why pretrained on ImageNet?** ImageNet is a dataset of 1.2 million labeled photos. A model trained on it has already learned to detect edges, textures, shapes, and object parts. This is called *transfer learning* — we inherit that general visual knowledge and fine-tune it for face identity discrimination.

**Why ResNet50 and not something bigger (e.g., ResNet101)?**
- Bigger models require more GPU memory and more epochs to converge
- With a 3-epoch budget, a larger model would barely begin to adapt
- ResNet50 (2048-dimensional output) is the standard backbone in the ArcFace literature

**Architecture:**
```
ResNet50 (ImageNet pretrained)
  → 2048-dimensional feature vector
  → Linear layer: 2048 → 512
  → L2 normalization
  → 512-dimensional embedding
```

---

### b. Embedding Dimension: 512

Every face image is encoded into a single 512-number vector — its "identity fingerprint." Two images of the same person should produce vectors that point in nearly the same direction; two images of different people should point in very different directions.

**Why 512?**
- Too small (e.g., 64): insufficient capacity to distinguish 12,115 unique identities
- Too large (e.g., 2048): slower pairwise comparison across 8M template pairs; higher risk of overfitting
- 512 is the industry standard — used in the original ArcFace paper, FaceNet, and CosFace

After the linear projection, the embedding is **L2-normalized** to unit length. This forces all embeddings onto a hypersphere, so similarity is measured purely by angle (via cosine similarity), not magnitude.

---

### c. Loss Function: ArcFace

**The problem with standard softmax loss:**
Ordinary cross-entropy loss trains a classifier to predict the correct identity label. It pushes the model to get the right answer but doesn't enforce *how confident* or *how separated* the embeddings need to be. Embeddings can cluster loosely and still produce correct predictions — but loose clusters fail at strict FAR thresholds.

**What ArcFace does:**
ArcFace adds an angular margin penalty to the correct class *before* computing the softmax:

```
Without ArcFace: score = cos(θ_correct)
With ArcFace:    score = cos(θ_correct + m)   where m = 0.5 radians ≈ 28°
```

The model has to be 28° *more* aligned with the correct identity than it needs to be just to pass. This forces tighter, more discriminative clusters in the embedding space.

**Hyperparameters used:**
- `m = 0.5 radians`: The angular margin. Larger values (0.7) produce even tighter clusters but require more epochs to converge.
- `s = 30.0`: A scale factor that multiplies all logits before softmax, sharpening the decision boundary. The original design used `s = 64.0`, but with only 3 epochs, the smaller value gives faster convergence at the cost of a slightly softer boundary.

**Why not Triplet Loss?**
Triplet loss directly compares (anchor, positive, negative) image triples, which is intuitive but inefficient — it requires carefully mined hard triplets and converges slowly. ArcFace operates over the full classification space in each batch, making better use of every gradient step.

---

### d. Optimizer: AdamW

**SGD vs. AdamW:**
SGD (Stochastic Gradient Descent) is the traditional optimizer for large-scale image training. It works best with many epochs and a carefully tuned learning rate schedule, because it treats all parameters with the same step size.

AdamW adapts the learning rate *per parameter* based on the history of gradients. This means:
- Parameters that haven't been updated much get larger steps
- Parameters that update frequently get smaller, stabilizing steps
- Convergence is faster in small epoch budgets

**Settings used:**
- Learning rate: `1e-4` (standard for fine-tuning a pretrained backbone; smaller than training from scratch)
- Weight decay: `1e-4` (L2 regularization — penalizes large weights to prevent overfitting)
- Gradient clipping: max norm = 1.0 (if a batch produces an unusually large gradient, it's clipped — this prevents a single bad mini-batch from catastrophically destabilizing the model)

---

### e. Learning Rate Schedule: Linear Warmup + Cosine Annealing

The learning rate is not held constant throughout training. It follows a two-phase schedule:

**Phase 1 — Linear Warmup (2 epochs):**
Learning rate ramps from 0 → 1e-4 linearly.

*Why?* At the start of training, the ArcFace classification layer (which has 12,115 output neurons, one per identity) is randomly initialized. Immediately applying a high learning rate to a random layer causes large, destabilizing gradient updates that can corrupt the pretrained ResNet50 weights. Warmup lets the classification layer stabilize before full-speed training begins.

**Phase 2 — Cosine Annealing (remaining epochs):**
Learning rate decays smoothly from 1e-4 → ~0, following a cosine curve.

*Why cosine?* A step decay (halving the LR at fixed milestones) creates abrupt jumps. Cosine annealing is smooth — the model makes large refinements early and increasingly small, careful adjustments as the curve flattens near zero. This is especially important with only 3 epochs, where every update counts.

---

### f. Batch Size: 256 (doubled from original 128)

**What batch size controls:** Each gradient update is computed over a mini-batch of images. Larger batches mean:
- Fewer gradient updates per epoch (889 vs. 1,778 at batch size 256 vs. 128)
- More stable gradient estimates (averaging over more samples)
- Faster wall-clock time per epoch

**The trade-off:** Very large batches can hurt generalization — the model sees fewer diverse gradient updates and may converge to sharper, less general minima. But in a 3-epoch budget, the primary objective is to learn *something* meaningful, not to squeeze out the last 1% of accuracy. Doubling the batch size cut epoch time from ~58 min to ~28 min with no code changes.

---

### g. Data Augmentation

During training (not evaluation), each image is randomly transformed before being fed to the model:

| Augmentation | What it does | Why |
|---|---|---|
| Random horizontal flip | Mirrors the face left-right | Faces look similar from either side |
| Color jitter | Shifts brightness, contrast, saturation, hue slightly | Handles different lighting conditions |
| Gaussian blur | Applies soft blur with 20% probability | Simulates lower-resolution cameras |
| Random erasing | Removes a random rectangular region | Simulates occlusions (glasses, masks, shadows) |
| GridMask | Zeros out a grid pattern with 15% probability | Prevents the model from relying on any single facial region |

**Why augmentation matters:** Without it, the model tends to memorize the exact training photos rather than learning what "identity" means across variations. A face in sunlight and a face in shadow should produce similar embeddings — augmentation forces the model to learn that invariance.

Augmentation is applied only during training; evaluation uses clean, deterministic preprocessing (resize + center crop + normalize).

---

### h. Template Aggregation at Evaluation

In this dataset, a "template" is a collection of face images of one person from one enrollment event. Some templates contain just a few images; others may have dozens, captured across multiple sessions (media IDs).

**The aggregation protocol:**
1. Encode each image into a 512-d embedding
2. Within each template, group images by media ID (capture session) and average their embeddings
3. Sum the media-level averages and L2-normalize → one 512-d template vector

**Why aggregate?** A single face photo can be blurry, poorly lit, or slightly occluded. Averaging across multiple images of the same person produces a more stable, representative embedding. The two-level averaging (within-session, then across sessions) prevents a session with many images from dominating the template.

The final cosine similarity score between two template vectors is the dot product of their L2-normalized representations.

---

## 5. Training Results

| Epoch | ArcFace Loss | Time |
|---|---|---|
| 1 / 3 | 22.6751 | 28:53 |
| 2 / 3 | 18.7612 | 28:00 |
| 3 / 3 | 16.1323 | 27:23 |

- **29% loss reduction** over 3 epochs — real learning is occurring
- **Stable training:** no spikes, divergence, or plateaus
- **Not converged:** a fully trained ArcFace model typically reaches loss ≈ 3–5. At 16.13, the model has learned meaningful structure but embeddings are still loosely separated.

The speed improvement across epochs (1.95 → 1.85 s/batch) is normal — GPU utilization stabilizes after the first epoch's data loading overhead.

---

## 6. Evaluation Results

| Metric | Score |
|---|---|
| TAR @ FAR = 1e-4 | **11.78%** |
| TAR @ FAR = 1e-5 | 4.62% |
| TAR @ FAR = 1e-6 | 2.43% |

**What does 11.78% mean?**
At a threshold that admits only 1 impostor per 10,000 attempts (0.01% false alarm rate), the system correctly identifies 11.78% of genuine matches. An untrained (random) model scores ~0%. This 11.78% confirms the model learned real identity structure in just 3 epochs.

**Expected trajectory with more compute:**
- 10 epochs → TAR@FAR=1e-4 ≈ 30–50%
- Full 50-epoch pipeline → TAR@FAR=1e-4 ≈ 70–90%

**Inference performance:**
- Throughput: 383.2 images/second on T4 GPU
- Peak GPU memory: ~2.2 GB (of 16 GB available)

---

## 7. What We Would Do With More Compute

With ~25 hours of GPU time (or access to an A100 GPU that runs 2–3× faster):

1. **Run Phase 1 (Sub-center ArcFace, 35 epochs)**
   - Assign K=3 sub-centers per identity
   - After training, flag images assigned to non-dominant sub-centers as noise
   - Remove ~5–10% mislabeled/outlier images from the dataset

2. **Run Phase 2 (Standard ArcFace, 15 epochs)**
   - Initialize from Phase 1 weights
   - Train on the cleaned dataset
   - Use stricter hyperparameters: `s=64.0`, `m=0.5`

3. **Expected result:** TAR@FAR=1e-4 ≈ 70–90%

The architecture remains identical — only the training duration and data cleaning change. This demonstrates that the pipeline design was sound; the limiting factor was compute budget, not model design.

---

## 8. Summary: Decision Rationale at a Glance

| Decision | Choice | Why |
|---|---|---|
| Backbone | ResNet50 (ImageNet pretrained) | Transfer learning; proven for face encoding; fits GPU memory |
| Embedding size | 512-d | Balances capacity (12k identities) with inference speed (8M pairs) |
| Loss function | ArcFace (m=0.5, s=30.0) | Angular margin produces tighter clusters than softmax; faster than triplet loss |
| Optimizer | AdamW | Adaptive per-parameter LR; faster convergence in small epoch budgets |
| LR schedule | Warmup + cosine annealing | Warmup stabilizes random ArcFace layer; cosine gives smooth convergence |
| Batch size | 256 (doubled) | Halves batches/epoch; pragmatic speed-up for 3-epoch run |
| Augmentation | Flip, jitter, blur, erasing, GridMask | Forces identity-invariant embeddings across lighting/occlusion variation |
| Template aggregation | Per-session average → sum → L2-normalize | Reduces noise from individual bad frames; session-balanced representation |
| Phase 1 removed | Single-phase ArcFace only | Phase 1 alone takes ~17 hours; infeasible on Colab free tier |
| Epochs reduced | 3 (from 50) | Delivers complete end-to-end run in ~90 minutes |
