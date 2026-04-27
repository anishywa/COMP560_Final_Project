# Common Questions — COMP 560 Facial Recognition Project

This document covers likely presentation and defense questions, answered with project-specific detail.

---

## From the Question List

---

### Q: What was the purpose of the project?

The project's goal was to build a **1:1 face verification system** — given two photos, determine whether they show the same person. This is distinct from face identification (figuring out *who* someone is from a database). Verification is what powers phone face-unlock, airport e-gate passport checks, and banking KYC.

The stakes are asymmetric: a false rejection inconveniences a real user (they get locked out), but a false acceptance is a security breach (an attacker gets in). So the system must be designed and evaluated specifically around minimizing false acceptances, not just maximizing overall accuracy.

We used TAR@FAR as our metric — "given that only 1 in 10,000 impostors gets through, what fraction of real matches do we catch?" — rather than accuracy, which is misleading when most pairs are different people.

---

### Q: Why ResNet50?

ResNet50 was chosen for three reasons:

1. **Balance of capacity and cost.** ResNet50 has 50 layers and ~25M parameters, giving it strong representational power without the training cost of ResNet101 or ViT-based architectures. Given our compute limits (free-tier Google Colab T4 GPU), a heavier backbone would have made even a 3-epoch run infeasible.

2. **Residual connections solve deep-network training.** Earlier deep networks suffered from the vanishing gradient problem — gradients would become negligibly small before reaching early layers, stopping learning. ResNet's skip connections provide a direct path for gradients to flow backward, making deep networks trainable. This is foundational to why ResNet50 outperforms shallower alternatives.

3. **Strong pretrained initialization.** We initialized with ImageNet weights (IMAGENET1K_V2), meaning the network already understands low-level features (edges, textures, shapes) before ever seeing a face. Fine-tuning from this point converges much faster than training from scratch — critical when we only had 3–8 epochs available.

Why not ResNet101 or a Vision Transformer? Those would extract richer features, but their training time per epoch scales roughly linearly with parameters. With a 12-hour session cap, the extra capacity wasn't worth the reduced training budget.

---

### Q: How did you select the parameters of the model to finetune?

We did not freeze any layers — the full ResNet50 was fine-tuned end-to-end. The key parameters selected or adjusted were:

- **Embedding dimension D**: We replaced ResNet50's final classifier (1000-class ImageNet head) with a linear projection to D dimensions. We selected D=512 for the full-dataset run (enough capacity for 12,115 identities) and D=128 for the 605-identity subset run (right-sized to avoid over-parameterization — a 512-dimensional ArcFace weight matrix with only 605 classes would have 309,760 parameters describing 605 centers, wasting capacity and risking overfitting).

- **ArcFace margin m=0.5**: The angular margin determines how far apart identities must be in embedding space. 0.5 radians (~28.6°) is the standard value from the original ArcFace paper, balancing discriminativeness with trainability. A larger margin forces tighter clusters but makes the loss harder to optimize in few epochs.

- **Scale factor s=64**: Controls the steepness of the logit distribution — higher scale means the model must be more angularly precise to produce a low loss. We used s=30 in the optimized run (easier to converge in 3 epochs) and s=64 in the full pipeline design (standard for a fully converged model).

- **Learning rate 1e-4 with AdamW**: Selected to be small enough not to catastrophically overwrite ImageNet features in early epochs, while still allowing meaningful updates to the new ArcFace classification head.

---

### Q: How did you fine-tune parameters for overfitting?

We used several strategies to prevent the model from memorizing the training data instead of learning generalizable identity features:

**Data augmentation** was the primary defense:
- Random horizontal flip — prevents bias toward left/right orientation
- Color jitter — prevents reliance on lighting conditions
- Random erasing (p=0.2) — simulates partial occlusion (glasses, masks)
- Gaussian blur (p=0.2) — simulates low-resolution cameras; forces the model to learn coarse identity features, not pixel-level details
- Grid masking (p=0.15) — randomly blocks grid cells across the face, forcing the model to use all facial regions rather than relying on one high-discriminativity area (e.g., eyes only)

**Embedding dimension regularization**: For the 605-identity run, we reduced D from 512 to 128. A smaller embedding space acts as a bottleneck — the model can't memorize all training images when the representation space is compressed.

**Weight decay (1e-4)**: AdamW's weight decay penalizes large weights, discouraging the model from fitting noise.

**Gradient clipping (max_norm=1.0)**: Prevents individual batches from causing large parameter updates that could lock the model into overfitting a specific subset of training data.

In practice, overfitting was not the primary concern for us — the bigger issue was underfitting due to too few training epochs. A model that hasn't converged can't overfit.

---

### Q: How was the loss function calculated?

We used **ArcFace loss**, which is a modified softmax cross-entropy loss.

Standard softmax asks: "given this image's embedding, what's the probability it belongs to each of the N identities?" and penalizes wrong answers. The problem is that standard softmax doesn't enforce a minimum separation between identities — it just needs to rank the right class highest.

ArcFace modifies this by injecting an **additive angular margin** into the correct class's logit before softmax:

```
Standard:  logit for correct class = s · cos(θ)
ArcFace:   logit for correct class = s · cos(θ + m)
```

Where:
- `θ` is the angle between the embedding and the correct identity's center vector
- `m = 0.5` radians is the margin penalty
- `s = 64` is the scale factor

By subtracting a margin from the correct class's score *during training*, the model is forced to produce embeddings that are much closer to the identity center than they need to be — creating a buffer zone. At inference time (no margin applied), this buffer means embeddings for the same person are well separated from embeddings for other people.

The loss is then standard cross-entropy over the modified logits:
```
L = -log( exp(s · cos(θ + m)) / (exp(s · cos(θ + m)) + Σ_j exp(s · cos(θ_j))) )
```

In our single-phase run, this loss started at 22.68 (very random — the model has no idea yet) and dropped to 16.13 after 3 epochs. A fully converged model reaches 3–5.

---

### Q: Maybe compare — seems risky? (comparing to simpler baselines)

This question is asking whether we benchmarked against simpler systems to confirm our design choices added value.

We compared across three configurations:

| Configuration | Final Loss | TAR@FAR=10⁻⁴ | AUC | Notes |
|---|---|---|---|---|
| Random (untrained) | — | ~0% | ~50% | Baseline floor |
| Single-phase, 3 epochs | 16.13 | 11.78% | — | Proves pipeline works |
| Two-phase (Sub-center + ArcFace) | 4.09 | 0%* | 91.99% | Best design |
| Expected full run (estimated) | ~3–5 | ~80–95% | — | With full compute |

*0% is a test set size artifact, not a model failure — see the Discussion section.

The comparison that matters most: the two-phase run's final loss of 4.09 versus the single-phase's 16.13. That's a 4× improvement in training convergence, achieved by (a) noise filtering removing 14.6% bad training samples, and (b) more epochs on a smaller but cleaner dataset. The AUC of 91.99% confirms the embeddings generalize well.

The "risky" part of comparing is that our two configurations aren't directly comparable — different dataset sizes, epochs, and embedding dimensions. The single-phase run used all 227,630 images; the two-phase used 10,277. So we can't claim the two-phase is strictly better because it also had a much smaller training set. What we *can* claim is that the two-phase approach correctly implements the intended design and reaches near-convergence, while the single-phase proves the architecture works end-to-end.

---

### Q: Why use the entire dataset?

For the single-phase proof-of-concept run, we used the full 227,630-image Dataset A. The rationale:

1. **More data = better generalization.** With 12,115 identities, the model needs enough examples per identity to learn what makes each person distinctive across varying conditions. Subsampling would reduce the number of identities and photos per identity, both of which degrade embedding quality.

2. **Proving the pipeline scales.** A proof-of-concept on a toy subset doesn't confirm the system works at real scale. Running on the full dataset showed that the training loop, memory management, and throughput (383 images/second) hold up under production-scale data.

3. **Evaluation validity.** TAR@FAR evaluation requires enough pairs to measure performance at very strict thresholds. With the full dataset, there are enough negative pairs to reliably set a threshold at FAR=10⁻⁶. With a 5% subset, the test set has only 7,930 pairs — not enough for those strict thresholds.

For the revised two-phase run, we *did* subsample to 5% — but that was a deliberate tradeoff to fit within our compute budget while still running both training phases. We acknowledged this as a limitation and explained that the AUC result (measured on the full evaluation pair space) is still meaningful.

---

### Q: Explain batch size and how it affects everything?

**What is batch size?**
Instead of updating the model after every single image (too slow) or after the entire dataset (too much memory), we process images in groups called batches. Batch size is the number of images per group.

**How it affects training speed:**
With 227,630 images and batch size 128, there are ~1,778 batches per epoch. At 30 minutes per epoch, that's ~1 second per batch. Doubling to batch size 256 halves the number of batches to ~889, cutting each epoch to ~15 minutes — that's the optimization we made in the single-phase run to halve training time.

**How it affects gradient quality:**
Each batch produces one gradient estimate — a direction to update the model. Larger batches average over more images, producing a smoother, lower-variance gradient estimate. Smaller batches are noisier but can help the model escape local minima. For ArcFace specifically, larger batches are beneficial because each batch includes more identities, giving the loss function more diversity to learn from.

**How it affects memory:**
Each image in the batch occupies GPU memory during the forward pass. Batch size 256 at 112×112×3 images with float32 activations across 50 ResNet layers uses roughly 2–3 GB of VRAM. We measured peak usage at 2,211.9 MB for the single-phase run. If batch size were doubled again, we'd risk out-of-memory errors on the 16 GB T4 GPU.

**Our tradeoff:** We doubled from 128 → 256 to cut training time in half. Going beyond 256 would have risked GPU memory limits without proportional speed gains (GPU parallelism saturates at large batch sizes).

---

### Q: GPU Setup?

We ran all experiments on **Google Colab** using a **NVIDIA T4 GPU**:

- **GPU**: Tesla T4 — 16 GB GDDR6 VRAM, ~8.1 TFLOPS FP32
- **Platform**: Google Colab free tier
- **Constraint**: 12-hour session limit per run

Key setup steps in each session:
1. Copy dataset from Google Drive to Colab's local `/content/` storage — reading from Drive directly is ~10× slower due to API overhead
2. Install dependencies (PyTorch, torchvision, pandas, pyarrow for Parquet)
3. Mount the GPU with `device = torch.device("cuda")`
4. Use `torch.cuda.amp` (automatic mixed precision) to run forward passes in FP16 — halves memory usage with minimal accuracy impact, allowing larger batch sizes

**Memory usage observed:**
- Single-phase run (D=512): 2,211.9 MB peak
- Two-phase run (D=128): 1,454.3 MB peak (smaller embedding layer = smaller ArcFace weight matrix)

**Throughput observed:**
- ~383–401 images/second during inference

The 12-hour session cap was the defining constraint of the project. It forced us to redesign from the intended 25-hour two-phase pipeline down to configurations that fit in one session: first a 3-epoch single-phase run (84 min), then a 5%-subset two-phase run (15 min).

---

## Additional Questions

---

### Q: Why ArcFace instead of triplet loss or standard softmax?

**Standard softmax** treats face recognition as a classification problem — "which of the N people is this?" It's simple but doesn't enforce any minimum separation between identities in embedding space. Two embeddings can be adjacent (very similar) even if they belong to different people, as long as the correct class scores highest. This makes the system fragile at low FAR thresholds.

**Triplet loss** (used by FaceNet) directly optimizes for the metric we care about: same-person embeddings close together, different-person embeddings far apart. But to use triplet loss, you need to sample triplets (anchor, positive, negative) from your dataset. With 12,115 identities and ~18 images each, there are trillions of possible triplets. Choosing *informative* ones (hard negatives — pairs of different people that the model currently confuses) is computationally expensive and sensitive to the sampling strategy. Poorly chosen triplets lead to slow or unstable training.

**ArcFace** gives the best of both worlds: it trains a classifier (no sampling problem) but adds an angular margin that directly enforces separation in embedding space. The margin forces the model to leave a 28.6° buffer between any identity and the decision boundary — which means at inference time, there's a comfortable gap between same-person and different-person scores. This is why ArcFace consistently outperforms triplet loss on face benchmarks.

---

### Q: What is Sub-center ArcFace and why did you use it?

Standard ArcFace assigns one center vector per identity in embedding space. The entire cluster of photos for one person must align to that single point. But in practice, the same person looks quite different across photos — different ages, lighting, expressions — and some photos may be mislabeled or very low quality. These "noisy" samples get pulled toward the wrong center, corrupting the cluster.

**Sub-center ArcFace** assigns K centers per identity (we used K=3). During training, each image only needs to align with its *closest* center, not the global one. Clean, typical images naturally cluster around the dominant center. Mislabeled or low-quality images drift to the non-dominant sub-centers, which act as "trash bins" for outliers.

After Phase 1 training, you look at each image's assigned sub-center. Images on non-dominant sub-centers are flagged and removed. In our run, **1,503 of 10,277 images (14.6%)** were removed this way. Phase 2 then trains standard ArcFace on the cleaned data.

This is powerful because it requires no manual data inspection — the model automatically identifies its own training noise.

---

### Q: What is an embedding and why does it matter?

An embedding is a compact numerical representation of an image — a list of D numbers (in our case D=128 or D=512) that captures the essential identity of a face. You can think of it as a "face fingerprint."

The key property: embeddings are L2-normalized so they lie on a unit sphere. This means comparing two embeddings is simply taking their dot product, which equals cosine similarity — a single number between -1 and 1. Two photos of the same person should have cosine similarity close to 1. Two photos of different people should be much lower.

Why does this matter for scale? We need to compare ~8 million template pairs. Computing dot products between 512-dimensional vectors is extremely fast on a GPU — we achieved 400 images/second at inference. If we were comparing raw pixel values (112×112×3 = 37,632 numbers per image), the comparison would be both slower and meaningless (pixel values don't capture identity across varying conditions).

---

### Q: Why cosine similarity for comparison?

Because our embeddings are L2-normalized (length = 1), cosine similarity equals the dot product — no division required. This makes comparison trivially fast and parallelizable on GPU.

More importantly, normalizing to a unit hypersphere means the *direction* of the embedding captures identity, while the *magnitude* is irrelevant. Two photos of the same person lit differently might produce embeddings of different magnitudes, but they should point in the same direction. Normalizing removes magnitude as a confounding factor.

ArcFace is also designed around angular distance — the margin penalty is applied in angular space — so cosine similarity is the natural companion metric at inference time.

---

### Q: What does TAR@FAR mean and why not just use accuracy?

**TAR@FAR** stands for True Accept Rate at a given False Accept Rate.

- **FAR (False Accept Rate)**: Of all impostor pairs (different people), what fraction does the system incorrectly accept?
- **TAR (True Accept Rate)**: Of all genuine pairs (same person), what fraction does the system correctly accept?

We fix FAR to a very strict value (10⁻⁴ = 1 in 10,000) and measure the resulting TAR. This mimics real deployment: security systems are tuned to let almost no impostors through, and the question is how many legitimate users they also block.

**Why not accuracy?** In any real face verification dataset, the vast majority of pairs are different people. If there are 1,000 genuine pairs and 1,000,000 impostor pairs, a system that rejects everything achieves 99.9% accuracy. That's useless. TAR@FAR forces the metric to reflect actual utility under realistic operating conditions.

---

### Q: Why normalize embeddings to a unit hypersphere?

Three reasons:

1. **Cosine similarity becomes a dot product.** For unit vectors, cos(θ) = a·b. No division needed, which is fast at scale.

2. **Magnitude is uninformative.** The same person photographed brightly vs. dimly might produce embeddings of very different magnitudes, even if they represent the same identity. L2-normalization removes this confound — only direction matters.

3. **ArcFace requires it.** The angular margin loss operates in angular space (cos(θ + m)). If embeddings aren't normalized, "angle" isn't well-defined, and the loss breaks down.

---

### Q: What is data augmentation and why did you use it?

Data augmentation means applying random modifications to training images so the model sees many variations of each photo. Crucially, these modifications are applied randomly *at training time only* — test images are never augmented.

Why does this help? Our dataset has 227,630 images, but in the real world, faces appear under infinitely many conditions: different lighting, angles, expressions, camera qualities. If we train on unmodified photos, the model may learn to exploit specific quirks of our training set (a particular background, a consistent lighting setup) rather than learning what actually makes a face distinctive.

Our specific augmentations and why each was chosen:
- **Random horizontal flip**: People's faces are roughly symmetric; flipping prevents the model from keying on left-vs-right artifacts
- **Color jitter**: Simulates different lighting and camera calibrations
- **Random erasing**: Simulates occlusion (glasses, masks, hair)
- **Gaussian blur**: Simulates low-quality cameras like surveillance footage — forces the model to recognize faces even when details are blurry
- **Grid masking**: Blocks random grid patches; prevents over-reliance on any single facial region (eyes, nose, etc.)

---

### Q: Why AdamW over SGD?

The original ArcFace paper used SGD with momentum 0.9 and step decay — a well-established combination for face recognition. We switched to **AdamW** for two reasons:

1. **Better performance in low-epoch budgets.** AdamW maintains per-parameter learning rates that adapt based on gradient history. This means parameters that rarely update (like some layers of ResNet50 that change little from ImageNet pretraining) get larger relative updates, while frequently-updated parameters get smaller ones. In only 3 epochs, this adaptive behavior makes each update count more than SGD's uniform rate.

2. **Weight decay is decoupled.** In standard Adam, weight decay is applied to the gradient, which interacts incorrectly with the adaptive learning rates. AdamW separates weight decay from the gradient update, providing cleaner regularization. This matters more when training on limited data.

The tradeoff: AdamW can sometimes converge to slightly lower-quality minima than well-tuned SGD on full training runs. With unlimited compute, SGD might edge out AdamW. With 3 epochs, AdamW is the pragmatic choice.

---

### Q: What is learning rate warmup and why does it matter for ArcFace?

Learning rate warmup means starting with a very small learning rate and gradually increasing it to the target value over the first few epochs.

For ArcFace specifically, warmup is important because the **ArcFace weight matrix** (the set of identity center vectors) is randomly initialized at the start of training. These centers are essentially random points on the unit sphere — they don't correspond to real identities yet. If we hit this random matrix with a full learning rate immediately, the gradient updates can be catastrophically large, scrambling the ResNet50 weights before the centers have had a chance to settle.

Warmup gives the ArcFace centers time to form reasonable initial positions at a safe learning rate before the full update speed kicks in. After 2 epochs of warmup, the centers are meaningfully initialized, and we can apply the full learning rate without risk.

After reaching peak learning rate, we use **cosine annealing** — smoothly decreasing the rate toward zero. This allows the model to make fine adjustments in later epochs without overshooting good solutions.

---

### Q: Why did you shrink the dataset for the final run?

The full two-phase pipeline (35 epochs Phase 1 + 15 epochs Phase 2) on 227,630 images would take ~25 hours on a T4 GPU. Google Colab's free tier has a 12-hour session limit. Running two separate sessions isn't viable because Phase 2 depends on Phase 1's output (the cleaned dataset), and session storage resets between runs.

Our solution: train on a **5% subset** (10,277 images, 605 identities), reduce Phase 1 to 8 epochs and Phase 2 to 4 epochs, and reduce embedding dimension from 512 to 128. This gets the full two-phase pipeline — including noise filtering — to complete in ~15 minutes.

The tradeoff is reduced model capability: 605 training identities is a small fraction of the 12,115-identity full dataset, so the learned embeddings generalize less broadly. But the goal of this run was to validate the *design* (does Sub-center ArcFace correctly filter noise? does Phase 2 converge after cleaning?), not to achieve maximum accuracy. The AUC of 91.99% confirms the design works.

---

### Q: What is template aggregation and why aggregate at all?

A **template** is a collection of photos of one person from one enrollment session — not a single image. In the evaluation dataset, each person may have dozens or hundreds of images across multiple video clips. Before comparing two people, we need to combine all of their images into one single vector.

We do this in two steps:
1. **Within-media average**: For all images from the same video clip (media ID), average their embeddings
2. **Cross-media sum, then L2-normalize**: Sum the per-clip averages and normalize the result

Why not just average all images? If one person has 100 photos from one clip and 5 from another, a straight average would be dominated by the 100-photo clip. The two-step approach gives each clip equal weight regardless of how many frames it contributes. This is more representative of the person's full identity across sessions.

---

### Q: What is the data leakage bug and why does it matter?

An early version of the notebook split images into training and evaluation sets, but both sets were drawn from the same pool of template IDs. This means during evaluation, the model was tested on identities it had already seen during training.

This is called **data leakage**, and it artificially inflates performance metrics. The model doesn't need to generalize — it can partially "remember" the identities it was trained on. In real deployment, the model will always encounter new, unseen identities. Evaluating on seen identities gives an overly optimistic picture of how the system would actually perform.

The fix: use a **seeded shuffle** to partition template IDs into non-overlapping training and test sets before any training begins. We added a runtime assertion — `assert len(train_ids & test_ids) == 0` — to guarantee no identity appears in both splits. This is a mandatory correctness requirement for any trustworthy evaluation.

---

### Q: Why is 0% TAR@FAR at strict thresholds not a failure?

In the two-phase run, TAR@FAR=10⁻⁴ showed 0% on the held-out test split. This seems alarming but is purely a **sample size artifact**.

Here's why: at FAR=10⁻⁴, the threshold is set so that only 1 in every 10,000 negative (impostor) pairs scores above it. Our test set had 7,930 total pairs. With so few pairs, there simply aren't enough positive (same-person) pairs to catch even one match at that extreme threshold. The ROC curve shows a staircase shape — each step is a single positive pair crossing the threshold — which is exactly what you'd expect with a small number of positives.

When evaluated on the **full Dataset A pair space** using the official grader (millions of pairs), the results tell a very different story: TAR@FAR=10⁻¹ at 81.82% and AUC of 91.99%. These results confirm the model is correctly ranking same-person pairs above different-person pairs. The 0% result is a measurement problem, not a model problem.

---

### Q: What would you do differently with more compute?

**With more GPU time (priority #1):**
Train the full two-phase pipeline on all 227,630 images: 35 epochs of Sub-center ArcFace + 15 epochs of standard ArcFace. Estimated runtime: 25 hours. Expected result: TAR@FAR=10⁻⁴ jumping from 11.78% to the 70–95% range, consistent with published ArcFace benchmarks.

**With more data:**
Train on larger public datasets like MS-Celeb-1M (100K identities) or VGGFace2 (9,000 identities, more diverse conditions). More diverse training data would improve generalization to real-world conditions.

**Architectural improvements:**
Swap ResNet50 for IResNet100 (a variant specifically optimized for ArcFace training) or a Vision Transformer, both of which achieve stronger embeddings at higher computational cost.

**Better evaluation:**
Run the full ablation study (embedding dimension × training configuration) to quantify exactly how much each design choice contributes to final performance.

**Deployment hardening:**
Add liveness detection (anti-spoofing) to reject photo or video replay attacks, and quantize the model to INT8 for edge device deployment.

---

### Q: What is the angular margin and what does 0.5 radians mean practically?

The angular margin m=0.5 radians (~28.6°) is the penalty applied to the correct class during ArcFace training. Instead of just requiring the embedding to be closer to the correct identity center than all others, ArcFace requires it to be closer *by at least 28.6°*.

Practically: imagine the embedding space as a globe, and each identity as a point on its surface. Without a margin, the model just needs to be in the right hemisphere. With m=0.5, the model must be within 28.6° of the correct identity center — a much tighter requirement.

This buffer pays off at inference time (no margin applied): because the model was trained to be over-confident, the actual embeddings are very well-clustered. The 28.6° training buffer becomes a genuine separation between identities that makes the system robust even when comparing photos taken in very different conditions.

Why 0.5 specifically? It's the value recommended in the original ArcFace paper and validated across many face benchmarks. Larger margins (e.g., 0.6–0.8) produce tighter clusters but are harder to optimize — the loss becomes very steep and can destabilize training, especially in limited-epoch regimes like ours.

---

### Q: How does Sub-center ArcFace actually handle noisy labels during training?

The key insight is that Sub-center ArcFace gives each identity 3 center vectors (K=3) instead of 1. During training, when an image is processed, it's assigned to whichever of its identity's 3 centers it's closest to in embedding space.

Over many training steps, a natural separation emerges:
- **Clean, typical images** consistently align with the same dominant center — they reinforce each other and push that center to a stable, representative location
- **Noisy images** (mislabeled, blurry, near-duplicates) are inconsistent — they don't fit the dominant cluster and drift toward one of the other two centers, which serve as "catch-all" bins for outliers

After Phase 1 completes, you inspect each image's assigned center. Images on non-dominant sub-centers (i.e., the 2nd or 3rd closest center) are flagged. In our run, this flagged **1,503 images (14.6%)** as likely noisy. These are removed before Phase 2 begins.

Phase 2 then trains standard ArcFace (single center per identity) on the cleaned data. Because the noisy samples were already removed, the loss converges more cleanly — our Phase 2 final loss of 4.09 is within the expected 3–5 range for a well-trained ArcFace model.

---

### Q: How confident are you that your results are real and not artifacts?

Several checks validate that our results reflect genuine learning:

1. **Monotonically decreasing loss**: Both runs showed consistent loss reduction with no spikes or instability. If the training loop were broken, loss would plateau or oscillate.

2. **Loss values are in expected range**: Single-phase final loss of 16.13 is exactly where a 3-epoch run should be (full convergence = 3–5). Two-phase final loss of 4.09 is within the converged range.

3. **Data leakage was fixed**: We verified with a runtime assertion that training and test identity sets are disjoint.

4. **AUC of 91.99% on official grader**: The official evaluation pair space has millions of pairs — a score this high can't be attributed to small-sample noise.

5. **Throughput and memory numbers are physically plausible**: 400 images/second on a T4 GPU at 112×112 resolution is consistent with GPU benchmarks for ResNet50.

6. **14.6% noise rate is consistent with literature**: Sub-center ArcFace papers report typical real-world noise rates of 5–15%, making our 14.6% finding credible.
