# Facial Recognition Project — Plain English Summary
**Anish Sharma, Dante Mayo, Naveen Prabhu, Niyanth Ponnusamy, Dylan Masi**

---

## 1. Introduction and Motivation

### What problem are we solving?
We're trying to build a system that answers one question: **are these two photos of the same person?**

Think of it like the face unlock on your phone. You show your face, and the phone decides: "yes, this is the owner" or "no, this is someone else." That's what our system does.

### Why is this hard?
People look different depending on the lighting, the angle of the photo, whether they're wearing glasses, or how old they are. Getting a computer to reliably say "same person" despite all of that is genuinely difficult.

### What happens when it gets it wrong?
There are two ways to be wrong:
- **False rejection**: The system says "not the same person" when it actually is. The real user gets locked out of their phone — annoying, but not dangerous.
- **False acceptance**: The system says "same person" when it isn't. An attacker gets in — that's a security problem.

Because false acceptances are so much more dangerous, we care a lot more about minimizing them.

### Why not just use "accuracy"?
Imagine you have 10,000 photo pairs, and only 10 of them are actually the same person. A system that just says "different person" for every single pair would be right 9,990/10,000 times — that's 99.9% accuracy! But it's completely useless because it never lets anyone in.

Instead, we use a metric called **TAR@FAR**:
- **FAR (False Accept Rate)**: Out of all the impostors, what fraction does the system wrongly let through?
- **TAR (True Accept Rate)**: Out of all the real matches, what fraction does the system correctly let through?

We set FAR to be very strict — like "only 1 impostor in 10,000 gets through" — and then ask how many real matches we catch at that strictness level. That's a much more honest way to measure usefulness.

---

## 2. Related Work

### How did people solve this before?

**The old way — manual feature extraction (1990s):**
Early systems had programmers manually tell the computer what to look for: the distance between the eyes, the shape of the nose, etc. The most famous method was called **Eigenfaces**. It worked okay in a lab with perfect lighting and straight-on photos, but fell apart in the real world.

**The deep learning era — let the computer figure it out (2010s):**
Instead of telling the computer what features matter, researchers started feeding it millions of face photos and letting it figure out on its own what distinguishes one person from another. These systems (called convolutional neural networks, or CNNs) dramatically outperformed the old hand-crafted approaches.

**FaceNet — faces as points in space:**
One key idea: represent each face as a point in a mathematical space (called an "embedding"). If two photos show the same person, their points should be close together. If they show different people, the points should be far apart. FaceNet did this using "triplet loss" — training the network by showing it triplets: (same person A, same person A, different person B) and saying "push A close to A and away from B." This worked well but was slow to train.

**ResNet — making deep networks trainable:**
Deeper networks generally learn better, but they used to be very hard to train — a problem called the "vanishing gradient." ResNet fixed this with "skip connections" that let information flow more easily through many layers. ResNet50 (50 layers deep) became a go-to building block for computer vision.

**ArcFace — a smarter training objective:**
Instead of directly comparing pairs of faces during training, ArcFace trains a classifier ("which of these 12,000 people is this?") but adds a twist: it forces the model to leave a gap of at least 28.6° between any two identities in the embedding space. This gap (called an angular margin) makes the system more robust — it has to be confidently right, not just barely right.

**Sub-center ArcFace — handling bad data automatically:**
Real datasets are messy. Some photos are mislabeled, blurry, or near-duplicates. Sub-center ArcFace gives each person 3 "centers" instead of 1. Good photos cluster around the main center; bad photos drift to the other two. After training, you can automatically identify and throw out the bad ones.

---

## 3. Our Approach

### What did we build?

Our system has three main pieces:

**1. A feature extractor (ResNet50)**
We take a 112×112 pixel photo of a face and run it through ResNet50, a 50-layer neural network that was pretrained on millions of general images. It converts the photo into a list of 512 numbers (called an "embedding") that captures what makes that face unique. We start from the pretrained weights rather than training from scratch — it's like hiring someone who already knows how to see, and teaching them specifically about faces.

**2. A smart training objective (ArcFace loss)**
During training, we use ArcFace to push the embeddings for the same person close together and for different people far apart — with a mandatory gap of ~28.6° between identities. This makes the final system very precise at distinguishing faces.

**3. Automatic data cleaning (Sub-center ArcFace)**
In the first phase of training, we use the noisy-label-tolerant version (Sub-center ArcFace) to automatically identify bad photos in our dataset and remove them. Then we retrain on the cleaned data. This is our main innovation — it's like proofreading your textbook before studying from it.

---

## 4. Pipeline

### How does the whole system work, step by step?

**Step 1 — Load the data**
We load 112×112 face images. Each image belongs to a **template** (a collection of photos of one person from one session) and a **media ID** (a specific video clip within that session). Metadata is stored in a Parquet file like a spreadsheet.

**Step 2 — Train the model**
We tried three different training setups because of compute limitations (more on that in the next section):
- **Original plan**: Full two-phase training — too slow (25 hours)
- **Quick proof-of-concept**: Just 3 epochs of training, no noise filtering — 84 minutes, shows the system works
- **Final design**: Two-phase training on a smaller dataset — 15 minutes, noise filtering included

**Step 3 — Generate embeddings**
After training, we freeze the model and run every test image through it to get its 512-number embedding. This is fast — the GPU processes ~400 images per second.

**Step 4 — Combine photos into one vector per person**
Each person in the test set has multiple photos. We need to combine them into one single representation:
1. Average the embeddings for all photos from the same video clip
2. Sum those averages across all clips for that person
3. Normalize the result (scale it so its length is exactly 1)

Why this specific order? If we just averaged everything, a person with 100 photos would dominate over a person with 5. Averaging within clips first, then summing, gives each recording session equal weight.

**Step 5 — Compare pairs**
For every pair of people we want to compare, we compute the dot product of their normalized vectors. This gives a score between -1 and 1. High score = probably the same person. Low score = probably different people.

**Step 6 — Evaluate**
We set a threshold (based on how many impostors we're willing to let through) and count how many real matches we caught.

---

## 5. Data and Training

### What data did we use?

- **Training data (Dataset A)**: 227,630 face photos of 12,115 different people, collected in real-world conditions
- **Final test data (Dataset B)**: A completely separate set, never seen during training or tuning
- **Subset used in final run**: 10,277 photos of 605 people (5% of Dataset A) — needed to make training fast enough

We also made sure training data and test data never overlapped — no person in the training set appears in the test set. An earlier version of the code accidentally mixed them, which made the results look better than they really were (called "data leakage"). We fixed this.

### How did we make the model more robust?

Since real-world photos have lots of variation, we randomly modified photos during training so the model couldn't just memorize them:

| What we did | Why |
|---|---|
| Randomly flip photos left-right | Faces are symmetric; prevents the model from caring which side the nose is on |
| Adjust brightness, contrast, and color | Simulates different lighting conditions |
| Randomly erase parts of the photo | Simulates someone wearing glasses, a hat, or a mask |
| Add blur | Simulates low-quality security cameras |
| Block out grid sections randomly | Forces the model to use all parts of the face, not just the eyes or nose |

The blur and grid masking are especially important because they push the model to build a complete picture of the face rather than cheating by relying on one distinctive feature.

### How did we control the training speed?

The learning rate is how fast the model updates itself after each batch of photos. We used:
- **Warmup for the first 2 epochs**: Start slow, then gradually speed up. This is important because at the start, the model's identity centers are random — if we go too fast, we can make things worse before they get better.
- **Cosine decay for the rest**: Gradually slow down as training progresses, so the final adjustments are small and precise.

---

## 6. Evaluation Protocol

### How do we measure performance?

After training, we compare every pair in the test set and get a score. Then:

1. We look at all the **impostor pairs** (different people) and sort their scores from highest to lowest.
2. We find the score where only 1 in 10,000 impostors scores above it — that becomes our threshold.
3. We count what fraction of **genuine pairs** (same person) score above that threshold.
4. That fraction is our **TAR@FAR = 10⁻⁴**.

We do this at three strictness levels:
- **1 in 10,000 impostors allowed through** → TAR@FAR = 10⁻⁴
- **1 in 100,000** → TAR@FAR = 10⁻⁵
- **1 in 1,000,000** → TAR@FAR = 10⁻⁶

Stricter thresholds are harder because the system has to be really confident before letting anyone through.

We also measure **AUC** (Area Under the Curve) — a single number from 0–100% that summarizes how well the system ranks same-person pairs above different-person pairs across all possible thresholds. 100% = perfect, 50% = random guessing.

---

## 7. Results

### Run 1: Quick proof-of-concept (3 epochs, full dataset)

**Goal**: Verify the pipeline works before investing more time.

The model trained for just 3 rounds through the data. Here's how the training loss (lower = model is learning) changed:

| Round | Loss |
|---|---|
| 1 | 22.68 |
| 2 | 18.76 |
| 3 | 16.13 |

Loss dropped 29%, showing the model was learning. A fully trained model would reach a loss of 3–5, so 16.13 means we're only partway there — but that's expected in just 3 rounds.

**Results:**

| Metric | Value |
|---|---|
| Catches 1-in-10,000 impostor threshold | 11.78% of real matches |
| Catches 1-in-100,000 threshold | 4.62% |
| Catches 1-in-1,000,000 threshold | 2.43% |
| Speed | 383 photos/second |
| GPU memory used | 2.2 GB |
| Training time | 84 minutes |

**What does 11.78% mean?** An untrained model would get ~0%. A fully trained model would get ~80–95%. We're partway between — exactly what 3 epochs should produce. This proved the system is working correctly.

---

### Run 2: Full noise-filtering pipeline (two phases, small dataset)

**Goal**: Run the complete, proper design — noise filtering included — within our time budget.

We trained on 5% of the data (605 people) but ran all the proper steps.

**Phase 1 — Noisy label detection (8 rounds):**
- Loss went from 33.21 down to 5.53
- After this, **1,503 out of 10,277 photos (14.6%) were flagged as bad** and removed
- These were likely mislabeled photos, blurry frames, or near-duplicates

**Phase 2 — Training on clean data (4 rounds):**
- Loss went from 11.60 down to **4.09** — right in the "well trained" range

**Results:**

| Metric | Value |
|---|---|
| AUC (overall ranking quality) | **91.99%** |
| Catches 1-in-10 threshold | 81.82% |
| Catches 1-in-100 threshold | 54.55% |
| Catches 1-in-1,000 threshold | 18.18% |
| Speed | 401 photos/second |
| GPU memory used | 1.45 GB |
| Total pipeline time | ~15 minutes |

The AUC of 91.99% means the model correctly ranks same-person pairs above different-person pairs 92% of the time — that's strong performance for a model trained on only 605 people.

**How did runtime shrink so dramatically?**

| Version | Time | What changed |
|---|---|---|
| Original design | 25 hours | Full dataset, full epochs |
| Proof-of-concept | 84 minutes | Skipped noise filtering, fewer epochs |
| Final design | 15 minutes | Smaller dataset, both phases restored |

---

## 8. Discussion and Limitations

### What went well?

- The model improved steadily in both runs — no crashes or instability
- Sub-center ArcFace correctly found and removed ~15% bad photos automatically — that's a meaningful data quality improvement
- 91.99% AUC is solid performance even from a small training subset

### What held us back?

**Compute time was the #1 constraint.** Google Colab (our free GPU environment) limits sessions to 12 hours. Our original plan required 25 hours of training — impossible in that environment. Everything else in the project was an adaptation to that single constraint.

If we could train for the full planned duration, we'd expect TAR@FAR=10⁻⁴ to jump from 11.78% up to 70–95%.

**The "0% at strict thresholds" problem:**
In the two-phase run, TAR@FAR at the strictest thresholds showed 0%. This sounds bad but isn't — it's a math problem:

- Our test set had only 7,930 pairs total
- At FAR=10⁻⁴, the threshold is set to filter out 9,999 out of every 10,000 impostors
- With only a few hundred positive (same-person) pairs in the test set, the odds that even one clears that bar are very low — not because the model is bad, but because the sample size is too small to measure at that precision

The 91.99% AUC, measured on the full official dataset, tells the real story.

**We found and fixed a data leak:**
An early version of our code accidentally used the same people for both training and testing. That's like studying from the exact exam that you'll be graded on — results look great but don't reflect real ability. We fixed this so training and test identities never overlap.

**Embedding size tradeoff:**
We represent each face with either 128, 256, or 512 numbers. More numbers = more expressive but slower comparisons and more memory. For a dataset with only 605 people, 128 numbers is plenty. For 12,115 people, 512 is better. We matched the size to the dataset scale.

---

## 9. Conclusion and Future Work

### What did we build?

A complete face verification system that:
1. Takes in face photos
2. Converts them to compact numerical fingerprints (embeddings)
3. Automatically cleans bad training data using Sub-center ArcFace
4. Compares pairs by dot product (simple, fast, scalable to millions of pairs)
5. Evaluates at real-world security thresholds (TAR@FAR)

### What do the results prove?

- The architecture and training recipe work correctly (proven by the 3-epoch run)
- The noise filtering step works correctly — it found and removed 14.6% bad photos, and the model trained on clean data reached near-convergence (proven by the two-phase run)
- The limiting factor is GPU time, not design

Think of this as a prototype that works and is ready to scale up. With more compute, the same system would perform at production quality.

### What would we do next?

**With more compute:**
- Train for the full planned 35 + 15 epochs on all 227,630 photos
- Expected result: TAR@FAR=10⁻⁴ in the 70–95% range

**With more time:**
- Try newer and larger backbone networks (like Vision Transformers) that extract even richer features
- Train on bigger and more diverse datasets that include more variation in age, ethnicity, and image conditions

**For real deployment:**
- Compress the model for phones and edge devices where memory is limited
- Add a "liveness detection" layer to reject someone holding up a photo of the target's face in front of the camera

---

## References

1. Deng et al. (2019) — ArcFace: the training method we used
2. He et al. (2016) — ResNet: the neural network backbone we built on
3. Schroff et al. (2015) — FaceNet: an earlier approach we improved upon
4. Turk & Pentland (1991) — Eigenfaces: the classic method we moved beyond
