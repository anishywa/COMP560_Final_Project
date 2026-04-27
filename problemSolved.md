# The Problem We Solved

## The Core Question

Given two photographs of faces, **are they the same person?**

This is called **1:1 face verification** — not "who is this?" (identification), but "are these two the same person?" (verification). It's the technology behind unlocking your phone with your face, or a border agent confirming your passport photo matches your face.

## The Specific Task

We were given:
- A **training set** of 227,630 face images labeled by identity (12,115 unique people)
- A **test set** of face images organized into *templates* — a template is a collection of photos of one person from a single enrollment event

The task had two parts:

1. **Train a model** that maps any face image to a compact numerical vector (an *embedding*) such that two photos of the same person produce similar vectors, and two photos of different people produce dissimilar vectors.

2. **Score ~8 million template pairs** — for each pair, output a number between -1 and 1 representing how confident the system is that both templates belong to the same person.

## What "Solving" It Means

The system is evaluated on **TAR@FAR** — True Accept Rate at a given False Accept Rate.

Imagine a security door:
- You set the door to be extremely strict: it will mistakenly let a stranger through no more than **1 in every 10,000 attempts** (FAR = 0.01%)
- At that strictness, how many *real authorized people* does it correctly let through?

That percentage is the TAR@FAR=1e-4 score. A higher number means the system is both accurate and trustworthy at a high-security threshold.

## Why This Is Hard

The challenge is not just matching near-identical photos. The model must handle:
- **Lighting variation** — the same person in sunlight vs. indoors
- **Aging** — photos taken years apart
- **Expression** — neutral vs. smiling vs. looking away
- **Occlusion** — glasses, hats, partial shadows
- **Image quality** — high-resolution vs. blurry or compressed

A good embedding must capture "what makes this person *this person*" while ignoring all of those surface-level variations.

## What We Built

We trained a **ResNet50 neural network** with an **ArcFace loss function** to produce 512-dimensional face embeddings. Given any face image, the model outputs a 512-number vector. To compare two templates:

1. Encode every image in each template to a 512-d vector
2. Aggregate the per-image vectors into one representative vector per template
3. Compute the cosine similarity between the two template vectors
4. Use that score to decide: same person or not?

## The Result

After 3 epochs of training (limited by GPU compute time), our system achieved:

| Threshold | Result |
|---|---|
| TAR @ FAR = 1e-4 | **11.78%** |
| TAR @ FAR = 1e-5 | 4.62% |

An untrained model scores 0%. Our 11.78% at the strictest threshold demonstrates that the model learned real, meaningful identity structure — it just needed more training time to fully converge.

## In Plain Terms

We taught a neural network to recognize whether two photos show the same face — not by memorizing people, but by learning a universal notion of facial identity that generalizes to people it has never seen before. The output of that system is a similarity score that can be thresholded to make a yes/no decision at any desired level of strictness.

---

## Future Applications

The core capability this research develops — a model that maps any face to a compact, identity-preserving vector — is a building block for a wide range of real-world systems.

### Security & Access Control
The most direct application. A fully converged version of this model (70–90% TAR@FAR=1e-4) could power:
- **Border control and airport e-gates** — verify a traveler's live face against their passport photo without a human agent
- **Physical access control** — replace keycards with face-based entry at secure facilities
- **Device unlock** — the same verification pipeline used in smartphone face unlock, but trained on a larger, more diverse dataset for robustness

The TAR@FAR metric was designed specifically for these use cases, where the cost of a false accept (letting an impostor through) must be tightly controlled.

### Fraud Prevention & Identity Verification
- **KYC (Know Your Customer) onboarding** — banks and fintech apps can verify that a user's selfie matches their government ID photo at account creation, without manual review
- **Document fraud detection** — flag cases where the same face appears under multiple identities, or where a submitted photo has been reused across applications

### Deduplication at Scale
The embedding approach scales naturally to large databases:
- **Deduplicating photo collections** — group duplicate or near-duplicate images of the same person across millions of records
- **Cross-database matching** — link records across separate datasets (e.g., two government databases that were never joined) by comparing face embeddings rather than names or IDs

### Medical & Research Applications
- **Rare disease research** — some genetic disorders produce distinctive facial features; face embeddings have been used to assist in diagnosis by clustering patients with similar phenotypes
- **Longitudinal identity tracking** — link the same patient across medical records taken years apart, even when names or IDs differ, by using face embeddings as a stable identifier

### Extending the Architecture
The embedding model itself is reusable beyond verification:
- **Facial attribute prediction** — fine-tune the ResNet50 backbone for age estimation, emotion recognition, or demographic analysis, leveraging the rich facial representations already learned
- **Few-shot learning** — because embeddings generalize to unseen identities, the same model can recognize a new person from just 1–2 reference photos, without retraining
- **Anti-spoofing integration** — pair this verification model with a liveness detection model (detecting printed photos or video replays) to build a complete, production-hardened biometric system
