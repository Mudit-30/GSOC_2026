# GENIE GSoC 2026 — Deep Graph Anomaly Detection with Contrastive Learning

**ML4SCI Evaluation Submission** | Author: Mudit

---

## Project Overview

This repository solves all three ML4SCI GENIE GSoC 2026 evaluation tasks, building a complete pipeline for **quark/gluon jet classification and anomaly detection** using graph-based deep learning.

**Dataset:** QCD quark/gluon jet events — 3-channel 125×125 images (Tracks, ECAL, HCAL), N=139,306 events.

---

## Setup

```bash
pip install -r requirements.txt
```

> **PyTorch Geometric:** install `torch-scatter` and `torch-sparse` matching your CUDA version from https://data.pyg.org/whl/

**Alternatively, run everything on free GPU via the Colab notebook:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](GENIE_GSoC2026_Colab.ipynb)

---

## Task Solutions

### Common Task 1 — Convolutional Autoencoder (CAE)

```bash
python src/task1_cae.py
```

Trains a symmetric convolutional autoencoder on raw 3-channel jet images.

**Architecture:**
```
Input (N, 3, 125, 125)
│
├─ Encoder: 3 × [Conv2d → BatchNorm → ReLU → MaxPool2d]
│           Channels: 3 → 16 → 32 → 64   Spatial: 125 → 62 → 31 → 15
│
├─ Bottleneck: (N, 64, 15, 15) latent representation
│
└─ Decoder: 3 × [ConvTranspose2d → BatchNorm → ReLU] + Upsample + Conv2d → Sigmoid
            Restores: (N, 3, 125, 125)
```

- **Loss:** MSE  |  **Optimizer:** Adam + cosine LR schedule  |  **Epochs:** 30
- **Output:** `outputs/task1_reconstructions.png` — 8 events × 6 channels (orig | recon)

---

### Common Task 2 — GNN Jet Classifier (GAT)

```bash
python src/task2_gnn.py
```

Converts jet images to graphs and classifies quark vs. gluon jets using a **Graph Attention Network**.

**Image → Graph pipeline:**
1. Extract non-zero pixels as particle nodes with features `[η_norm, φ_norm, E_track, E_ECAL, E_HCAL]`
2. Build a **k-NN graph (k=8)** in (η, φ) coordinate space

**Why k-NN over radius graphs?**
Jet images are sparse with variable particle multiplicity. Radius graphs produce isolated nodes in low-multiplicity events, causing degenerate global pooling. k-NN guarantees every node has exactly 8 neighbours, maintaining well-conditioned message-passing gradients across all events.

**Why GAT over GCN?**
Graph Attention Networks learn adaptive, data-driven edge weights rather than fixed normalised adjacency weights. Each jet constituent has varying relevance to its neighbours depending on pT and angular separation — attention heads naturally capture this heterogeneity.

**GAT model:**
```
GATConv(5→32, heads=4) → ELU → BN
GATConv(128→32, heads=4) → ELU → BN
GATConv(128→64, heads=1)
global_{mean,max}_pool → cat → 128-dim
Linear(128→64) → ELU → Dropout(0.3) → Linear(64→1)
```

- **Loss:** BCE with pos_weight  |  **Optimizer:** AdamW  |  **Metric:** ROC-AUC
- **Output:** `outputs/task2_roc.png`

---

### Specific Task 1 — Contrastive Anomaly Detection (GraphCLR)

```bash
python src/task3_contrastive.py
```

Learns jet representations **without labels** via contrastive learning (SimCLR-style), then evaluates both unsupervised anomaly detection and supervised linear-probe classification.

**Framework:**

**Phase A — Unsupervised Pretraining:**
1. **Augmentations:** Each jet graph gets two stochastically augmented views:
   - Node drop (10%) — removes nodes + their edges
   - Edge drop (20%) — removes edges
   - Feature noise (σ=0.05) — Gaussian noise on energy channels
2. **Encoder:** Same GAT × 3 + global {mean, max} pool backbone as Task 2
3. **Projection head:** 2-layer MLP (`GNN_DIM → 128`), L2-normalised output
4. **NT-Xent loss** (τ = 0.5):
   ```
   loss = -log[ exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) ]
   ```

**Phase B — Supervised Linear Probe:**
5. Freeze encoder weights
6. Train a single linear layer on top → evaluates representation quality

**Evaluation:**
- **Unsupervised:** ROC-AUC using L2 distance from training centroid (gluon = anomaly)
- **Supervised:** ROC-AUC of linear classifier on frozen embeddings

**Outputs:** `outputs/task3_roc.png` (linear probe) + `outputs/task3_roc_anomaly.png` (unsupervised)

---

## Results

| Task | Mode | Metric | Value |
|------|------|--------|-------|
| Task 1 — CAE | — | Val MSE | *(run to fill)* |
| Task 2 — GAT | Supervised | Test ROC-AUC | *(run to fill)* |
| Task 2 — GAT | Supervised | Test Accuracy | *(run to fill)* |
| Task 3 — GraphCLR | Unsupervised anomaly | Test ROC-AUC | *(run to fill)* |
| Task 3 — GraphCLR | Linear probe | Test ROC-AUC | *(run to fill)* |

---

## Repository Structure

```
genie_evaluation_mudit/
├── data/              # dataset files (not committed — too large)
│   └── *.parquet      # QCD quark/gluon jet images
├── outputs/           # generated plots and model checkpoints
│   ├── task1_reconstructions.png
│   ├── task1_loss_curve.png
│   ├── task2_roc.png
│   ├── task3_roc.png
│   └── task3_roc_anomaly.png
├── src/
│   ├── data_utils.py         # shared data loading & point cloud conversion
│   ├── task1_cae.py          # Common Task 1 — Convolutional Autoencoder
│   ├── task2_gnn.py          # Common Task 2 — GAT Jet Classifier
│   └── task3_contrastive.py  # Specific Task 1 — GraphCLR Contrastive Learning
├── GENIE_GSoC2026_Colab.ipynb  # All-in-one Colab notebook for GPU training
└── requirements.txt
```

---

## Physics Motivation

Jets are collimated sprays of particles produced in high-energy collisions. Distinguishing quark-initiated from gluon-initiated jets is important for new physics searches. Treating jets as **graphs** (particles = nodes, angular proximity = edges) preserves the relational structure lost in 2D image representations.

The contrastive learning framework is particularly promising for anomaly detection: by learning rich jet representations without labels, we can flag events that are "far" from the learned manifold — a crucial capability for beyond-Standard-Model searches where signal topology is unknown.
