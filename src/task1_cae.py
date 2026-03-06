"""
task1_cae.py — Common Task 1
==============================
Convolutional Autoencoder (CAE) for quark/gluon 3-channel jet images.

Architecture
------------
Encoder: 3 × (Conv2d → BatchNorm → ReLU → MaxPool2d)
Decoder: 3 × (ConvTranspose2d → BatchNorm → ReLU) + final Conv2d → Sigmoid

Input/Output shape: (N, 3, 125, 125)

Deliverable
-----------
outputs/task1_reconstructions.png — 8-event side-by-side grid
outputs/task1_loss_curve.png      — training & validation loss
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")          # headless — works without a display
import matplotlib.pyplot as plt

# allow importing from sibling src/ when run as a script
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_parquet_files, JetImageDataset, make_splits

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
OUT_DIR   = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 64
NUM_EPOCHS  = 20
LR          = 1e-3
MAX_EVENTS  = 500   # cap for faster iteration; set None for full dataset


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """
    Symmetric encoder-decoder CAE.

    Encoder path (125 → 62 → 31 → 15):
        Conv(3→16, k=3, p=1) → BN → ReLU → MaxPool(2)
        Conv(16→32, k=3, p=1) → BN → ReLU → MaxPool(2)
        Conv(32→64, k=3, p=1) → BN → ReLU → MaxPool(2)   [spatial: ~15×15]

    Decoder path (15 → 31 → 62 → 125):
        ConvTranspose(64→32, k=2, s=2) → BN → ReLU
        ConvTranspose(32→16, k=2, s=2) → BN → ReLU
        ConvTranspose(16→8,  k=2, s=2) → BN → ReLU
        Conv(8→3, k=3, p=1) → Sigmoid   (restore 3 channels)
    """

    def __init__(self):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────
        self.encoder = nn.Sequential(
            # Block 1: (N,3,125,125) → (N,16,62,62)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 125→62

            # Block 2: (N,16,62,62) → (N,32,31,31)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 62→31

            # Block 3: (N,32,31,31) → (N,64,15,15)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 31→15 (15.5 → floor 15)
        )

        # ── Decoder ──────────────────────────────────────────
        self.decoder = nn.Sequential(
            # Block 1: (N,64,15,15) → (N,32,30,30)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: (N,32,30,30) → (N,16,60,60)
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Block 3: (N,16,60,60) → (N,8,120,120)
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            # Final: upsample (120→125) + restore 3 channels
            nn.Upsample(size=(125, 125), mode="bilinear", align_corners=False),
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (N, 3, 125, 125) → recon: (N, 3, 125, 125)"""
        z = self.encoder(x)
        return self.decoder(z)


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

CH_NAMES = ["Tracks", "ECAL", "HCAL"]

def save_reconstruction_grid(model, dataset, n_samples=8, path=None):
    """
    Saves a grid of originals vs. reconstructions.
    Rows: 8 sample events
    Columns: 3 channels × 2 (original | reconstruction) = 6 cols
    """
    model.eval()
    imgs, _ = zip(*[dataset[i] for i in range(n_samples)])
    imgs = torch.stack(imgs).to(DEVICE)

    with torch.no_grad():
        recons = model(imgs).cpu().numpy()
    imgs = imgs.cpu().numpy()

    fig, axes = plt.subplots(n_samples, 6, figsize=(18, n_samples * 2.2))
    fig.suptitle("CAE Reconstruction — Quark/Gluon Jets", fontsize=14, fontweight="bold")

    for row in range(n_samples):
        for ch in range(3):
            # Original
            ax_o = axes[row, ch]
            ax_o.imshow(imgs[row, ch], cmap="hot", aspect="auto",
                        vmin=0, vmax=imgs[row, ch].max() + 1e-6)
            if row == 0:
                ax_o.set_title(f"Orig {CH_NAMES[ch]}", fontsize=8)
            ax_o.axis("off")

            # Reconstruction
            ax_r = axes[row, ch + 3]
            ax_r.imshow(recons[row, ch], cmap="hot", aspect="auto",
                        vmin=0, vmax=imgs[row, ch].max() + 1e-6)
            if row == 0:
                ax_r.set_title(f"Recon {CH_NAMES[ch]}", fontsize=8)
            ax_r.axis("off")

    plt.tight_layout()
    save_path = path or os.path.join(OUT_DIR, "task1_reconstructions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved reconstruction grid → {save_path}")


def save_loss_curve(train_losses, val_losses, path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="Train MSE", color="#2563EB", linewidth=2)
    ax.plot(val_losses,   label="Val MSE",   color="#DC2626", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("CAE Training Loss Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    save_path = path or os.path.join(OUT_DIR, "task1_loss_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved loss curve → {save_path}")


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        recon = model(imgs)
        loss = criterion(recon, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        recon = model(imgs)
        loss = criterion(recon, imgs)
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    # 1. Load data
    print("\n[1/4] Loading dataset …")
    X, y = load_parquet_files(DATA_DIR, max_events=MAX_EVENTS)

    # 2. Splits & loaders
    print("[2/4] Building data loaders …")
    full_ds = JetImageDataset(X, y)
    train_ds, val_ds, test_ds = make_splits(full_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 3. Train
    print("[3/4] Training CAE …")
    model     = ConvAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss  = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "task1_cae_best.pth"))

        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  train={tr_loss:.5f}  val={val_loss:.5f}")

    # 4. Save artefacts
    print("[4/4] Saving outputs …")
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "task1_cae_best.pth"), map_location=DEVICE))
    save_reconstruction_grid(model, val_ds)
    save_loss_curve(train_losses, val_losses)
    print(f"\nBest val MSE: {best_val:.6f}")


if __name__ == "__main__":
    main()
