"""
data_utils.py
-------------
Shared utilities for loading the QCD quark/gluon jet image dataset.

Supports two formats automatically:
  1. HDF5  (.hdf5 / .h5)  — preferred, much faster
       X_jets: (N, 125, 125, 3) channels-LAST  → transposed to (N, 3, 125, 125)
  2. Parquet (.parquet)   — fallback
       X_jets: list<list<list<double>>>         → (N, 3, 125, 125)

Label: y=0 quark, y=1 gluon
"""

import os
import glob
import numpy as np
import h5py
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ─────────────────────────────────────────────────────────────
# Primary loader — auto-detects HDF5 or parquet
# ─────────────────────────────────────────────────────────────

def load_dataset(data_dir: str, max_events: int = None):
    """
    Load the quark/gluon dataset. Prefers HDF5 (faster); falls back to parquet.

    HDF5 search order:
      1. data_dir  (e.g. project/data/)
      2. parent of data_dir (e.g. project/../)  — handles the common case where
         the HDF5 lives one level outside the project data folder.

    Returns
    -------
    X : np.ndarray  shape (N, 3, 125, 125)  float32   [channels-first]
    y : np.ndarray  shape (N,)              int64
    """
    # ── Try HDF5 — search data_dir then its parent ────────────
    parent_dir = os.path.dirname(os.path.abspath(data_dir))
    h5_files = sorted(
        glob.glob(os.path.join(data_dir,  "*.hdf5")) +
        glob.glob(os.path.join(data_dir,  "*.h5"))   +
        glob.glob(os.path.join(parent_dir, "*.hdf5")) +
        glob.glob(os.path.join(parent_dir, "*.h5"))
    )
    if h5_files:
        fpath = h5_files[0]
        print(f"  Loading HDF5: {fpath} …", flush=True)
        with h5py.File(fpath, "r") as f:
            if max_events:
                X = f["X_jets"][:max_events].astype(np.float32)  # (N,125,125,3)
                y = f["y"][:max_events].astype(np.int64)
            else:
                X = f["X_jets"][:].astype(np.float32)
                y = f["y"][:].astype(np.int64)

        # HDF5 stores channels-LAST → convert to channels-FIRST
        X = np.transpose(X, (0, 3, 1, 2))   # (N,125,125,3) → (N,3,125,125)
        print(f"  Loaded {len(X):,} events — X shape {X.shape}, "
              f"quark={np.sum(y==0):,}, gluon={np.sum(y==1):,}")
        return X, y

    # ── Fallback: parquet ─────────────────────────────────────
    return _load_parquet_files(data_dir, max_events)


def _load_parquet_files(data_dir: str, max_events: int = None):
    """Internal: load from parquet files (slower, fallback only)."""
    pattern = os.path.join(data_dir, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No HDF5 or parquet files found in {data_dir}"
        )

    X_list, y_list = [], []
    for fpath in files:
        print(f"  Loading {os.path.basename(fpath)} …", flush=True)
        table = pq.read_table(fpath, columns=["X_jets", "y"])
        n = len(table)
        if max_events is not None:
            remaining = max_events - sum(len(x) for x in X_list)
            if remaining <= 0:
                break
            table = table.slice(0, min(n, remaining))

        x_col = table.column("X_jets").to_pylist()
        X_arr = np.array(x_col, dtype=np.float32)   # (batch, 3, 125, 125)
        y_arr = np.array(table.column("y").to_pydict()["y"], dtype=np.int64)
        X_list.append(X_arr)
        y_list.append(y_arr)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print(f"  Loaded {len(X):,} events — X shape {X.shape}")
    return X, y


# Keep old name as alias for backward compatibility
load_parquet_files = load_dataset


# ─────────────────────────────────────────────────────────────
# PyTorch Dataset wrappers
# ─────────────────────────────────────────────────────────────

class JetImageDataset(Dataset):
    """Plain (image, label) dataset for the CAE and supervised tasks."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Normalize each channel to [0, 1] using global max
        max_val = X.max()
        self.X = torch.from_numpy(X / (max_val + 1e-8))   # (N, 3, 125, 125)
        self.y = torch.from_numpy(y)                        # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────
# Train / val / test split helper
# ─────────────────────────────────────────────────────────────

def make_splits(dataset: Dataset, train_frac=0.70, val_frac=0.15, seed=42):
    """70 / 15 / 15 deterministic split."""
    n = len(dataset)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    n_test  = n - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


# ─────────────────────────────────────────────────────────────
# Image → point cloud conversion  (used in Task 2 & 3)
# ─────────────────────────────────────────────────────────────

def image_to_pointcloud(img: np.ndarray, threshold: float = 0.0):
    """
    Convert a single (3, 125, 125) jet image to a point cloud.

    Strategy: treat each pixel position (η_i, φ_j) where at least one
    channel has energy > threshold as a "particle" (graph node).

    Returns
    -------
    points : np.ndarray  shape (M, 5)
        Columns: [η_norm, φ_norm, E_track, E_ECAL, E_HCAL]
        η_norm, φ_norm ∈ [-1, 1] (pixel index normalised to [-1,1])
    """
    assert img.shape == (3, 125, 125), f"Expected (3,125,125), got {img.shape}"
    ch_track, ch_ecal, ch_hcal = img[0], img[1], img[2]

    # Mask: any channel with energy above threshold
    mask = (np.abs(ch_track) + np.abs(ch_ecal) + np.abs(ch_hcal)) > threshold
    eta_idx, phi_idx = np.where(mask)

    if len(eta_idx) == 0:
        # Return a single zero-padding node to avoid empty graphs
        return np.zeros((1, 5), dtype=np.float32)

    # Normalise pixel coordinates to [-1, 1]
    eta_norm = (eta_idx / 124.0) * 2.0 - 1.0   # 124 = 125 - 1
    phi_norm = (phi_idx / 124.0) * 2.0 - 1.0

    e_track = ch_track[eta_idx, phi_idx]
    e_ecal  = ch_ecal [eta_idx, phi_idx]
    e_hcal  = ch_hcal [eta_idx, phi_idx]

    points = np.column_stack([eta_norm, phi_norm, e_track, e_ecal, e_hcal]).astype(np.float32)
    return points
