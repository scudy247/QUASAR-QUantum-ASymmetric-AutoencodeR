"""Data loader for the Hybrid Quantum-Classical Asymmetric Autoencoder experiment.

The experiment (run_experiment.py) calls `load_dataset("har")`, which returns the real
UCI HAR inertial signals as windows of shape (num, 9 channels, 128 samples), with labels
in {0..5}. A synthetic fallback (`make_synthetic` + `train_test_split`) is kept so
`load_dataset("synthetic")` works without the dataset on disk; run_experiment.py uses HAR.

Functions:
  make_synthetic     -> toy sinusoid windows (single channel), for a quick smoke test
  load_uci_har       -> real UCI HAR inertial signals, cached to .npz after first load
  load_dataset(name) -> dispatch: "har" or "synthetic"; returns (Xtr, ytr, Xte, yte)
"""

import numpy as np


def make_synthetic(n_per_class=300, N=128, n_classes=4, noise=0.4, seed=0):
    """Each window is a noisy sinusoid; the class sets the base frequency.
    Returns X (num, N) float, y (num,) int."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N, endpoint=False)
    X, y = [], []
    for c in range(n_classes):
        freq = 2 * (c + 1)
        for _ in range(n_per_class):
            phase = rng.uniform(-0.4, 0.4)  # limited jitter -> learnable in time domain
            amp = rng.uniform(0.7, 1.3)
            sig = amp * np.sin(2 * np.pi * freq * t + phase) + noise * rng.standard_normal(N)
            X.append(sig)
            y.append(c)
    X = np.asarray(X)
    y = np.asarray(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def train_test_split(X, y, test_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], y[tr], X[te], y[te]


HAR_CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def load_uci_har(base="datasets/UCI HAR Dataset", cache="datasets/har_cache.npz"):
    """UCI HAR raw 'Inertial Signals': 9 channels x 128 samples (128 = 2**7), 6 classes.
    Returns Xtr, ytr, Xte, yte with X shape (num, C=9, N=128) and y in {0..5}.
    Caches to .npz on first load (loadtxt is slow)."""
    import os

    if os.path.exists(cache):
        d = np.load(cache)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]

    def load_split(split):
        sigs = [np.loadtxt(f"{base}/{split}/Inertial Signals/{ch}_{split}.txt")
                for ch in HAR_CHANNELS]
        X = np.stack(sigs, axis=1)  # (num, C, N)
        y = np.loadtxt(f"{base}/{split}/y_{split}.txt").astype(int) - 1  # 0-indexed
        return X, y

    Xtr, ytr = load_split("train")
    Xte, yte = load_split("test")
    np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte


def load_dataset(name):
    """Returns Xtr, ytr, Xte, yte. X is (num, N) [synthetic] or (num, C, N) [har]."""
    if name == "har":
        return load_uci_har()
    X, y = make_synthetic(n_per_class=250, N=128, n_classes=6, noise=1.4, seed=0)
    return train_test_split(X, y, test_frac=0.25, seed=1)
