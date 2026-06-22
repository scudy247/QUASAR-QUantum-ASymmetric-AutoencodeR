"""Post-sweep plots for the Conv1D-encoder autoencoder.

Loads the trained halves for each N, recomputes held-out test MSE, and produces:
  conv_mse_vs_N.png            test MSE vs N (one summary curve)
  conv_reconstruction_all_N.png  original vs reconstructed, rows = N, cols = test samples

RUN (after the training sweep has saved trained_*_N{N}.pt for all N):
  cd OurFramework/qic
  ../.venv/bin/python plot_sweep.py
"""

import warnings
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_experiment import EdgeEncoder, quantum_decoder, load_d256, D

warnings.filterwarnings("ignore")

N_VALUES = [2, 4, 6, 8, 10]
SAMPLES = [0, 1, 2]          # which test windows to visualize


def main():
    _, Xte = load_d256()
    mse, recon = {}, {}
    for N in N_VALUES:
        enc = EdgeEncoder(N); enc.load_state_dict(torch.load(f"trained_encoder_N{N}.pt")); enc.eval()
        dec = quantum_decoder(N); dec.load_state_dict(torch.load(f"trained_decoder_N{N}.pt")); dec.eval()
        with torch.no_grad():
            xhat = dec(enc(Xte))
            mse[N] = float(nn.MSELoss()(xhat, Xte))
            recon[N] = xhat
        print(f"N={N:>2} ({D//N}x): test MSE {mse[N]:.4f}")

    # ---- Plot 1: MSE vs N ----
    ys = [mse[N] for N in N_VALUES]
    plt.figure(figsize=(7, 4.5))
    plt.plot(N_VALUES, ys, "o-", color="#1f77b4")
    for N, y in zip(N_VALUES, ys):
        plt.annotate(f"{y:.4f}", (N, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)
    plt.xticks(N_VALUES, [f"{N}\n({D//N}x)" for N in N_VALUES])
    plt.xlabel("N  (qubits / latent size  —  compression D/N below)")
    plt.ylabel("test reconstruction MSE (lower is better)")
    plt.title("Conv1D edge encoder: reconstruction MSE vs N (HAR, D=256, 60 epochs)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("conv_mse_vs_N.png", dpi=130)
    print("saved -> conv_mse_vs_N.png")

    # ---- Plot 2: original vs reconstructed, rows = N, cols = samples ----
    nr, nc = len(N_VALUES), len(SAMPLES)
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 1.6 * nr), sharex=True)
    for r, N in enumerate(N_VALUES):
        for c, s in enumerate(SAMPLES):
            ax = axes[r, c]
            ax.plot(Xte[s].numpy(), lw=1.0, color="#1f77b4", label="original")
            ax.plot(recon[N][s].numpy(), lw=1.0, color="#ff7f0e", alpha=0.85, label="reconstructed")
            if c == 0:
                ax.set_ylabel(f"N={N}\n{D//N}x  MSE {mse[N]:.4f}", fontsize=8)
            if r == 0:
                ax.set_title(f"test sample {s}", fontsize=9)
    axes[0, -1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Original (blue) vs reconstructed (orange) — rows: more qubits N (less compression)",
                 fontsize=11)
    axes[-1, 0].set_xlabel("feature index (0-127 acc_x, 128-255 acc_y)")
    plt.tight_layout()
    plt.savefig("conv_reconstruction_all_N.png", dpi=130)
    print("saved -> conv_reconstruction_all_N.png")


if __name__ == "__main__":
    main()
