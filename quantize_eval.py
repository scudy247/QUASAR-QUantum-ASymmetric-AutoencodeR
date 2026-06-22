"""Latent quantization & rate-distortion study for the hybrid quantum-classical autoencoder.

The transmitted payload is not N floats but N x b BITS. This script trains each model once
(reusing the CNN encoder + decoders from run_experiment.py), then does POST-TRAINING
quantization of the latent z to b bits per value and measures how lossy reconstruction is:

    encode  ->  z in [-1,1]^N  ->  quantize to b bits  ->  decode  ->  x_hat

Outputs:
  * a table of MSE and SNR(dB) per (model, N, bits) with payload = N*b bits;
  * a rate-distortion plot (MSE vs payload bits) comparing the quantum decoder against the
    fair classical baseline -- i.e. does the quantum advantage SURVIVE quantization;
  * an N=8 detail plot (MSE vs bits).

Run (in the tmux 'simulations' session):
  ../.venv/bin/python quantize_eval.py
"""

import warnings
from collections import defaultdict

import numpy as np
import torch

from run_experiment import (EdgeEncoder, quantum_decoder, matched_classical_decoder,
                            load_d256, train, N_TRAIN)

warnings.filterwarnings("ignore")

N_VALUES = [4, 6, 8, 10]          # bottleneck sizes (= qubit counts)
BITS = [8, 4, 3, 2, 1]            # bits per latent value (8 ~ lossless reference)
SEEDS = [0, 1, 2]                 # repeats -> error bars
EPOCHS = 50
DECODERS = {"hybrid": quantum_decoder, "matched": matched_classical_decoder}


def quantize(z, bits):
    """Uniform mid-rise quantizer of z in [-1,1] to `bits` bits (2**bits levels)."""
    levels = 2 ** bits
    zc = torch.clamp(z, -1.0, 1.0)
    return torch.round((zc + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1


def lossiness(xhat, x):
    """Return (MSE, SNR in dB). Higher SNR = less lossy."""
    mse = float(torch.mean((xhat - x) ** 2))
    snr = 10.0 * np.log10(float(torch.mean(x ** 2)) / mse) if mse > 0 else float("inf")
    return mse, snr


def main():
    Xtr_pool, Xte = load_d256()
    print(f"D={Xte.shape[1]}  test {tuple(Xte.shape)}  seeds {SEEDS}  bits {BITS}\n")

    # res[(model, N, bits)] -> list of (mse, snr) over seeds
    res = defaultdict(list)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        Xtr = Xtr_pool[rng.integers(0, len(Xtr_pool), N_TRAIN)]    # bootstrap train
        for N in N_VALUES:
            for mname, dec_fn in DECODERS.items():
                torch.manual_seed(seed)                            # identical encoder init
                enc, dec = EdgeEncoder(N), dec_fn(N)
                train(enc, dec, Xtr, Xte, EPOCHS)                  # trains in place
                enc.eval(); dec.eval()
                with torch.no_grad():
                    z = enc(Xte)
                    for b in BITS:
                        res[(mname, N, b)].append(lossiness(dec(quantize(z, b)), Xte))
                print(f"[seed {seed}] N={N:>2} {mname:<8} done", flush=True)

    def agg(key):
        a = np.array(res[key])
        return a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean()   # mse_mean, mse_std, snr_mean

    # ---- table ----
    print(f"\n{'model':<8}{'N':>3}{'bits':>5}{'payload':>9}{'MSE (mean+/-std)':>22}{'SNR dB':>9}")
    print("-" * 56)
    for mname in DECODERS:
        for N in N_VALUES:
            for b in BITS:
                mm, ms, snr = agg((mname, N, b))
                print(f"{mname:<8}{N:>3}{b:>5}{N*b:>7} b{f'{mm:.4f}+/-{ms:.4f}':>22}{snr:>9.1f}")
        print()

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"hybrid": "#1f77b4", "matched": "#ff7f0e"}

        # (1) rate-distortion: all (N,b) points + Pareto frontier (best MSE at <= payload bits)
        plt.figure(figsize=(8, 5))
        for mname in DECODERS:
            pts = sorted((N * b, agg((mname, N, b))[0]) for N in N_VALUES for b in BITS)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.scatter(xs, ys, s=22, alpha=0.4, color=colors[mname])
            front_x, front_y, best = [], [], float("inf")
            for x_, y_ in pts:                          # cumulative-min MSE = achievable frontier
                best = min(best, y_)
                front_x.append(x_); front_y.append(best)
            plt.plot(front_x, front_y, "-o", color=colors[mname], lw=2,
                     label=f"{mname} (best achievable)")
        plt.xscale("log")
        plt.xlabel("payload per window = N x bits   (fewer bits ←)")
        plt.ylabel("reconstruction MSE   (lower is better ↓)")
        plt.title("Rate-distortion: does the quantum advantage survive quantization? (UCI HAR)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("results_quantization.png", dpi=140)

        # (2) detail at N=8 (the sweet spot): MSE vs bits, hybrid vs matched
        plt.figure(figsize=(7, 4.5))
        for mname in DECODERS:
            ys = [agg((mname, 8, b))[0] for b in BITS]
            es = [agg((mname, 8, b))[1] for b in BITS]
            plt.errorbar(BITS, ys, yerr=es, marker="o", capsize=4, color=colors[mname], label=mname)
        plt.gca().invert_xaxis()                         # fewer bits to the right (more compression)
        plt.xlabel("bits per latent value   (more compression →)")
        plt.ylabel("reconstruction MSE   (lower is better ↓)")
        plt.title("Quantization at N=8 (32x): hybrid vs. classical")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("results_quantization_N8.png", dpi=140)
        print("Saved -> results_quantization.png, results_quantization_N8.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
