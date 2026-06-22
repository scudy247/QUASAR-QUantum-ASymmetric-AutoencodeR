"""Fair comparison: quantum decoder vs. classical decoders, using the GOOD training pipeline.

This answers one question: "with proper training, does the quantum decoder reconstruct
better than a classical decoder of the same size?"

Unlike run_experiment.py (full-batch Adam on a 1200-sample bootstrap), every decoder here
is trained with the SAME high-quality regime as train_and_deploy.py:
  - full HAR training set, mini-batch Adam
  - a validation split with best-checkpoint (early-stopping) selection
For each seed the encoder starts from the SAME random init across all three decoders, so
any MSE difference is attributable to the decoder block alone (a controlled comparison).

Decoders compared (at a fixed bottleneck N):
  hybrid   encoder + [N-qubit VQC]          + Linear(N->D)    (quantum, the proposed model)
  matched  encoder + [Linear(N,N)+Tanh]     + Linear(N->D)    (fair classical baseline)
  pure     encoder + [ ]                     + Linear(N->D)    (weak baseline, no middle block)

Reports mean +/- std test MSE over seeds, prints a verdict, saves a comparison plot, and
saves the trained HYBRID halves (the deployable artifacts) from the first seed.

RUN
    cd OurFramework/qic
    ../.venv/bin/python compare_decoders.py --N 8 --epochs 120 --seeds 0 1 2
"""

import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn

from run_experiment import (EdgeEncoder, quantum_decoder,
                            matched_classical_decoder, pure_classical_decoder,
                            load_d256, D)
from train_and_deploy import train, save_halves

warnings.filterwarnings("ignore")

DECODERS = {"hybrid": quantum_decoder,
            "matched": matched_classical_decoder,
            "pure": pure_classical_decoder}


def main():
    ap = argparse.ArgumentParser(description="Quantum vs. classical decoder, fair comparison.")
    ap.add_argument("--N", type=int, default=8, help="bottleneck = qubit count (compression D/N)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    N = args.N

    Xtr_pool, Xte = load_d256()
    print(f"D={D}, N={N} (compression {D//N}x), seeds {args.seeds}")
    print(f"train pool {tuple(Xtr_pool.shape)}  test {tuple(Xte.shape)}\n")

    # mse[name] -> list of held-out test MSEs, one per seed
    mse = {name: [] for name in DECODERS}
    saved = False
    for seed in args.seeds:
        # seed-dependent train/val split (test stays fixed)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(Xtr_pool))
        n_val = int(len(Xtr_pool) * args.val_frac)
        Xval, Xtr = Xtr_pool[perm[:n_val]], Xtr_pool[perm[n_val:]]

        print(f"=== seed {seed} ===")
        for name, dfn in DECODERS.items():
            torch.manual_seed(seed)                 # identical ENCODER init across decoders
            encoder, decoder = EdgeEncoder(N), dfn(N)
            print(f" training '{name}' decoder...")
            encoder, decoder = train(encoder, decoder, Xtr, Xval,
                                     args.epochs, args.lr, args.batch_size, seed)
            with torch.no_grad():
                x_hat = decoder(encoder(Xte))       # full device->cloud path on TEST set
                mse[name].append(float(nn.MSELoss()(x_hat, Xte)))
            # save the deployable (hybrid) halves once, from the first seed
            if name == "hybrid" and not saved:
                save_halves(encoder, decoder, N)
                saved = True
        print()

    # ---- results table ----
    def ms(name):
        a = np.array(mse[name])
        return a.mean(), a.std()

    print(f"\n{'decoder':>10}{'test MSE (mean +/- std)':>28}")
    print("-" * 38)
    for name in DECODERS:
        m, s = ms(name)
        print(f"{name:>10}{f'{m:.4f} +/- {s:.4f}':>28}")

    # ---- verdict: hybrid vs. the FAIR classical baseline (matched) ----
    hm, hs = ms("hybrid")
    mm, mst = ms("matched")
    gap = mm - hm                                    # positive => hybrid is better
    overlap = abs(gap) < (hs + mst)                  # rough ~1-sigma overlap test
    print("\nverdict (hybrid vs. matched-classical, the fair test):")
    if gap > 0 and not overlap:
        print(f"  quantum BETTER by {gap:.4f} MSE, beyond ~1 sigma -> meaningful edge.")
    elif gap > 0:
        print(f"  quantum lower by {gap:.4f} MSE, but error bars overlap -> suggestive, not conclusive.")
    elif gap < 0 and not overlap:
        print(f"  quantum WORSE by {-gap:.4f} MSE, beyond ~1 sigma.")
    else:
        print(f"  essentially tied ({gap:+.4f} MSE, bars overlap).")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(DECODERS)
        means = [ms(n)[0] for n in names]
        stds = [ms(n)[1] for n in names]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        plt.figure(figsize=(6, 4.5))
        plt.bar(names, means, yerr=stds, capsize=6, color=colors, alpha=0.85)
        plt.ylabel("test reconstruction MSE (lower is better)")
        plt.title(f"Quantum vs. classical decoder (N={N}, {D//N}x, {len(args.seeds)} seeds)")
        for i, (m, s) in enumerate(zip(means, stds)):
            plt.text(i, m + s, f"{m:.4f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        out = f"compare_decoders_N{N}.png"
        plt.savefig(out, dpi=130)
        print(f"\nsaved plot -> {out}")
    except Exception as e:
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    main()
