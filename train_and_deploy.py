"""Train the hybrid quantum-classical autoencoder, then SPLIT it for deployment.

This is the deploy pipeline (run_experiment.py is the multi-seed benchmark). Here we do
ONE clean run with two clearly separated stages:

  STAGE A - TRAIN (joint, end-to-end)
    Build EdgeEncoder + quantum CloudDecoder and train them TOGETHER on the HAR training
    set. Gradients flow through the quantum circuit into the encoder, so both halves learn
    to cooperate. We monitor a validation split and save BOTH trained halves to disk.

  STAGE B - SPLIT + TEST (deployment simulation)
    Reload the two halves as INDEPENDENT artifacts and run the real data path on the
    held-out TEST set:
        IoT device :  x --[encoder]--> z          (z is the ONLY thing transmitted)
        cloud      :  z --[quantum decoder]--> x_hat
    then report reconstruction MSE. The encoder is also exported to ONNX (the file the
    device actually ships).

Artifacts written (for N qubits):
    trained_encoder_N{N}.pt    encoder weights      -> goes on the IoT device
    edge_encoder_N{N}.onnx     encoder, ONNX form   -> device deployment artifact
    trained_decoder_N{N}.pt    quantum decoder      -> stays in the cloud
    reconstruction_N{N}.png    a few test signals: original vs. reconstructed

RUN
    cd OurFramework/qic
    ../.venv/bin/python train_and_deploy.py                 # defaults: N=6, 80 epochs
    ../.venv/bin/python train_and_deploy.py --N 8 --epochs 120 --lr 0.01
"""

import os
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn

# Reuse the exact model definitions and data loader from the benchmark so the deployed
# model is identical to the one being studied there.
from run_experiment import EdgeEncoder, quantum_decoder, load_d256, D

warnings.filterwarnings("ignore")


# ---- STAGE A: train the full autoencoder end-to-end -------------------------------------
def train(encoder, decoder, Xtr, Xval, epochs, lr, batch_size, seed):
    """Jointly train encoder + decoder on Xtr, tracking validation MSE on Xval.
    Returns the trained (encoder, decoder) with the BEST validation weights restored."""
    torch.manual_seed(seed)
    model = nn.Sequential(encoder, decoder)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = len(Xtr)
    best_val, best_state = float("inf"), None
    rng = np.random.default_rng(seed)

    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)                       # shuffle each epoch
        for i in range(0, n, batch_size):
            xb = Xtr[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)               # autoencoder target == input
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_mse = float(loss_fn(model(Xtr), Xtr))
            val_mse = float(loss_fn(model(Xval), Xval))
        if val_mse < best_val:                          # keep the best-generalizing weights
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  epoch {ep:>3}/{epochs}   train MSE {tr_mse:.4f}   "
                  f"val MSE {val_mse:.4f}   (best {best_val:.4f})", flush=True)

    model.load_state_dict(best_state)                   # restore best checkpoint
    print(f"  -> best validation MSE: {best_val:.4f}")
    return encoder, decoder


# ---- STAGE B helpers: export + reload ---------------------------------------------------
def save_halves(encoder, decoder, N):
    """Persist the two halves separately (the 'severance' step) and export the encoder ONNX."""
    enc_pt = f"trained_encoder_N{N}.pt"
    dec_pt = f"trained_decoder_N{N}.pt"
    onnx_path = f"edge_encoder_N{N}.onnx"
    torch.save(encoder.state_dict(), enc_pt)
    torch.save(decoder.state_dict(), dec_pt)
    encoder.eval()
    torch.onnx.export(encoder, torch.randn(1, D), onnx_path,
                      input_names=["raw_sensor"], output_names=["latent_z"])
    kb = os.path.getsize(onnx_path) / 1024
    print(f"  saved encoder -> {enc_pt}, {onnx_path} ({kb:.1f} KB)")
    print(f"  saved decoder -> {dec_pt}")
    return enc_pt, dec_pt, kb


def load_device_encoder(N, path):
    """Reload the encoder as a fresh, independent object (simulating the IoT device)."""
    enc = EdgeEncoder(N)
    enc.load_state_dict(torch.load(path))
    enc.eval()
    return enc


def load_cloud_decoder(N, path):
    """Reload the quantum decoder as a fresh, independent object (simulating the cloud)."""
    dec = quantum_decoder(N)
    dec.load_state_dict(torch.load(path))
    dec.eval()
    return dec


# ---- main: train, split, then test the reassembled pipeline -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Train then split the hybrid quantum autoencoder.")
    ap.add_argument("--N", type=int, default=6, help="bottleneck = qubit count (compression D/N)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of train held out for validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; reload trained_*_N{N}.pt and just re-run the test")
    args = ap.parse_args()
    N = args.N

    # ---- data: train pool + FIXED held-out test set (never seen during training) ----
    Xtr_pool, Xte = load_d256()
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Xtr_pool))
    n_val = int(len(Xtr_pool) * args.val_frac)
    Xval = Xtr_pool[perm[:n_val]]
    Xtr = Xtr_pool[perm[n_val:]]
    print(f"D={D}, N={N} (compression {D//N}x)")
    print(f"train {tuple(Xtr.shape)}  val {tuple(Xval.shape)}  test {tuple(Xte.shape)}\n")

    enc_pt, dec_pt = f"trained_encoder_N{N}.pt", f"trained_decoder_N{N}.pt"
    onnx_path = f"edge_encoder_N{N}.onnx"

    if args.eval_only:
        # Skip STAGE A entirely; just reload the saved halves and report ONNX size.
        if not (os.path.exists(enc_pt) and os.path.exists(dec_pt)):
            raise SystemExit(f"--eval-only: missing {enc_pt} or {dec_pt}; train first.")
        print("EVAL-ONLY - skipping training, reloading saved halves")
        kb = os.path.getsize(onnx_path) / 1024 if os.path.exists(onnx_path) else float("nan")
    else:
        # ===== STAGE A: TRAIN =====
        print("STAGE A - training encoder + quantum decoder end-to-end")
        encoder = EdgeEncoder(N)
        decoder = quantum_decoder(N)
        encoder, decoder = train(encoder, decoder, Xtr, Xval,
                                 args.epochs, args.lr, args.batch_size, args.seed)

        # ===== severance: save the two halves separately =====
        print("\nsplitting the trained model into device + cloud artifacts")
        enc_pt, dec_pt, kb = save_halves(encoder, decoder, N)

    # ===== STAGE B: DEPLOY + TEST (reload halves as independent components) =====
    print("\nSTAGE B - deploying: reloading the two halves and testing the full pipeline")
    device_encoder = load_device_encoder(N, enc_pt)     # the IoT device
    cloud_decoder = load_cloud_decoder(N, dec_pt)       # the cloud

    with torch.no_grad():
        z = device_encoder(Xte)                         # device: x -> latent (transmitted)
        x_hat = cloud_decoder(z)                         # cloud:  latent -> reconstruction
        test_mse = float(nn.MSELoss()(x_hat, Xte))

    raw_bytes, sent_bytes = D * 4, N * 4
    print(f"\n  test reconstruction MSE : {test_mse:.4f}")
    print(f"  payload per window      : raw {raw_bytes} B -> sent {sent_bytes} B "
          f"({raw_bytes / sent_bytes:.0f}x smaller)")
    print(f"  edge encoder size       : {kb:.1f} KB (ONNX)")

    # ---- visualize a few reconstructions so we can see the trained system working ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = 4
        fig, axes = plt.subplots(k, 1, figsize=(8, 1.8 * k), sharex=True)
        for ax, idx in zip(axes, range(k)):
            ax.plot(Xte[idx].numpy(), label="original", lw=1.2)
            ax.plot(x_hat[idx].numpy(), label="reconstructed", lw=1.2, alpha=0.8)
            ax.set_ylabel(f"sample {idx}")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title(f"Original vs. reconstructed (N={N}, {D//N}x compression, "
                          f"test MSE {test_mse:.4f})")
        axes[-1].set_xlabel("feature index (0-127: body_acc_x, 128-255: body_acc_y)")
        plt.tight_layout()
        out = f"reconstruction_N{N}.png"
        plt.savefig(out, dpi=130)
        print(f"  saved reconstruction plot -> {out}")
    except Exception as e:
        print(f"  (reconstruction plot skipped: {e})")


if __name__ == "__main__":
    main()
