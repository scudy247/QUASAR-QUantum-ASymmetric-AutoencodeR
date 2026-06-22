"""Experiment: Hybrid Quantum-Classical Asymmetric Autoencoder for extreme IoT compression.

GOAL
----
Compress high-dimensional IoT telemetry on a constrained device with a *tiny* classical
encoder, and reconstruct it in the cloud with an expressive *quantum* decoder. We measure
reconstruction fidelity (MSE) vs. compression ratio, and check whether the quantum decoder
actually helps by comparing it against two classical decoders of matched/lower capacity.

ARCHITECTURE (four phases; see the PHASE comments next to each function)
-----------------------------------------------------------------------
  Phase 1  EdgeEncoder (classical, runs on the *dummy* device):  x in R^D -> z in [-1,1]^N
                                  -> ships z (no quantum-specific scaling on the device)
  Phase 2  CloudDecoder (cloud): qml.AngleEmbedding(z) FIRST (embeds encoder output directly)
                                  -> N-qubit VQC -> N expectation values
                                  -> classical Linear(N -> D) expansion -> x_hat in R^D
  Phase 3  end-to-end training: one optimizer trains encoder + decoder jointly (MSE)
  Phase 4  severance: export ONLY the encoder to ONNX (the artifact the device ships)

WHY THIS IS SIMULATABLE AT LARGE D
----------------------------------
The number of simulated qubits equals the *bottleneck* N (e.g. 2/4/8), NOT the data
dimension D (=256). The classical Linear(N -> D) layer does the expansion. So D can be
large while we still only simulate a few qubits.

BASELINES (to attribute any gain to the quantum layer)
------------------------------------------------------
  hybrid   : encoder + [N-qubit VQC] + Linear(N->D)          (the proposed model)
  matched  : encoder + [Linear(N,N)+Tanh] + Linear(N->D)     (FAIR baseline: classical N->N)
  pure     : encoder + [Linear(N->D)]                        (no middle block; weak baseline)
The fair comparison is hybrid vs. matched (identical except the middle N->N transform).

STATISTICS
----------
Each configuration is trained over several SEEDS; we report mean +/- std. Per seed we
bootstrap-resample the training set and reseed weight init; the test set is fixed. Within a
seed the encoder init is identical across the three decoders (a controlled comparison).

RUN
---
  cd OurFramework/qic
  ../.venv/bin/python run_experiment.py        # ~50 s on CPU; prints table, exports ONNX, saves plot
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from data import load_dataset

warnings.filterwarnings("ignore")   # silence benign torch/pennylane deprecation chatter

# ---- hyperparameters (tweak these) ------------------------------------------
D = 256                          # telemetry dimension: 2 HAR channels x 128 samples
N_VALUES = [2, 4, 6, 8, 10]      # bottleneck = qubit count -> compression D/N = 128x ... 26x
HIDDEN = 32                      # encoder hidden width (kept tiny for a TinyML footprint)
Q_LAYERS = 3                     # depth of the variational quantum circuit
EPOCHS = 60                      # training epochs (full-batch Adam)
LR = 0.01                        # learning rate
N_TRAIN, N_TEST = 1200, 1000     # samples used per run
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7] # repeats -> error bars (more seeds = tighter bars)


# ---- data: real UCI HAR turned into a D=256 multi-modal vector ---------------
def load_d256():
    """Build the dataset once. Returns:
      Xtr_pool : full scaled training tensor (each seed bootstraps from this)
      Xte      : a FIXED test subsample (same across seeds, for fair comparison)
    Each sample is two HAR inertial channels (128 each) concatenated -> 256 dims,
    min-max scaled to [-1, 1] (so it matches the decoder's tanh / <Z> output range)."""
    Xtr, _, Xte, _ = load_dataset("har")                  # HAR windows: (num, 9, 128)
    prep = lambda X: np.concatenate([X[:, 0, :], X[:, 1, :]], axis=1)   # 2 channels -> 256
    Xtr, Xte = prep(Xtr), prep(Xte)
    mn, mx = Xtr.min(0), Xtr.max(0)                        # scale using TRAIN stats only
    span = np.where(mx - mn == 0, 1.0, mx - mn)
    sc = lambda A: (2 * (A - mn) / span - 1).astype(np.float32)
    Xtr, Xte = sc(Xtr), sc(Xte)
    r = np.random.default_rng(0)
    Xte_fixed = torch.tensor(Xte[r.choice(len(Xte), N_TEST, replace=False)])
    return torch.tensor(Xtr), Xte_fixed


# ---- models -----------------------------------------------------------------
CHANNELS = 2                      # raw input is 2 HAR channels (body_acc_x, body_acc_y); D = CHANNELS * 128

# PHASE 1: classical edge encoder (the deployed TinyML artifact). A *dummy* device: it only
# produces data, compresses it, and ships the result. It uses a 1D-CONVOLUTIONAL front-end --
# the idiomatic design for raw inertial signals: the flat input is reshaped back to its true
# (2 channels x 128 samples) layout and small filters slide along the TIME axis with weights
# shared across time, capturing local temporal structure (the transient spikes a flat dense
# layer smears). Output: compressed latent z in [-1,1]^N (Tanh-bounded), transmitted as-is.
# The device does NOT know the cloud is quantum -- angle embedding is the cloud's job (Phase 2).
class EdgeEncoder(nn.Module):
    def __init__(self, N):
        super().__init__()
        L = D // CHANNELS                                  # samples per channel (128)
        self.features = nn.Sequential(
            nn.Conv1d(CHANNELS, 8, kernel_size=7, stride=2, padding=3), nn.ReLU(),  # (2,128) -> (8, 64)
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),        # (8, 64) -> (16, 32)
        )
        # flatten keeps WHERE features occur in time (vs. global pooling, which discards it) --
        # important because the cloud must reconstruct the full waveform, not just classify it.
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * (L // 4), N), nn.Tanh())

    def forward(self, x):
        x = x.view(x.size(0), CHANNELS, -1)    # (B, 256) -> (B, 2, 128): restore channel/time layout
        return self.head(self.features(x))     # latent z in [-1, 1]; no quantum-specific scaling


# PHASE 2: quantum cloud decoder. The cloud receives the raw latent z (the N-dim output of
# the classical encoder) from the (dummy) device and does the angle embedding ITSELF as its
# FIRST step: qml.AngleEmbedding consumes the encoder output directly as qubit rotation angles
# (no scaling -- the encoder learns suitable values; len(features)=N <= N qubits is required
# and is guaranteed here since both equal N). BasicEntanglerLayers then mixes them; we read N
# Pauli-Z expectation values; finally a classical Linear(N -> D) expands back to full telemetry.
def quantum_decoder(N):
    dev = qml.device("lightning.qubit", wires=N)        # fast C++ simulator

    @qml.qnode(dev, interface="torch", diff_method="adjoint")   # memory-efficient grads
    def circ(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N))         # FIRST cloud step: embed encoder output directly
        qml.BasicEntanglerLayers(weights, wires=range(N))  # parameterized entangling layers
        return [qml.expval(qml.PauliZ(i)) for i in range(N)]

    # TorchLayer makes the quantum circuit a normal PyTorch layer (trainable weights inside).
    qlayer = qml.qnn.TorchLayer(circ, {"weights": (Q_LAYERS, N)})
    return nn.Sequential(qlayer, nn.Linear(N, D))          # VQC + classical expansion


def matched_classical_decoder(N):
    # FAIR baseline: replace the quantum N->N map with a classical N->N map (Linear+Tanh).
    return nn.Sequential(nn.Linear(N, N), nn.Tanh(), nn.Linear(N, D))


def pure_classical_decoder(N):
    # Weak baseline: latent expanded straight to D, no middle nonlinearity.
    return nn.Sequential(nn.Linear(N, D))


DECODERS = {"hybrid": quantum_decoder,
            "matched": matched_classical_decoder,
            "pure": pure_classical_decoder}


# PHASE 3: end-to-end training. A single Adam optimizer updates encoder + decoder together;
# for the hybrid model the gradients flow through the quantum circuit into the encoder.
# Returns the held-out reconstruction MSE.
def train(encoder, decoder, Xtr, Xte, epochs=EPOCHS):
    model = nn.Sequential(encoder, decoder)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xtr), Xtr)        # autoencoder: target == input
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(loss_fn(model(Xte), Xte))


# PHASE 4: severance. Export ONLY the classical encoder to ONNX (the file the device ships,
# convertible to TFLite-Micro for an MCU) and report its size in KB. The decoder stays in the cloud.
def encoder_onnx_kb(encoder, N):
    path = f"edge_encoder_N{N}.onnx"
    encoder.eval()
    torch.onnx.export(encoder, torch.randn(1, D), path,
                      input_names=["raw_sensor"], output_names=["latent_z"])
    return os.path.getsize(path) / 1024


# ---- run the experiment -----------------------------------------------------
def main():
    Xtr_pool, Xte = load_d256()
    print(f"D={D} (2 HAR channels), train pool {tuple(Xtr_pool.shape)}, "
          f"test {tuple(Xte.shape)}, seeds {SEEDS}\n")

    # mse[(N, decoder_name)] -> list of test MSEs, one per seed
    mse = {(N, d): [] for N in N_VALUES for d in DECODERS}
    for si, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        Xtr = Xtr_pool[rng.integers(0, len(Xtr_pool), N_TRAIN)]   # bootstrap training set
        for N in N_VALUES:
            for dname, dfn in DECODERS.items():
                torch.manual_seed(seed)                           # identical encoder init per seed
                mse[(N, dname)].append(train(EdgeEncoder(N), dfn(N), Xtr, Xte))
            # progress line so the run is observable in the log / tmux pane
            done = ", ".join(f"{d}={mse[(N, d)][-1]:.4f}" for d in DECODERS)
            print(f"[seed {si+1}/{len(SEEDS)}  N={N:>2} ({D//N}x)]  {done}", flush=True)

    kb = {N: encoder_onnx_kb(EdgeEncoder(N), N) for N in N_VALUES}   # edge model size per N

    def ms(N, d):                                  # mean, std over seeds
        a = np.array(mse[(N, d)])
        return a.mean(), a.std()

    # ---- results table (mean +/- std) ----
    print(f"{'N':>3}{'comp':>7}{'hybrid (mean+/-std)':>22}{'matched':>20}{'pure':>20}{'enc KB':>9}")
    print("-" * 81)
    for N in N_VALUES:
        cells = "".join(f"{f'{ms(N, d)[0]:.4f}+/-{ms(N, d)[1]:.4f}':>20}" for d in DECODERS)
        print(f"{N:>3}{D // N:>6}x{cells}{kb[N]:>9.1f}")

    # ---- payload / link context (honest: this is the compression ratio, not a true PAoI) ----
    R = 1200.0  # acoustic bitrate ~1.2 kbps (bits/s)
    print(f"\nlink @ {R/1000:.1f} kbps:  raw {D*4} B -> {D*4*8/R:.2f}s; "
          f"N=4 -> {4*4} B -> {4*4*8/R:.3f}s  ({D/4:.0f}x smaller payload)")

    # ---- plot: MSE vs compression, with error bars ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        comps = [D / N for N in N_VALUES]
        styles = {"hybrid": "o-", "matched": "s-", "pure": "^-"}
        labels = {"hybrid": "hybrid (quantum decoder)",
                  "matched": "matched classical", "pure": "pure classical"}
        plt.figure(figsize=(7, 4.5))
        for d in DECODERS:
            means = [ms(N, d)[0] for N in N_VALUES]
            stds = [ms(N, d)[1] for N in N_VALUES]
            plt.errorbar(comps, means, yerr=stds, fmt=styles[d], capsize=3, label=labels[d])
        plt.xlabel("compression ratio (D / N)")
        plt.ylabel("reconstruction MSE (test)")
        plt.title(f"Hybrid vs. classical decoders (HAR, D=256, {len(SEEDS)} seeds)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results_hybrid.png", dpi=130)
        print("\nSaved plot -> results_hybrid.png")
    except Exception as e:
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    main()
