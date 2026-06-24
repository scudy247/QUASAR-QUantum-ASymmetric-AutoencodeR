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
import sys
import warnings
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")        # cap threads BEFORE numpy/torch import (32-core shared box)
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from data import load_dataset

torch.set_num_threads(8)
warnings.filterwarnings("ignore")   # silence benign torch/pennylane deprecation chatter

# ---- hyperparameters (tweak these) ------------------------------------------
D = 256                          # telemetry dimension: 2 HAR channels x 128 samples
N_VALUES = [2, 4, 6, 8, 10]      # bottleneck = qubit count -> compression D/N = 128x ... 26x
HIDDEN = 32                      # encoder hidden width (kept tiny for a TinyML footprint)
Q_LAYERS = 3                     # depth of the variational quantum circuit
EPOCHS = 120                     # training epochs (mini-batch Adam + cosine LR decay)
LR = 0.01                        # initial learning rate (cosine-decayed to ~0 over training)
BATCH = 128                      # mini-batch size -> many gradient updates per epoch (vs 1 for full-batch)
N_TRAIN, N_TEST = 1200, 1000     # N_TRAIN: legacy bootstrap size (the sweep now trains on the full pool)
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7] # repeats -> error bars (more seeds = tighter bars)
ENCODER_KIND = os.environ.get("QIC_ENCODER", "cnn").lower()  # "cnn" | "gru" | "fft" (env or --encoder)
RNN_HIDDEN = 32                  # GRU hidden size; ~14.5 KB encoder (< 66 KB edge budget)
DECODER_HEAD = os.environ.get("QIC_HEAD", "linear").lower()  # "linear" | "mlp": cloud expansion z->D
HEAD_HIDDEN = 256                # hidden width of the nonlinear (mlp) decoder head


# ---- data: real UCI HAR turned into a D=256 multi-modal vector ---------------
def load_d256():
    """Build the dataset once. Returns:
      Xtr_pool : full scaled training tensor (each seed bootstraps from this)
      Xte      : a FIXED test subsample (same across seeds, for fair comparison)
    Each sample is two HAR inertial channels (128 each) concatenated -> 256 dims,
    min-max scaled to [-1, 1] (so it matches the decoder's tanh / <Z> output range).
    Scaling is PER CHANNEL (one min/max shared across that channel's 128 time samples),
    NOT per time-step: a per-instant min/max would apply a different affine to each of the
    128 positions and warp the within-window waveform. One scalar per channel preserves the
    signal's temporal shape while still bringing every channel into [-1, 1]."""
    Xtr, _, Xte, _ = load_dataset("har")                  # HAR windows: (num, 9, 128)
    prep = lambda X: np.concatenate([X[:, 0, :], X[:, 1, :]], axis=1)   # 2 channels -> 256
    Xtr, Xte = prep(Xtr), prep(Xte)
    L = D // CHANNELS                                      # samples per channel (128)
    tr3 = Xtr.reshape(len(Xtr), CHANNELS, L)              # (num, 2, 128): expose the channel axis
    mn = tr3.min(axis=(0, 2), keepdims=True)             # per-channel min over TRAIN samples & time
    mx = tr3.max(axis=(0, 2), keepdims=True)             # -> shape (1, CHANNELS, 1), broadcasts over time
    span = np.where(mx - mn == 0, 1.0, mx - mn)
    sc = lambda A: (2 * (A.reshape(len(A), CHANNELS, L) - mn) / span - 1
                    ).reshape(len(A), D).astype(np.float32)
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
class CNNEncoder(nn.Module):
    """1D-CNN encoder: convolutions over the (2 x 128) inertial signals (weights shared over time)."""
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


class GRUEncoder(nn.Module):
    """Recurrent encoder: a GRU reads the window as a length-128 sequence of 2-channel samples
    and compresses its final hidden state to the latent. Captures temporal correlations; still
    tiny (~14.5 KB at H=32, well under the 66 KB edge budget)."""
    def __init__(self, N, hidden=None):
        super().__init__()
        H = hidden or RNN_HIDDEN
        self.rnn = nn.GRU(input_size=CHANNELS, hidden_size=H, batch_first=True)
        self.head = nn.Sequential(nn.Linear(H, N), nn.Tanh())

    def forward(self, x):
        x = x.view(x.size(0), CHANNELS, -1).transpose(1, 2)   # (B,256)->(B,2,128)->(B,128,2): 128 steps x 2 feats
        _, h = self.rnn(x)                                     # h: (1, B, H) final hidden state
        return self.head(h[-1])                                # latent z in [-1, 1]^N


class FFTEncoder(nn.Module):
    """Frequency-domain encoder. Takes the real FFT of each channel and keeps BOTH the real and
    imaginary parts (i.e. magnitude AND phase) -- phase is what encodes *where* a transient sits
    in time, so we must not drop it. A small MLP then LEARNS which spectral components matter
    (not a naive top-k truncation), letting the device allocate the N latents across low/high
    frequencies. Motivation: sharp peaks are broadband in frequency; an explicit spectral view
    may help the encoder preserve them better than a time-domain CNN. Output: z in [-1,1]^N."""
    def __init__(self, N, hidden=None):
        super().__init__()
        L = D // CHANNELS                          # samples per channel (128)
        bins = L // 2 + 1                          # rfft output length (65)
        feat = CHANNELS * bins * 2                 # real + imag, both channels (260)
        H = hidden or HIDDEN
        self.head = nn.Sequential(nn.Linear(feat, H), nn.ReLU(), nn.Linear(H, N), nn.Tanh())

    def forward(self, x):
        x = x.view(x.size(0), CHANNELS, -1)        # (B, 256) -> (B, 2, 128): restore channel/time
        Xf = torch.fft.rfft(x, dim=-1)             # (B, 2, 65) complex spectrum per channel
        feats = torch.cat([Xf.real, Xf.imag], dim=-1).flatten(1)  # keep phase; -> (B, 260)
        return self.head(feats)                    # latent z in [-1, 1]; no quantum-specific scaling


def EdgeEncoder(N):
    """Encoder factory: returns the selected edge encoder. Choose CNN, GRU or FFT via the
    QIC_ENCODER env var ('cnn'/'gru'/'fft') or run_experiment's --encoder flag; every script
    that imports EdgeEncoder respects the choice."""
    if ENCODER_KIND == "gru":
        return GRUEncoder(N)
    if ENCODER_KIND == "fft":
        return FFTEncoder(N)
    return CNNEncoder(N)


def _expansion(N):
    """Cloud-side expansion: maps the N decoder features back up to the full D telemetry.
    'linear' (default) = the original Linear(N -> D), an essentially LINEAR reconstruction
    that cannot beat the PCA floor. 'mlp' = a nonlinear head (Linear -> ReLU -> Linear); the
    cloud is unconstrained, so this can exploit nonlinear manifold structure and lower MSE at
    the SAME bottleneck N (no extra qubits, no extra transmitted data). Set via QIC_HEAD."""
    if DECODER_HEAD == "mlp":
        return nn.Sequential(nn.Linear(N, HEAD_HIDDEN), nn.ReLU(), nn.Linear(HEAD_HIDDEN, D))
    return nn.Linear(N, D)


# PHASE 2: quantum cloud decoder. The cloud receives the raw latent z (the N-dim output of
# the classical encoder) from the (dummy) device and does the angle embedding ITSELF as its
# FIRST step: qml.AngleEmbedding consumes the encoder output as qubit rotation angles. The
# device still ships raw z in [-1,1] (no quantum-specific scaling on-device); the CLOUD maps
# that to the full RX range by multiplying by pi, so the data-dependent rotations span
# [-pi, pi] (a full turn) instead of just +/-1 rad -- this widens the encoding's frequency
# content / input separability without changing what the device transmits. len(features)=N
# <= N qubits is required and guaranteed (both equal N). BasicEntanglerLayers then mixes them;
# we read N Pauli-Z expectation values; a classical Linear(N -> D) expands back to telemetry.
def quantum_decoder(N):
    dev = qml.device("lightning.qubit", wires=N)        # fast C++ simulator

    @qml.qnode(dev, interface="torch", diff_method="adjoint")   # memory-efficient grads
    def circ(inputs, weights):
        qml.AngleEmbedding(inputs * np.pi, wires=range(N))  # cloud-side scale: [-1,1] -> [-pi,pi] RX range
        qml.BasicEntanglerLayers(weights, wires=range(N))  # parameterized entangling layers
        return [qml.expval(qml.PauliZ(i)) for i in range(N)]

    # TorchLayer makes the quantum circuit a normal PyTorch layer (trainable weights inside).
    qlayer = qml.qnn.TorchLayer(circ, {"weights": (Q_LAYERS, N)})
    return nn.Sequential(qlayer, _expansion(N))            # VQC + classical expansion (linear|mlp)


def matched_classical_decoder(N):
    # FAIR baseline: replace the quantum N->N map with a classical N->N map (Linear+Tanh).
    return nn.Sequential(nn.Linear(N, N), nn.Tanh(), _expansion(N))


def pure_classical_decoder(N):
    # Weak baseline: no middle block -- just the cloud expansion (linear|mlp) straight from z.
    return _expansion(N)


class _CosSinFeatures(nn.Module):
    """Classical analogue of the quantum decoder's response: map z -> [cos(pi/2 z), sin(pi/2 z)].
    Smooth and Lipschitz-bounded (|d/dz| <= pi/2), like the <Z>-style outputs of the VQC, so it
    cannot amplify quantization noise. We use HALF frequency (pi/2, not pi) on purpose: at pi the
    map is degenerate at z=+/-1 (cos(+/-pi)=-1, sin(+/-pi)=0 -> both 1-bit levels collapse to the
    same feature, losing the sign). pi/2 keeps z=+1 -> (0,+1), z=-1 -> (0,-1): bounded AND
    injective at the 1-bit levels, so it's a fair bounded-gain baseline. No trainable params."""
    def forward(self, z):
        return torch.cat([torch.cos(0.5 * np.pi * z), torch.sin(0.5 * np.pi * z)], dim=-1)


def bounded_classical_decoder(N):
    """F4 control: a BOUNDED-GAIN classical decoder. If this matches the hybrid's low-bit
    robustness, the advantage is 'bounded/smooth decoder', not 'quantum'. If it does NOT, the
    quantum circuit is doing something a cheap classical bounded map can't -- a stronger claim."""
    return nn.Sequential(_CosSinFeatures(), nn.Linear(2 * N, D))


class _AffineReadout(nn.Module):
    """Elementwise affine + tanh on the 2**N measured probabilities (no mixing across
    outputs) -> the quantum circuit must carry the reconstruction; this only rescales.
    Probabilities are ~1/dim, so we pre-scale by `dim` to give the affine usable range."""
    def __init__(self, dim):
        super().__init__()
        self.scale = float(dim)
        self.w = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, p):
        return torch.tanh(p * self.scale * self.w + self.b)


def amplitude_decoder(N, layers=Q_LAYERS):
    """Low-parameter quantum decoder: the VQC's 2**N output probabilities ARE the D
    reconstruction values (requires 2**N == D, e.g. N=8, D=256). The dominant classical
    Linear(N->D) is removed, so the quantum circuit does the decoding -> far fewer params."""
    assert 2 ** N == D, f"amplitude readout needs 2**N == D (got 2**{N}={2 ** N}, D={D})"
    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circ(inputs, weights):
        qml.AngleEmbedding(inputs * np.pi, wires=range(N))  # cloud-side scale: [-1,1] -> [-pi,pi] RX range
        qml.StronglyEntanglingLayers(weights, wires=range(N))
        return qml.probs(wires=range(N))                 # 2**N = D outputs

    qlayer = qml.qnn.TorchLayer(circ, {"weights": qml.StronglyEntanglingLayers.shape(layers, N)})
    return nn.Sequential(qlayer, _AffineReadout(D))


def lowrank_classical_decoder(N, rank=2):
    """Fair LOW-parameter classical baseline (N -> rank -> D) so its parameter count can be
    matched against the amplitude decoder's."""
    return nn.Sequential(nn.Linear(N, rank), nn.Tanh(), nn.Linear(rank, D))


DECODERS = {"hybrid": quantum_decoder,
            "matched": matched_classical_decoder,
            "pure": pure_classical_decoder}


# PHASE 3: end-to-end training. A single Adam optimizer updates encoder + decoder together;
# for the hybrid model the gradients flow through the quantum circuit into the encoder.
# Returns the held-out reconstruction MSE.
def train(encoder, decoder, Xtr, Xte, epochs=EPOCHS):
    model = nn.Sequential(encoder, decoder)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)  # LR: LR -> ~0 (settle into the minimum)
    loss_fn = nn.MSELoss()
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n)               # shuffle each epoch
        for i in range(0, n, BATCH):           # mini-batches: ~n/BATCH gradient updates per epoch
            xb = Xtr[perm[i:i + BATCH]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)      # autoencoder: target == input
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        return float(loss_fn(model(Xte), Xte))


# PHASE 4: severance. Export ONLY the classical encoder to ONNX (the file the device ships,
# convertible to TFLite-Micro for an MCU) and report its size in KB. The decoder stays in the cloud.
def encoder_onnx_kb(encoder, N):
    path = f"edge_encoder_N{N}.onnx"
    encoder.eval()
    try:
        torch.onnx.export(encoder, torch.randn(1, D), path,
                          input_names=["raw_sensor"], output_names=["latent_z"])
        return os.path.getsize(path) / 1024
    except Exception:
        return float("nan")    # some ops (e.g. FFT) may not export to ONNX cleanly; skip sizing



# ---- run the experiment -----------------------------------------------------
def main():
    global N_VALUES, SEEDS, EPOCHS, BATCH, ENCODER_KIND
    if "--encoder" in sys.argv:                    # pick CNN/GRU/FFT at the start
        ENCODER_KIND = sys.argv[sys.argv.index("--encoder") + 1].lower()
    if "--quick" in sys.argv:                      # fast smoke test before a heavy run
        N_VALUES, SEEDS, EPOCHS = [4, 8], [0], 15
    if "--epochs" in sys.argv:                      # override training length
        EPOCHS = int(sys.argv[sys.argv.index("--epochs") + 1])
    if "--nseeds" in sys.argv:                      # use seeds 0..n-1 (fewer = faster validation)
        SEEDS = list(range(int(sys.argv[sys.argv.index("--nseeds") + 1])))
    if "--batch" in sys.argv:
        BATCH = int(sys.argv[sys.argv.index("--batch") + 1])
    Xtr_pool, Xte = load_d256()
    print(f"encoder={ENCODER_KIND} | D={D} (2 HAR channels), train pool {tuple(Xtr_pool.shape)}, "
          f"test {tuple(Xte.shape)}, seeds {SEEDS}, epochs {EPOCHS}\n")

    # mse[(N, decoder_name)] -> list of test MSEs, one per seed
    mse = {(N, d): [] for N in N_VALUES for d in DECODERS}
    for si, seed in enumerate(SEEDS):
        Xtr = Xtr_pool                                            # full training set (per-seed variation = init + shuffle)
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
