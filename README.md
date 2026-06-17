# Hybrid Quantum-Classical Asymmetric Autoencoder (AIoT 2026)

Extreme compression of IoT telemetry for constrained links (UAN / LoRa): a tiny
**classical edge encoder** ships a few latent values; an expressive **quantum cloud
decoder** reconstructs the full signal. Trained end-to-end; only the encoder is deployed.

```
x in R^D --[Phase1: classical edge encoder, TinyML]--> z in R^N  (N << D)
z --[Phase2: VQC on N qubits]--> <Z_i> (N values) --[classical Linear N->D]--> x_hat in R^D
```

The classical expansion `Linear(N -> D)` means **only N (= bottleneck) qubits are
simulated**, so large D (here 256) is fully simulatable.

## Files

| File | Role |
|------|------|
| `run_experiment.py` | The whole experiment (Phases 1-4) + matched/pure classical baselines + plot. |
| `data.py` | Loads real UCI HAR (`load_dataset("har")`); synthetic fallback. |
| `datasets/` | UCI HAR inertial signals + `.npz` cache. |
| `edge_encoder_N{2,4,8}.onnx` | Exported edge encoders (Phase 4 artifact, ~7 KB each). |
| `results_hybrid.png` | Reconstruction MSE vs. compression: hybrid vs. classical. |

Environment: a Python venv with `torch`, `pennylane`, `pennylane-lightning`,
`onnx`, `onnxscript`, `scikit-learn`.

## Setup (the venv and dataset are not in git)

```bash
# 1) environment
python3 -m venv .venv
./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./.venv/bin/pip install pennylane pennylane-lightning onnx onnxscript scikit-learn matplotlib

# 2) UCI HAR dataset -> ./datasets/
mkdir -p datasets && cd datasets
curl -sSL -o har.zip "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
unzip -q har.zip && unzip -q "UCI HAR Dataset.zip"   # yields ./datasets/UCI HAR Dataset/
cd ..
```

First run caches the dataset to `datasets/har_cache.npz`.

## Run

```bash
cd /home/fabio/Quantum/OurFramework/qic
../.venv/bin/python run_experiment.py    # ~50 s (5 seeds); prints the table, exports ONNX, saves the plot
```

Knobs at the top of `run_experiment.py`: `D` (telemetry dim = 256), `N_VALUES` (bottleneck =
qubit count; keep small so it stays simulatable), `HIDDEN`, `Q_LAYERS`, `EPOCHS`, `SEEDS`.

## Result (real HAR, D=256, 5 seeds: mean ± std)

| N | compression | MSE hybrid | MSE matched-classical | MSE pure-classical | encoder |
|--:|--:|--:|--:|--:|--:|
| 2 | 128× | 0.0365 ± .0042 | 0.0338 ± .0023 | 0.0352 ± .0020 | 7.0 KB |
| 4 | 64×  | 0.0285 ± .0004 | 0.0288 ± .0003 | 0.0291 ± .0001 | 6.9 KB |
| 8 | 32×  | 0.0253 ± .0009 | 0.0267 ± .0013 | 0.0280 ± .0005 | 6.9 KB |

**Honest reading** (matched-classical is the fair test — same encoder + expansion, only the
middle N->N block differs):

- 128×: hybrid slightly *worse* than matched (error bars overlap) — a wash.
- 64×:  statistically **tied**.
- 32×:  hybrid shows a **small edge** (0.0253 vs 0.0267), but the ~1σ bands still overlap.
- The hybrid consistently beats *pure*-classical (no middle layer) — expected, weak baseline.

**Supported claims:** comparable fidelity at 32-128× with a **~7 KB TinyML edge encoder**;
simulatable at large D; a *suggestive (not conclusive)* small benefit at moderate (32×)
compression. **Not yet supported:** a robust "outperforms classical AEs" — the edge is within
~1σ and inconsistent across N; more seeds / a second dataset would be needed to claim it.

## Caveats / next steps

- Single seed; MSE gaps (~0.0005) are likely within noise -> add a seed loop for mean+/-std
  to firmly establish parity (each run is ~15 s).
- "PAoI reduction" = payload/compression ratio only (ignores propagation); not a true PAoI.
- Older directions (quantum-inspired transform coding, symmetric QAE) are archived in
  `/home/fabio/Quantum/_ourframework_old_workflows.tgz`.
