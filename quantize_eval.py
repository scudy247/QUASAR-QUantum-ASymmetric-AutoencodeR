"""Evaluation experiments for the hybrid quantum-classical autoencoder.

This single eval script now covers three modes (it incorporates the former noise_eval.py
and param_efficiency.py). Models/training/data are imported from run_experiment.py.

  python quantize_eval.py quant      # latent quantization / rate-distortion  -> results_quantization.png
  python quantize_eval.py noise      # channel-noise robustness               -> results_noise.png
  python quantize_eval.py parameff   # parameter-efficiency Pareto at N=8      -> results_param_efficiency.png

Flags:  --encoder cnn|gru   --quick   --epochs E   --seeds S [S ...]
"""

import argparse
import os
import warnings
from collections import defaultdict
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")        # cap threads BEFORE numpy/torch import
import numpy as np
import torch
torch.set_num_threads(8)

import run_experiment as R

warnings.filterwarnings("ignore")


def _ms(d, k):
    a = np.array(d[k])
    return a.mean(), a.std()


# -------------------- quant: latent quantization / rate-distortion --------------------
def quant(a):
    N_VALUES = [4, 8] if a.quick else [4, 6, 8, 10]
    BITS = [4, 1] if a.quick else [8, 4, 3, 2, 1]
    SEEDS = a.seeds or ([0] if a.quick else [0, 1, 2])
    EPOCHS = a.epochs or (5 if a.quick else 50)
    decoders = {"hybrid": R.quantum_decoder, "matched": R.matched_classical_decoder}
    if getattr(a, "control", False):
        decoders["bounded"] = R.bounded_classical_decoder   # F4 bounded-gain control (cos/sin map)
    Xtr_pool, Xte = R.load_d256()
    print(f"[quant] encoder={R.ENCODER_KIND} N={N_VALUES} bits={BITS} seeds={SEEDS}\n")

    def quantize(z, bits):
        lv = 2 ** bits
        return torch.round((torch.clamp(z, -1, 1) + 1) / 2 * (lv - 1)) / (lv - 1) * 2 - 1

    res = defaultdict(list)
    for s in SEEDS:
        rng = np.random.default_rng(s)
        Xtr = Xtr_pool[rng.integers(0, len(Xtr_pool), R.N_TRAIN)]
        for N in N_VALUES:
            for dn, df in decoders.items():
                torch.manual_seed(s)
                enc, dec = R.EdgeEncoder(N), df(N)
                R.train(enc, dec, Xtr, Xte, EPOCHS)
                enc.eval(); dec.eval()
                with torch.no_grad():
                    z = enc(Xte)
                    for b in BITS:
                        res[(dn, N, b)].append(float(((dec(quantize(z, b)) - Xte) ** 2).mean()))
                print(f"  [seed {s} N={N} {dn}] done", flush=True)

    print(f"\n{'dec':<8}{'N':>3}{'bits':>5}{'payload':>9}{'MSE':>10}")
    for dn in decoders:
        for N in N_VALUES:
            for b in BITS:
                print(f"{dn:<8}{N:>3}{b:>5}{N * b:>7} b{_ms(res, (dn, N, b))[0]:>10.4f}")
    Nshow = 8 if 8 in N_VALUES else N_VALUES[-1]
    _plot({dn: [(b, *_ms(res, (dn, Nshow, b))) for b in BITS] for dn in decoders},
          "bits per latent value (more compression →)", f"Quantization at N={Nshow}",
          "results_quantization.png", invert=True)


# -------------------- noise: channel-noise robustness --------------------
def noise(a):
    N_VALUES = [4, 8] if a.quick else [4, 6, 8, 10]
    SIGMAS = [0.0, 0.2] if a.quick else [0.0, 0.05, 0.1, 0.2, 0.4]
    SEEDS = a.seeds or ([0] if a.quick else [0, 1, 2])
    EPOCHS = a.epochs or (5 if a.quick else 50)
    decoders = {"hybrid": R.quantum_decoder, "matched": R.matched_classical_decoder}
    if getattr(a, "control", False):
        decoders["bounded"] = R.bounded_classical_decoder   # F4 bounded-gain control (cos/sin map)
    Xtr_pool, Xte = R.load_d256()
    print(f"[noise] encoder={R.ENCODER_KIND} N={N_VALUES} sigmas={SIGMAS} seeds={SEEDS}\n")

    res = defaultdict(list)
    for s in SEEDS:
        rng = np.random.default_rng(s)
        g = torch.Generator().manual_seed(s)
        Xtr = Xtr_pool[rng.integers(0, len(Xtr_pool), R.N_TRAIN)]
        for N in N_VALUES:
            for dn, df in decoders.items():
                torch.manual_seed(s)
                enc, dec = R.EdgeEncoder(N), df(N)
                R.train(enc, dec, Xtr, Xte, EPOCHS)
                enc.eval(); dec.eval()
                with torch.no_grad():
                    z = enc(Xte)
                    for sig in SIGMAS:
                        zc = z + sig * torch.randn(z.shape, generator=g)
                        res[(dn, N, sig)].append(float(((dec(zc) - Xte) ** 2).mean()))
                print(f"  [seed {s} N={N} {dn}] done", flush=True)

    Nshow = 8 if 8 in N_VALUES else N_VALUES[-1]
    print(f"\n{'dec':<8}{'N':>3}{'sigma':>7}{'MSE':>10}")
    for dn in decoders:
        for N in N_VALUES:
            for sig in SIGMAS:
                print(f"{dn:<8}{N:>3}{sig:>7.2f}{_ms(res, (dn, N, sig))[0]:>10.4f}")
    _plot({dn: [(sig, *_ms(res, (dn, Nshow, sig))) for sig in SIGMAS] for dn in decoders},
          "latent noise std (more channel noise →)", f"Channel-noise robustness at N={Nshow}",
          "results_noise.png", invert=False)


# -------------------- parameff: parameter-efficiency Pareto at N=8 --------------------
def parameff(a):
    N = 8
    SEEDS = a.seeds or ([0] if a.quick else [0, 1, 2])
    EPOCHS = a.epochs or (5 if a.quick else 60)
    decs = {"amplitude (q)": lambda: R.amplitude_decoder(N), "low-rank cls": lambda: R.lowrank_classical_decoder(N),
            "hybrid": lambda: R.quantum_decoder(N), "pure": lambda: R.pure_classical_decoder(N),
            "matched": lambda: R.matched_classical_decoder(N)}
    Xtr_pool, Xte = R.load_d256()
    print(f"[parameff] N={N} seeds={SEEDS} epochs={EPOCHS}\n")
    mse, params = defaultdict(list), {}
    for s in SEEDS:
        rng = np.random.default_rng(s)
        Xtr = Xtr_pool[rng.integers(0, len(Xtr_pool), R.N_TRAIN)]
        for name, df in decs.items():
            torch.manual_seed(s)
            enc, dec = R.EdgeEncoder(N), df()
            mse[name].append(R.train(enc, dec, Xtr, Xte, EPOCHS))
            params[name] = sum(p.numel() for p in dec.parameters())
            print(f"  [seed {s}] {name:<14} mse={mse[name][-1]:.4f} params={params[name]}", flush=True)
    print(f"\n{'decoder':<14}{'params':>10}{'MSE':>10}")
    for name in decs:
        print(f"{name:<14}{params[name]:>10}{_ms(mse, name)[0]:>10.4f}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(7.5, 5))
        for name in decs:
            q = "q" in name or name == "hybrid"
            m, sd = _ms(mse, name)
            plt.errorbar(params[name], m, yerr=sd, marker="o" if q else "s", ms=9, capsize=4,
                         color="#1f77b4" if q else "#ff7f0e")
            plt.annotate(name, (params[name], m), fontsize=8, textcoords="offset points", xytext=(6, 5))
        plt.xscale("log"); plt.xlabel("decoder params (fewer ←)"); plt.ylabel("MSE (lower ↓)")
        plt.title(f"Parameter efficiency at N={N}"); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig("results_param_efficiency.png", dpi=140)
        print("Saved -> results_param_efficiency.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


# -------------------- shared plot (MSE vs x, hybrid vs matched) --------------------
def _plot(series, xlabel, title, fname, invert):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        col = {"hybrid": "#1f77b4", "matched": "#ff7f0e", "bounded": "#2ca02c"}
        plt.figure(figsize=(7, 4.5))
        for name, pts in series.items():
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; es = [p[2] for p in pts]
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=4, color=col.get(name), label=name)
        if invert:
            plt.gca().invert_xaxis()
        plt.xlabel(xlabel); plt.ylabel("reconstruction MSE (lower ↓)"); plt.title(title)
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(fname, dpi=140)
        print(f"Saved -> {fname}")
    except Exception as e:
        print(f"(plot skipped: {e})")


def main():
    p = argparse.ArgumentParser(description="Evaluation experiments (quant / noise / parameff)")
    sub = p.add_subparsers(dest="mode", required=True)
    for m in ("quant", "noise", "parameff"):
        sp = sub.add_parser(m)
        sp.add_argument("--encoder", default="cnn", choices=["cnn", "gru"])
        sp.add_argument("--quick", action="store_true")
        sp.add_argument("--epochs", type=int, default=None)
        sp.add_argument("--seeds", type=int, nargs="+", default=None)
        sp.add_argument("--control", action="store_true")   # F4: add bounded-gain classical decoder
    a = p.parse_args()
    R.ENCODER_KIND = a.encoder
    {"quant": quant, "noise": noise, "parameff": parameff}[a.mode](a)


if __name__ == "__main__":
    main()
