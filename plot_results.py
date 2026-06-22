"""Make a clear figure of the hybrid-vs-classical results.

Parses run_sweep.log (so it never needs to re-run the experiment), aggregates
mean +/- std per bottleneck size, and plots with evenly-spaced, dual-labelled ticks
(N and compression ratio), a 'lower is better' axis, and the hybrid-wins region shaded.

Run:  ../.venv/bin/python plot_results.py
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = 256
LOG = "run_sweep.log"

# parse lines like: [seed 1/8  N= 8 (32x)]  hybrid=0.0236, matched=0.0255, pure=0.0280
pat = re.compile(r"N=\s*(\d+).*?hybrid=([\d.]+),\s*matched=([\d.]+),\s*pure=([\d.]+)")
vals = {}
for line in open(LOG):
    m = pat.search(line)
    if not m:
        continue
    n = int(m.group(1))
    d = vals.setdefault(n, {"hybrid": [], "matched": [], "pure": []})
    d["hybrid"].append(float(m.group(2)))
    d["matched"].append(float(m.group(3)))
    d["pure"].append(float(m.group(4)))

Ns = sorted(vals, reverse=True)          # 10,8,6,4,2 -> compression ascending left->right
x = np.arange(len(Ns))


def stat(n, k):
    a = np.array(vals[n][k])
    return a.mean(), a.std()


series = [("hybrid", "hybrid (quantum decoder)", "o", "#1f77b4"),
          ("matched", "matched classical (fair baseline)", "s", "#ff7f0e"),
          ("pure", "pure classical", "^", "#2ca02c")]

fig, ax = plt.subplots(figsize=(8, 5))

# shade the region where the hybrid has the lowest error (compression <= 42x, i.e. N >= 6)
win = [i for i, n in enumerate(Ns) if n >= 6]
ax.axvspan(min(win) - 0.5, max(win) + 0.5, color="#1f77b4", alpha=0.07)
ax.text(np.mean(win), 0.95, "hybrid has lowest error",
        transform=ax.get_xaxis_transform(), ha="center", va="top",
        color="#1f77b4", fontsize=10)
ax.text(len(Ns) - 1, 0.95, "hybrid worse\n(too few qubits)",
        transform=ax.get_xaxis_transform(), ha="center", va="top",
        color="#7f7f7f", fontsize=9)

for key, label, mk, c in series:
    means = [stat(n, key)[0] for n in Ns]
    stds = [stat(n, key)[1] for n in Ns]
    ax.errorbar(x, means, yerr=stds, marker=mk, ms=7, lw=2, capsize=4, label=label, color=c)

ax.set_xticks(x)
ax.set_xticklabels([f"N={n}\n{D // n}×" for n in Ns])
ax.set_xlabel("bottleneck size  /  compression ratio        (more compression →)")
ax.set_ylabel("reconstruction MSE   (lower is better ↓)")
ax.set_title(f"Hybrid quantum vs. classical decoders — UCI HAR, D={D}, {len(vals[Ns[0]]['hybrid'])} seeds")
ax.grid(axis="y", alpha=0.3)
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig("results_hybrid.png", dpi=140)
print("Saved -> results_hybrid.png")
