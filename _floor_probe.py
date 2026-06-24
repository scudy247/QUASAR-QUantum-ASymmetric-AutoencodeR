"""Option B -- reconstruction floor probe (read-only on the data, clean channel / no noise).

At each bottleneck N it reports three reconstruction MSEs on the FIXED test set:
  PCA floor : best possible LINEAR rebuild from N components (the linear wall)
  AE linear : encoder + linear cloud expansion  (QIC_HEAD=linear)  -- current model
  AE mlp    : encoder + nonlinear cloud expansion (QIC_HEAD=mlp)   -- Option A

Reading:
  AE mlp << PCA floor  -> nonlinear decoding wins; adopt the mlp head (MSE can drop at fixed N)
  AE mlp ~= PCA floor  -> at the noise floor; stop chasing MSE, lean on the robustness story
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np, torch, warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(8)
import run_experiment as R

N_VALUES, SEED, EPOCHS = [2, 4, 8, 10], 0, 80
Xtr, Xte = R.load_d256()                       # uses per-channel normalization
Xtr_np, Xte_np = Xtr.numpy(), Xte.numpy()

# --- PCA floor: keep top-N components from the TRAIN basis, reconstruct the TEST set ---
mu = Xtr_np.mean(0)
_, _, Vt = np.linalg.svd(Xtr_np - mu, full_matrices=False)

def pca_mse(N):
    P = Vt[:N]                                   # (N, D) top-N principal directions
    rec = (Xte_np - mu) @ P.T @ P + mu           # project to N dims and back
    return float(((rec - Xte_np) ** 2).mean())

def ae_mse(N, head):
    R.DECODER_HEAD = head                        # _expansion() reads this global at build time
    torch.manual_seed(SEED)                      # identical encoder init across heads
    return R.train(R.EdgeEncoder(N), R.pure_classical_decoder(N), Xtr, Xte, EPOCHS)

print(f"floor probe | encoder={R.ENCODER_KIND} epochs={EPOCHS} seed={SEED} "
      f"| test {tuple(Xte.shape)}\n")
print(f"{'N':>3}{'comp':>7}{'PCA floor':>12}{'AE linear':>12}{'AE mlp':>12}{'mlp vs PCA':>12}")
print("-" * 58)
for N in N_VALUES:
    pca = pca_mse(N)
    lin = ae_mse(N, "linear")
    mlp = ae_mse(N, "mlp")
    gain = (pca - mlp) / pca * 100                # % below the linear wall the mlp reaches
    print(f"{N:>3}{R.D // N:>6}x{pca:>12.4f}{lin:>12.4f}{mlp:>12.4f}{gain:>11.1f}%", flush=True)
