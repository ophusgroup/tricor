"""Diagnostic: does the trained model use ``target_r`` from shell_target?

Companion to ``diagnose_weight_conditioning.py``.  That one tests whether
the model uses the *regime* features in the weight vector (grain_size,
crystalline_fraction, num_grains).  THIS one tests whether the model
uses the *bond-length target* in shell_target's per-pair features —
specifically, whether perturbing ``target_r`` for the (Si, O) pair
changes the predicted Si–O bond distance.

Why this matters: in cross-coordination evaluation on stishovite, the
model predicts Si–O at ~2.0 Å while shell_target says 1.78 Å.  Two
non-mutually-exclusive failure modes:

  (a) BOND-LENGTH EXTRAPOLATION GAP — model receives target_r=1.78 Å
      and tries to use it, but its training data only contains 6-coord
      M–O ≥ 1.85 Å.  Can't extrapolate below that range.  Fix: add
      training data with shorter 6-coord M–O distances.

  (b) SPECIES PRIOR DOMINATES — model receives target_r=1.78 Å but
      ignores it, defaulting to the memorized "6-coord oxide M–O ≈ 2.0 Å"
      from training.  Fix: architectural change (per-edge shell_target
      injection so target_r is delivered directly to the edge that
      bond-length is being predicted on).

Test design: forward stishovite (or any Si–O-containing structure) at
several artificial ``target_r`` values for the (Si, O) pair.  Other
pairs unchanged.  Run iterative inference for each.  Measure where the
predicted Si–O peak in g(r) lands.

Headline metric: SENSITIVITY SLOPE = Δpeak_predicted / Δtarget_r.
  * slope > 0.7  → conditioning works.  Failure on real stishovite is
                   bond-length extrapolation.  Fix: data.
  * slope 0.3-0.7 → partial conditioning.  Both contribute.
  * slope < 0.3  → species prior dominates.  Fix: architecture.

Run on whichever env has graphite + ase + a CUDA GPU available:
    python scripts/relaxml/diagnose_shell_target_conditioning.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

GPU_ID = 0
NUM_THREADS = 2

# Which variant's model to load.
#   "phys"     -> shelltgt_phys     (v1 phys; deep-set ShellTargetEncoder)
#   "baseline" -> model_shelltgt    (v1 plain nn.Embedding)
#   "v2"       -> shelltgt_phys_v2  (per-edge shell_target injection;
#                                    pair features fed directly into edges,
#                                    TripletTargetEncoder for triplets)
VARIANT = "v2"     # "phys" | "baseline" | "v2"

# Checkpoint to test.  Run on both baseline and phys checkpoints from the
# same training set to see whether physics features change the
# conditioning sensitivity.
#
# Heuristic for picking VARIANT correctly:
#   * If CHECKPOINT path contains ".../shelltgt_phys_v2/lightning_logs/..." or
#     run name has "_rinject" or "_v2" -> VARIANT="v2"
#   * If CHECKPOINT path contains ".../shelltgt_phys/lightning_logs/..." or
#     run name has "_phys" -> VARIANT="phys"
#   * If path is under top-level lightning_logs/ AND run name has "_z" or
#     "_Zembed" -> VARIANT="baseline"
#   * If load_state_dict errors with missing/unexpected ".table", ".mlp"
#     or "triplet_target_encoder" keys, you have the wrong VARIANT — flip it.
CHECKPOINT = (
    # "/home/ehrdt/tricor/scripts/relaxml/shelltgt_phys/"
    # "lightning_logs/phase_contrast-phys/version_1/checkpoints/last.ckpt"
    "/home/ehrdt/tricor/scripts/relaxml/shelltgt_phys_v2/"
    "lightning_logs/coord_phys_rinject/version_1/checkpoints/last.ckpt"
)

# Held-out structure to probe.  A stishovite trajectory is the
# motivating case (cross-coordination test), but this works on any .npz
# that contains both Si and O atoms — the script edits the (Si, O)
# pair's target_r and watches the predicted Si–O peak respond.
#
# Verify the file actually exists:
#   ls /home/ehrdt/tricor/scripts/relaxml/data/sio2_polymorphs_v1/SiO2/stishovite_trajectories/*.npz | head -3
# (then paste one of the listed filenames here)
NPZ_PATH = (
    "/home/ehrdt/tricor/scripts/relaxml/data/sio2_polymorphs_v1/SiO2/"
    "stishovite_trajectories/SiO2_stishovite_nanocrystalline_cell050_idx00005_seed000800005.npz"
)

# Perturbation grid.  These are the values we substitute for the (Si, O)
# entries in shell_pair_features[:, 0] (the target_r column).  Defaults
# bracket the relevant chemistry:
#
#   1.62 Å — 4-coord α-quartz Si–O distance
#   1.74 Å — average between 4- and 6-coord
#   1.78 Å — TRUE stishovite Si–O (the held-out chemistry)
#   1.90 Å — between stishovite and training-set average
#   2.00 Å — training-set 6-coord M–O average (Al2O3 / TiO2 / Ga2O3)
#   2.10 Å — outside training-set 6-coord range (longer than anything seen)
#
# Edit to add or remove points.  Want at least 3 to fit a sensitivity slope.
TARGET_R_VALUES = [1.62, 1.74, 1.78, 1.90, 2.00, 2.10]

# Inference parameters.  Match evaluate.py for consistency.
MAX_ITER = 50
CONVERGENCE_TOL_ANG = 0.01
USE_EMA_WEIGHTS = True
CUTOFF = 5.0

# Atomic numbers of the species pair to perturb.  For SiO2 / Si3N4
# stishovite-style tests this is (Si, O) = (14, 8).  Generalises to
# any pair by changing these.
SPECIES_A = 14   # Si
SPECIES_B = 8    # O

# Histogram parameters for measuring the predicted A–B peak.
HIST_R_MIN = 1.2
HIST_R_MAX = 3.0
HIST_BINS = 100

# ─────────────────────────────────────────────────────────────────────────────

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
_n = str(NUM_THREADS)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = _n

import sys
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(NUM_THREADS)
from torch_geometric.data import Batch

from tricor.flowmatch.flow_utils import periodic_radius_graph_cell_list

if VARIANT == "phys":
    from tricor.relaxml.shelltgt_phys import LitRelaxML
elif VARIANT == "baseline":
    from tricor.relaxml.model_shelltgt import LitRelaxML
elif VARIANT == "v2":
    # v2 reuses the same forward signature as phys/baseline (same data
    # layer), so the rest of this script is variant-agnostic — only the
    # LitRelaxML class differs (different module names: triplet_target_
    # encoder instead of shell_target_encoder; wider edge_encoder input).
    from tricor.relaxml.shelltgt_phys_v2 import LitRelaxML
else:
    raise SystemExit(
        f"VARIANT must be 'phys', 'baseline', or 'v2', got {VARIANT!r}"
    )

from tricor.relaxml.data_shelltgt import (
    NUM_WEIGHT_FEATURES, ShellTargetData, _weight_vector_from_row,
)


def _load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Mirror evaluate.py's loader: strip torch.compile prefix, return EMA."""
    state = torch.load(str(ckpt_path), map_location=device)
    sd = {k.replace("._orig_mod.", "."): v for k, v in state["state_dict"].items()}
    lit = LitRelaxML(**state["hyper_parameters"])
    lit.load_state_dict(sd, strict=True)
    lit.eval().to(device)
    if USE_EMA_WEIGHTS and hasattr(lit, "ema_model"):
        lit.ema_model.eval()
        return lit.ema_model.module.to(device)
    return lit.model.to(device)


def _build_batch(pos, cell, z, w, shell, cutoff):
    edge_index, edge_vec = periodic_radius_graph_cell_list(pos, cutoff, cell)
    edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
    P = shell["pair_species"].shape[0]
    T = shell["trip_species"].shape[0]
    data = ShellTargetData(
        z=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
        w=w.unsqueeze(0),
        shell_pair_species=shell["pair_species"],
        shell_pair_features=shell["pair_features"],
        shell_pair_batch=torch.zeros(P, dtype=torch.long, device=pos.device),
        shell_trip_species=shell["trip_species"],
        shell_trip_features=shell["trip_features"],
        shell_trip_batch=torch.zeros(T, dtype=torch.long, device=pos.device),
    )
    return Batch.from_data_list([data])


def _wrap_positions(pos, cell):
    """Fold positions back into the fundamental cell [0, L)^3."""
    inv_cell = torch.linalg.inv(cell)
    frac = pos @ inv_cell.T
    frac = frac - torch.floor(frac)
    return frac @ cell


def _forward(model, batch):
    return model(
        batch.z, batch.edge_index, batch.edge_attr,
        batch.w, batch.batch,
        batch.shell_pair_species, batch.shell_pair_features,
        batch.shell_pair_batch,
        batch.shell_trip_species, batch.shell_trip_features,
        batch.shell_trip_batch,
    )


@torch.no_grad()
def _run_iterative_inference(model, initial_pos, cell, species, w_vec,
                              shell, device, max_iter, tol, cutoff):
    """Same loop as evaluate.py.  No print spam; just returns final positions."""
    pos = torch.tensor(initial_pos, dtype=torch.float32, device=device)
    cell_t = torch.tensor(cell, dtype=torch.float32, device=device)
    w_t = torch.tensor(w_vec, dtype=torch.float32, device=device)
    z = torch.tensor(species, dtype=torch.long, device=device)

    for _ in range(max_iter):
        batch = _build_batch(pos, cell_t, z, w_t, shell, cutoff)
        delta = _forward(model, batch)
        pos_new = _wrap_positions(pos + delta, cell_t)
        max_step = (pos_new - pos).norm(dim=-1).max().item()
        pos = pos_new
        if max_step < tol:
            break

    return pos.cpu().numpy()


def _ab_distance_histogram(positions, species, cell, z_a, z_b,
                            r_min, r_max, n_bins):
    """First-shell-style distance histogram for (z_a, z_b) atom pairs.

    Computes minimum-image periodic distances between every (a, b) pair
    in the box and bins those that fall in [r_min, r_max].  Returns
    (bin_centers, counts, peak_position).  Peak position = bin center
    with the most counts (Si–O bond length).
    """
    cell_arr = np.asarray(cell, dtype=np.float64)
    inv_cell = np.linalg.inv(cell_arr)
    pos = np.asarray(positions, dtype=np.float64)
    sp = np.asarray(species, dtype=np.int64)
    a_idx = np.where(sp == z_a)[0]
    b_idx = np.where(sp == z_b)[0]
    if a_idx.size == 0 or b_idx.size == 0:
        raise ValueError(
            f"No atoms with z={z_a} or z={z_b} in the structure (have "
            f"{sorted(set(sp.tolist()))})."
        )

    # Pairwise (a -> b) displacements with minimum-image convention.
    # Done in chunks to avoid O(N_a * N_b * 3) memory blow-up on big cells.
    bin_edges = np.linspace(r_min, r_max, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    chunk = 1024
    for i_start in range(0, a_idx.size, chunk):
        i_end = min(i_start + chunk, a_idx.size)
        ai = a_idx[i_start:i_end]
        diff = pos[ai, None, :] - pos[None, b_idx, :]   # (chunk, N_b, 3)
        # Apply minimum-image
        frac = diff @ inv_cell.T
        frac -= np.round(frac)
        diff_min = frac @ cell_arr
        d = np.linalg.norm(diff_min, axis=-1).ravel()
        d = d[(d >= r_min) & (d < r_max)]
        if d.size > 0:
            h, _ = np.histogram(d, bins=bin_edges)
            counts += h

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if counts.sum() == 0:
        return centers, counts, float("nan")
    peak_idx = int(counts.argmax())
    return centers, counts, float(centers[peak_idx])


def _set_pair_target_r(pair_species_np, pair_features_np, z_a, z_b, new_r):
    """Return a copy of pair_features with target_r set to ``new_r`` for
    every entry whose species pair is (z_a, z_b) or (z_b, z_a)."""
    pair_features_out = pair_features_np.copy()
    mask = (
        ((pair_species_np[:, 0] == z_a) & (pair_species_np[:, 1] == z_b))
        | ((pair_species_np[:, 0] == z_b) & (pair_species_np[:, 1] == z_a))
    )
    if not mask.any():
        raise ValueError(
            f"No shell_target entry found for pair (Z={z_a}, Z={z_b}) — "
            f"check that the held-out structure actually contains both."
        )
    pair_features_out[mask, 0] = float(new_r)
    return pair_features_out, int(mask.sum())


def _weight_vector_from_npz(npz):
    fake_row = {k: str(float(npz[k])) for k in (
        "bond_weight", "angle_weight", "repulsion_weight",
        "hard_core_scale", "nonbond_push_scale", "displacement_sigma",
        "grain_size", "crystalline_fraction",
    )}
    fake_row["num_grains"] = str(int(npz["num_grains"]))
    return _weight_vector_from_row(fake_row)


def main() -> None:
    npz_path = Path(NPZ_PATH).resolve()
    if not npz_path.is_file():
        sys.exit(f"npz not found: {npz_path}")
    ckpt_path = Path(CHECKPOINT).resolve()
    if not ckpt_path.is_file():
        sys.exit(f"checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")
    print(f"Variant:    {VARIANT}")
    print(f"Checkpoint: {ckpt_path}  (EMA={USE_EMA_WEIGHTS})")
    print(f"NPZ:        {npz_path}")
    print(f"Pair (Z_a={SPECIES_A}, Z_b={SPECIES_B})")
    print(f"target_r grid: {TARGET_R_VALUES} Å")
    print(f"max_iter:   {MAX_ITER}  (tol {CONVERGENCE_TOL_ANG} Å)")
    print()

    model = _load_model(ckpt_path, device)

    npz = np.load(npz_path)
    if "shell_pair_species" not in npz.files:
        sys.exit(
            f"{npz_path.name} has no shell_target arrays.  Run "
            f"add_shell_tgt_to_npz.py first."
        )

    initial_pos = np.asarray(npz["positions"][0], dtype=np.float32)
    cell = np.asarray(npz["cell"], dtype=np.float32)
    species = np.asarray(npz["species_numbers"], dtype=np.int64)
    w_vec = _weight_vector_from_npz(npz)

    pair_species_np = np.asarray(npz["shell_pair_species"], dtype=np.int64)
    pair_features_np = np.asarray(npz["shell_pair_features"], dtype=np.float32)
    trip_species_np = np.asarray(npz["shell_triplet_species"], dtype=np.int64)
    trip_features_np = np.asarray(npz["shell_triplet_features"], dtype=np.float32)

    # Sanity: print the actual target_r in the .npz for our pair, so the
    # user can confirm they're testing meaningful perturbations.
    pair_mask = (
        ((pair_species_np[:, 0] == SPECIES_A) & (pair_species_np[:, 1] == SPECIES_B))
        | ((pair_species_np[:, 0] == SPECIES_B) & (pair_species_np[:, 1] == SPECIES_A))
    )
    if pair_mask.any():
        real_target_r = float(pair_features_np[pair_mask, 0].mean())
        print(f"Real target_r for (Z={SPECIES_A}, Z={SPECIES_B}) in this .npz: "
              f"{real_target_r:.3f} Å  ({int(pair_mask.sum())} entries)")
    else:
        sys.exit(f"No (Z={SPECIES_A}, Z={SPECIES_B}) entries in shell_target.")
    print()

    # Compute the tricor target's Si–O peak as a reference floor
    best_pos = np.asarray(npz["best_positions"], dtype=np.float32)
    _, _, tricor_peak = _ab_distance_histogram(
        best_pos, species, cell, SPECIES_A, SPECIES_B,
        HIST_R_MIN, HIST_R_MAX, HIST_BINS,
    )
    print(f"Tricor target  Si–O peak: {tricor_peak:.3f} Å  (from best_positions)")
    print()

    # Sweep perturbations
    results: list[tuple[float, float]] = []   # (target_r, predicted_peak)
    print(f"{'target_r':>10s} | {'pred peak':>10s} | {'shift vs tricor':>18s}")
    print("-" * 50)
    for tr in TARGET_R_VALUES:
        # Build a perturbed shell_pair_features
        perturbed_features, n_changed = _set_pair_target_r(
            pair_species_np, pair_features_np, SPECIES_A, SPECIES_B, tr,
        )
        shell = {
            "pair_species":  torch.tensor(pair_species_np,
                                          dtype=torch.long, device=device),
            "pair_features": torch.tensor(perturbed_features,
                                          dtype=torch.float32, device=device),
            "trip_species":  torch.tensor(trip_species_np,
                                          dtype=torch.long, device=device),
            "trip_features": torch.tensor(trip_features_np,
                                          dtype=torch.float32, device=device),
        }
        pred = _run_iterative_inference(
            model, initial_pos, cell, species, w_vec, shell, device,
            MAX_ITER, CONVERGENCE_TOL_ANG, CUTOFF,
        )
        _, _, pred_peak = _ab_distance_histogram(
            pred, species, cell, SPECIES_A, SPECIES_B,
            HIST_R_MIN, HIST_R_MAX, HIST_BINS,
        )
        results.append((tr, pred_peak))
        print(f"{tr:>10.3f} | {pred_peak:>10.3f} | "
              f"{pred_peak - tricor_peak:>+18.3f}")

    # Sensitivity slope: linear fit of pred_peak vs target_r.  Slope of
    # 1.0 = perfect tracking; 0 = model ignores target_r.
    print()
    target_r_arr = np.array([r for r, _ in results], dtype=np.float64)
    pred_peak_arr = np.array([p for _, p in results], dtype=np.float64)
    valid = np.isfinite(pred_peak_arr)
    if valid.sum() < 2:
        print("[VERDICT] Not enough valid predictions to fit a slope.")
        return
    slope, intercept = np.polyfit(target_r_arr[valid], pred_peak_arr[valid], 1)

    # R² for goodness-of-fit
    fit_vals = slope * target_r_arr[valid] + intercept
    ss_res = np.sum((pred_peak_arr[valid] - fit_vals) ** 2)
    ss_tot = np.sum((pred_peak_arr[valid] - pred_peak_arr[valid].mean()) ** 2)
    r_sq = 1.0 - ss_res / max(ss_tot, 1e-12)

    print(f"Linear fit: pred_peak = {slope:.3f} * target_r + {intercept:.3f}")
    print(f"  slope = {slope:.3f}  (ideal 1.0; 0.0 = model ignores target_r)")
    print(f"  R² = {r_sq:.3f}")
    print()

    if slope > 0.7:
        verdict = (
            f"STRONG conditioning (slope {slope:.2f}).  Model uses target_r.  "
            f"Failure on real stishovite is bond-length extrapolation — the "
            f"model can produce bonds at {pred_peak_arr.min():.2f}-"
            f"{pred_peak_arr.max():.2f} Å, but stishovite needs 1.78 Å "
            f"which is outside training-set 6-coord range.  FIX: data — "
            f"add training compounds with shorter 6-coord M-O."
        )
    elif slope > 0.3:
        verdict = (
            f"PARTIAL conditioning (slope {slope:.2f}).  Model partially uses "
            f"target_r but is also pulled by species priors.  Both data and "
            f"architecture contribute.  Try (1) higher SHELL_TARGET_DROPOUT "
            f"in retraining, (2) more diverse 6-coord training compounds, "
            f"(3) per-edge shell_target injection (architectural)."
        )
    else:
        verdict = (
            f"WEAK conditioning (slope {slope:.2f}).  Model is essentially "
            f"ignoring target_r — predictions sit near the species-prior "
            f"value regardless of conditioning.  More data won't fix this.  "
            f"FIX: architecture — per-edge shell_target injection so the "
            f"specific pair's target_r is delivered to the edge encoder for "
            f"that edge directly (GemNet-OC pattern)."
        )
    print(f"[VERDICT] {verdict}")


if __name__ == "__main__":
    main()
