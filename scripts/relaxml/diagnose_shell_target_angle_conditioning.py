"""Diagnostic: does the trained model use TRIPLET ANGLES from shell_target?

Companion to ``diagnose_shell_target_conditioning.py`` (which perturbs
the per-pair target_r and got slope=0 on the dropout=0.2 phys checkpoint).
THIS one perturbs the per-triplet angle target — testing whether the
model is sensitive to a different shell_target dimension.

Why this matters: the pair-target_r diagnostic returned slope=0,
suggesting the encoder ignores per-pair distances.  But the model
DID generalize to held-out coesite (cross-polymorph) — which means
SOMETHING in shell_target carries signal.  Hypothesis: the encoder
attends to features that varied during training, and ignores those
that were constant.  In SiO2 training (α-quartz, α-cristobalite,
β-cristobalite), Si–O target_r was ~constant (~1.61 Å) but the
Si–O–Si and O–Si–O angles varied substantially across polymorphs.
So the encoder may have learned to attend to angles, not distances.

Test design: forward stishovite at several artificial angle values
for the (Z_a, Z_b, Z_c) triplet of interest (default: O–Si–O,
which is 90°+180° in 6-coord stishovite vs 109.5° in tetrahedral
SiO2 polymorphs).  Other features unchanged.  Run iterative
inference for each.  Measure where the predicted angle in the
relaxed structure's ADF (or related geometric quantity like
predicted Si–O–Si angle distribution) lands.

For a quick proxy of "did the angles of the predicted structure
respond" we use the Si–Si nearest-neighbor distance histogram —
in SiO2, Si–Si distance is set by the Si–O–Si angle (Si–Si =
2 × d_SiO × sin(angle/2)).  Shorter Si–O–Si → shorter Si–Si.
For 6-coord stishovite the Si–Si is ~3.0 Å (Si–O–Si ~98°);
for 4-coord α-quartz it's ~3.07 Å (Si–O–Si ~144°).  Wider range
than the angle directly via histogram bin centers.

Headline metric: SENSITIVITY SLOPE = Δpeak_predicted / Δangle_perturbed.
  * |slope| > some threshold → encoder uses the perturbed feature
  * slope ≈ 0 → encoder ignores it

Combined with the pair-target_r result, this diagnoses WHICH
features of shell_target the encoder actually attends to.

Run on whichever env has graphite + ase + a CUDA GPU available:
    python scripts/relaxml/diagnose_shell_target_angle_conditioning.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

GPU_ID = 3
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
    "/home/ehrdt/tricor/scripts/relaxml/shelltgt_phys/"
    "lightning_logs/coord_test_z/version_0/checkpoints/last.ckpt"
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

# Perturbation grid for triplet angle (in DEGREES, converted to radians
# internally).  Defaults bracket SiO2-relevant geometry:
#
#    60° — small-angle / under-coordinated
#    90° — octahedral O–Si–O (TRUE stishovite)
#   109.5° — tetrahedral O–Si–O (4-coord training)
#   144° — Si–O–Si in α-quartz (typical for SiO2 polymorph training)
#   170° — near-linear, never-trained extreme
#
# Want at least 3 values to fit a slope.  Adjust if probing a different
# triplet (e.g. for Si–O–Si variability, use 100°-160° range).
TARGET_ANGLE_DEG_VALUES = [60.0, 90.0, 109.5, 144.0, 170.0]

# Inference parameters.  Match evaluate.py for consistency.
MAX_ITER = 50
CONVERGENCE_TOL_ANG = 0.01
USE_EMA_WEIGHTS = True
CUTOFF = 5.0

# Triplet to perturb.  shell_triplet_species rows are stored as
# (Z_centre, Z_neighbour_1, Z_neighbour_2) — the CENTRE atom is at
# index 0, not index 1.  For O–Si–O (octahedral angle in stishovite),
# centre is Si and both neighbours are O:
# TRIPLET_CENTRE = 14    # Si (centre of the angle)
# TRIPLET_NBR_A = 8      # O (one neighbour)
# TRIPLET_NBR_B = 8      # O (other neighbour)
TRIPLET_CENTRE = 8     # O (centre)
TRIPLET_NBR_A = 14     # Si
TRIPLET_NBR_B = 14     # Si

# To probe Si–O–Si instead (the angle that varies most across SiO2
# polymorphs in training): TRIPLET_CENTRE=8, TRIPLET_NBR_A=TRIPLET_NBR_B=14.

# Measurement: we want to know if the predicted structure's GEOMETRY
# responds to the perturbed angle.  Two cheap proxies on the predicted
# positions, computed without needing the full ADF compute:
#
#  (1) Pair-distance peak between the two NEIGHBOUR atoms in the triplet
#      (e.g. O–O distance for O–Si–O).  In a fixed bond length d, the
#      neighbour–neighbour distance equals 2*d*sin(angle/2).  So if the
#      model responds to the perturbed angle, the O–O peak should shift
#      monotonically with the perturbation.  Most sensitive geometric
#      proxy with the lowest implementation cost.
#
#  (2) For TRIPLET_NBR_A == TRIPLET_NBR_B (symmetric-neighbour case
#      like O–Si–O): the neighbour–neighbour distance is a clean
#      proxy.
#
#  (3) For asymmetric triplets (e.g. O-Si-N): just measure the angle
#      distribution directly — costlier; left as a follow-up.
#
# Histogram for the neighbour-neighbour distance:
HIST_R_MIN = 1.5
HIST_R_MAX = 4.0
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


def _set_triplet_angle(trip_species_np, trip_features_np,
                        z_centre, z_nbr_a, z_nbr_b, new_angle_rad):
    """Return a copy of trip_features with angle_mode_rad set to
    ``new_angle_rad`` for every entry matching the (centre, neighbours)
    triplet.

    shell_target stores triplets as (Z_centre, Z_neighbour_1,
    Z_neighbour_2) — centre at index 0.  Matching is symmetric in the
    two neighbour slots: {Z_nbr_a, Z_nbr_b} == {trip[1], trip[2]}.
    """
    trip_features_out = trip_features_np.copy()
    centre_match = trip_species_np[:, 0] == z_centre
    nbr_match = (
        ((trip_species_np[:, 1] == z_nbr_a) & (trip_species_np[:, 2] == z_nbr_b))
        | ((trip_species_np[:, 1] == z_nbr_b) & (trip_species_np[:, 2] == z_nbr_a))
    )
    mask = centre_match & nbr_match
    if not mask.any():
        raise ValueError(
            f"No shell_target entry for triplet centred at Z={z_centre} "
            f"with neighbours ({z_nbr_a},{z_nbr_b}) — check shell_target "
            f"contents.  (Convention: trip_species[:, 0] = centre.)"
        )
    trip_features_out[mask, 0] = float(new_angle_rad)
    return trip_features_out, int(mask.sum())


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
    print(f"Triplet centre Z={TRIPLET_CENTRE}, neighbours "
          f"(Z={TRIPLET_NBR_A}, Z={TRIPLET_NBR_B})")
    print(f"angle grid (deg): {TARGET_ANGLE_DEG_VALUES}")
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

    # Sanity: confirm the target triplet exists in shell_target and
    # report its actual angle so the user knows whether perturbations
    # span real territory or extrapolate.
    centre_match = trip_species_np[:, 0] == TRIPLET_CENTRE
    nbr_match = (
        ((trip_species_np[:, 1] == TRIPLET_NBR_A)
         & (trip_species_np[:, 2] == TRIPLET_NBR_B))
        | ((trip_species_np[:, 1] == TRIPLET_NBR_B)
           & (trip_species_np[:, 2] == TRIPLET_NBR_A))
    )
    trip_mask = centre_match & nbr_match
    if trip_mask.any():
        real_angle_rad = float(trip_features_np[trip_mask, 0].mean())
        real_angle_deg = np.degrees(real_angle_rad)
        print(f"Real angle for triplet centred Z={TRIPLET_CENTRE} with "
              f"neighbours ({TRIPLET_NBR_A},{TRIPLET_NBR_B}) "
              f"in this .npz: {real_angle_deg:.2f}°  "
              f"({int(trip_mask.sum())} entries)")
    else:
        sys.exit(
            f"No triplet centred at Z={TRIPLET_CENTRE} with neighbours "
            f"({TRIPLET_NBR_A},{TRIPLET_NBR_B}) in shell_target."
        )
    print()

    # We measure the predicted neighbour–neighbour distance histogram
    # peak for the (TRIPLET_NBR_A, TRIPLET_NBR_B) pair.  In a fixed
    # bond length d_centre-nbr, this peak shifts with the perturbed
    # angle as d_NN = 2 * d * sin(angle/2).  Tricor reference: predicted
    # neighbour-neighbour peak from best_positions.
    best_pos = np.asarray(npz["best_positions"], dtype=np.float32)
    _, _, tricor_peak = _ab_distance_histogram(
        best_pos, species, cell, TRIPLET_NBR_A, TRIPLET_NBR_B,
        HIST_R_MIN, HIST_R_MAX, HIST_BINS,
    )
    print(f"Tricor target neighbour–neighbour peak "
          f"(Z={TRIPLET_NBR_A}-Z={TRIPLET_NBR_B}): "
          f"{tricor_peak:.3f} Å  (from best_positions)")
    print()

    # Sweep angle perturbations
    results: list[tuple[float, float]] = []   # (angle_deg, predicted_peak_Å)
    print(f"{'angle (deg)':>12s} | {'pred d_NN':>12s} | "
          f"{'shift vs tricor':>18s}")
    print("-" * 52)
    for ang_deg in TARGET_ANGLE_DEG_VALUES:
        ang_rad = float(np.radians(ang_deg))
        perturbed_trip_features, n_changed = _set_triplet_angle(
            trip_species_np, trip_features_np,
            TRIPLET_CENTRE, TRIPLET_NBR_A, TRIPLET_NBR_B, ang_rad,
        )
        shell = {
            "pair_species":  torch.tensor(pair_species_np,
                                          dtype=torch.long, device=device),
            "pair_features": torch.tensor(pair_features_np,
                                          dtype=torch.float32, device=device),
            "trip_species":  torch.tensor(trip_species_np,
                                          dtype=torch.long, device=device),
            "trip_features": torch.tensor(perturbed_trip_features,
                                          dtype=torch.float32, device=device),
        }
        pred = _run_iterative_inference(
            model, initial_pos, cell, species, w_vec, shell, device,
            MAX_ITER, CONVERGENCE_TOL_ANG, CUTOFF,
        )
        _, _, pred_peak = _ab_distance_histogram(
            pred, species, cell, TRIPLET_NBR_A, TRIPLET_NBR_B,
            HIST_R_MIN, HIST_R_MAX, HIST_BINS,
        )
        results.append((ang_deg, pred_peak))
        print(f"{ang_deg:>12.3f} | {pred_peak:>12.3f} | "
              f"{pred_peak - tricor_peak:>+18.3f}")

    # Sensitivity slope: linear fit of pred d_AC vs perturbed angle.
    # If model uses the angle, we expect pred_peak ≈ 2*d_AB*sin(ang/2),
    # which is monotonic but nonlinear.  A simple linear-fit slope captures
    # whether there's ANY responsiveness; it'll be roughly d_AB*cos(real_angle/2)
    # in radian units.  Convert: if angle perturbed in DEGREES and pred in Å,
    # slope is in Å/deg.  An ideal angle-following model would have slope of
    # d_AB * cos(real_angle/2) * (π/180).
    print()
    angle_arr = np.array([a for a, _ in results], dtype=np.float64)
    pred_peak_arr = np.array([p for _, p in results], dtype=np.float64)
    valid = np.isfinite(pred_peak_arr)
    if valid.sum() < 2:
        print("[VERDICT] Not enough valid predictions to fit a slope.")
        return
    slope, intercept = np.polyfit(angle_arr[valid], pred_peak_arr[valid], 1)

    # R² for goodness-of-fit
    fit_vals = slope * angle_arr[valid] + intercept
    ss_res = np.sum((pred_peak_arr[valid] - fit_vals) ** 2)
    ss_tot = np.sum((pred_peak_arr[valid] - pred_peak_arr[valid].mean()) ** 2)
    r_sq = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Reference: ideal slope if model perfectly tracks angle.  For the
    # geometric relation d_NN = 2 * d_centre-nbr * sin(angle/2), the
    # local linear slope around the real angle is
    # d_centre-nbr * cos(real_angle/2) * (π/180) (Å per degree).  Use
    # the (centre, neighbour) bond length from shell_target's
    # pair_features.  Match either (centre, nbr) or (nbr, centre) since
    # pair ordering in shell_target is symmetric.
    ab_pair_mask = (
        ((pair_species_np[:, 0] == TRIPLET_CENTRE) & (pair_species_np[:, 1] == TRIPLET_NBR_A))
        | ((pair_species_np[:, 0] == TRIPLET_NBR_A) & (pair_species_np[:, 1] == TRIPLET_CENTRE))
    )
    if ab_pair_mask.any():
        d_ab = float(pair_features_np[ab_pair_mask, 0].mean())
    else:
        d_ab = float("nan")
    ideal_slope = d_ab * np.cos(real_angle_rad / 2) * (np.pi / 180)

    print(f"Linear fit: pred d_AC = {slope:+.5f} * angle_deg + {intercept:.4f}")
    print(f"  slope    = {slope:+.5f} Å/deg")
    print(f"  ideal    = {ideal_slope:+.5f} Å/deg "
          f"(geometry-perfect: d_AB={d_ab:.3f} Å, "
          f"d/dθ[2*d_AB*sin(θ/2)] at real angle)")
    if abs(ideal_slope) > 1e-6:
        ratio = slope / ideal_slope
        print(f"  ratio    = {ratio:+.3f}  (1.0 = perfect tracking; 0.0 = ignored)")
    else:
        ratio = float("nan")
    print(f"  R²       = {r_sq:.3f}")
    print()

    abs_ratio = abs(ratio) if np.isfinite(ratio) else 0.0
    if abs_ratio > 0.7:
        verdict = (
            f"STRONG ANGLE conditioning (ratio {ratio:.2f} of ideal).  "
            f"The encoder propagates triplet angle perturbations to "
            f"predicted geometry.  This contrasts with the pair-target_r "
            f"diagnostic (slope=0): the encoder learned to attend to "
            f"angles, NOT pair distances.  Likely because angles VARIED "
            f"across SiO2 polymorph training while Si–O target_r was "
            f"~constant.  Fix for stishovite Si–O bond length still has "
            f"to be architectural (per-edge shell_target injection)."
        )
    elif abs_ratio > 0.3:
        verdict = (
            f"PARTIAL ANGLE conditioning (ratio {ratio:.2f}).  Encoder "
            f"uses triplet angles but only partially.  Combined with the "
            f"pair-target_r=0 result, this suggests the encoder learned "
            f"some shell_target dimensions better than others — likely "
            f"correlated with which features VARIED in training."
        )
    else:
        verdict = (
            f"WEAK ANGLE conditioning (ratio {ratio:.2f}).  Combined with "
            f"slope=0 on the pair diagnostic, this means the encoder is "
            f"essentially inert.  Coesite generalization must have come "
            f"from a different signal (initial-positions geometric "
            f"features, or species-pair embedding alone).  Architectural "
            f"fix is the only path forward — per-edge shell_target injection "
            f"so geometry targets reach the prediction directly."
        )
    print(f"[VERDICT] {verdict}")


if __name__ == "__main__":
    main()
