"""Diagnostic: does the trained model condition on the weight vector?

The weight vector (9-D) carries `grain_size`, `crystalline_fraction`,
and `num_grains` — the features that distinguish "nanocrystalline" from
"amorphous" at the same composition.  If the model ignores them, every
regime collapses to the same prediction, and the most-common regime
in training (amorphous / liquid) dominates — exactly the failure mode
that explains why nanocrystalline grains wash out in evaluate.py xyz
animations.

This script forward-passes a real nanocrystalline trajectory's first
snapshot twice:

  * Forward A: real weight vector (high crystalline_fraction, sensible
    num_grains, real grain_size).
  * Forward B: spoofed weight vector — the regime-distinguishing
    features are zeroed (grain_size=0, crystalline_fraction=0,
    num_grains=0), simulating "no grains, no crystallinity" while
    keeping the relaxation-dynamics weights (bond_weight,
    angle_weight, etc.) unchanged.

Then compares the predicted displacements.  Headline metric:

    ratio = |delta_real - delta_spoof|.mean() / |delta_real|.mean()

  * ratio > 0.10  → model IS using the weight vector.  Amorphization
                    drift is from a different cause (try MAX_ITER tests).
  * 0.01 < ratio < 0.10 → weak conditioning.  Some signal but not
                          enough to preserve grains.
  * ratio < 0.01  → model is essentially ignoring the weight vector.
                    Retrain with classifier-free-guidance dropout on
                    the weight channel (analog of SHELL_TARGET_DROPOUT).

This is the weight-vector analog of step 2 in smoke_test_shelltgt_phys.py
(the shell_target perturbation check that confirmed the shell_target
encoder is live).

Run on mallard / wherever your env has graphite installed:
    python scripts/relaxml/diagnose_weight_conditioning.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

GPU_ID = 1
NUM_THREADS = 2

# Which variant's model to load.  "phys" -> shelltgt_phys (physics-feature
# SpeciesEncoder); "baseline" -> the plain nn.Embedding model_shelltgt.
VARIANT = "baseline"     # "phys" or "baseline"

# Checkpoint path for the chosen variant.
CHECKPOINT = "./lightning_logs/si-n_Zembed/version_0/checkpoints/last.ckpt"

# Path to a single .npz file to probe.  Pick a NANOCRYSTALLINE trajectory
# — that's the regime whose distinguishing features (grain_size etc.) we
# most want the model to be using.  Fallback to any well-defined regime
# if no nanocrystalline file is available for the chosen compound.
NPZ_PATH = (
    "./data/multi_species_v1/Si3N4_trajectories/"
    "Si3N4_nanocrystalline_cell050_idx00636_seed000800636.npz"
)

CUTOFF = 5.0
USE_EMA_WEIGHTS = True

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
else:
    raise SystemExit(f"VARIANT must be 'phys' or 'baseline', got {VARIANT!r}")

from tricor.relaxml.data_shelltgt import (
    NUM_WEIGHT_FEATURES, ShellTargetData, WEIGHT_FEATURE_KEYS,
    _weight_vector_from_row,
)


def _load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Mirror evaluate.py's checkpoint loader: strip the torch.compile
    ``._orig_mod.`` prefix and return the EMA module if requested."""
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


def _forward(model, batch):
    return model(
        batch.z, batch.edge_index, batch.edge_attr,
        batch.w, batch.batch,
        batch.shell_pair_species, batch.shell_pair_features,
        batch.shell_pair_batch,
        batch.shell_trip_species, batch.shell_trip_features,
        batch.shell_trip_batch,
    )


def _row_from_npz(npz, *, spoof: bool) -> dict:
    """Build the input dict for ``_weight_vector_from_row``.

    When ``spoof=True``, zero the regime-distinguishing features
    (grain_size, crystalline_fraction, num_grains) while leaving the
    dynamics weights unchanged.  This isolates the "is the model
    conditioning on the regime?" question from "is the model
    conditioning on any weight feature at all?".
    """
    row = {k: str(float(npz[k])) for k in (
        "bond_weight", "angle_weight", "repulsion_weight",
        "hard_core_scale", "nonbond_push_scale", "displacement_sigma",
        "grain_size", "crystalline_fraction",
    )}
    row["num_grains"] = str(int(npz["num_grains"]))
    if spoof:
        row["grain_size"] = "0.0"
        row["crystalline_fraction"] = "0.0"
        row["num_grains"] = "0"
    return row


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
    print()

    model = _load_model(ckpt_path, device)

    npz = np.load(npz_path)
    if "shell_pair_species" not in npz.files:
        sys.exit(
            f"{npz_path.name} has no shell_target arrays.  Run "
            f"add_shell_tgt_to_npz.py first."
        )

    pos = torch.tensor(npz["positions"][0], dtype=torch.float32, device=device)
    cell = torch.tensor(npz["cell"], dtype=torch.float32, device=device)
    species = torch.tensor(npz["species_numbers"], dtype=torch.long, device=device)
    shell = {
        "pair_species":  torch.tensor(np.asarray(npz["shell_pair_species"]),
                                      dtype=torch.long, device=device),
        "pair_features": torch.tensor(np.asarray(npz["shell_pair_features"]),
                                      dtype=torch.float32, device=device),
        "trip_species":  torch.tensor(np.asarray(npz["shell_triplet_species"]),
                                      dtype=torch.long, device=device),
        "trip_features": torch.tensor(np.asarray(npz["shell_triplet_features"]),
                                      dtype=torch.float32, device=device),
    }
    regime = str(npz["regime"].item())

    # Build both weight vectors.  These are 9-D numpy arrays from
    # _weight_vector_from_row, which already applies the per-feature
    # scaling from WEIGHT_FEATURE_SCALES so they're in the same space
    # the model trained on.
    w_real_np = _weight_vector_from_row(_row_from_npz(npz, spoof=False))
    w_spoof_np = _weight_vector_from_row(_row_from_npz(npz, spoof=True))

    print(f"Regime in source file: {regime}")
    print()
    print("Weight vector comparison (after scaling):")
    print(f"  {'feature':>22s}  {'real':>12s}  {'spoofed':>12s}  {'Δ':>10s}")
    for i, name in enumerate(WEIGHT_FEATURE_KEYS):
        delta = w_spoof_np[i] - w_real_np[i]
        print(f"  {name:>22s}  {w_real_np[i]:12.5f}  "
              f"{w_spoof_np[i]:12.5f}  {delta:10.5f}")
    print()

    w_real = torch.tensor(w_real_np, dtype=torch.float32, device=device)
    w_spoof = torch.tensor(w_spoof_np, dtype=torch.float32, device=device)

    # Forward A: real weights
    batch_real = _build_batch(pos, cell, species, w_real, shell, CUTOFF)
    with torch.no_grad():
        delta_real = _forward(model, batch_real)
    assert torch.isfinite(delta_real).all(), "NaN/Inf in real-weight output"

    # Forward B: spoofed weights (same positions, shell_target, species)
    batch_spoof = _build_batch(pos, cell, species, w_spoof, shell, CUTOFF)
    with torch.no_grad():
        delta_spoof = _forward(model, batch_spoof)
    assert torch.isfinite(delta_spoof).all(), "NaN/Inf in spoofed-weight output"

    # Per-atom norms and the difference.  Reported in Å (same units as
    # the predicted displacement).
    n_real = delta_real.norm(dim=-1)
    n_spoof = delta_spoof.norm(dim=-1)
    n_diff = (delta_real - delta_spoof).norm(dim=-1)

    real_mean = n_real.mean().item()
    spoof_mean = n_spoof.mean().item()
    diff_mean = n_diff.mean().item()
    real_max = n_real.max().item()
    diff_max = n_diff.max().item()

    ratio_mean = diff_mean / max(real_mean, 1e-12)
    ratio_max = diff_max / max(real_max, 1e-12)

    print("Predicted per-atom displacement (Å):")
    print(f"  |delta_real|       mean={real_mean:.4f}  max={real_max:.4f}")
    print(f"  |delta_spoof|      mean={spoof_mean:.4f}  max={n_spoof.max().item():.4f}")
    print(f"  |Δreal - Δspoof|   mean={diff_mean:.4f}  max={diff_max:.4f}")
    print()
    print(f"Ratio (mean): {ratio_mean:.4f}  ({ratio_mean*100:.2f}%)")
    print(f"Ratio (max):  {ratio_max:.4f}  ({ratio_max*100:.2f}%)")
    print()

    if ratio_mean < 0.01:
        verdict = (
            "Model is essentially IGNORING the regime features of the "
            "weight vector (mean ratio < 1%).  The grain_size / "
            "crystalline_fraction / num_grains channel is inert at "
            "inference.  This is consistent with the observed "
            "amorphization drift in xyz animations — the model treats "
            "every regime the same, so the most-common regime in "
            "training (amorphous) dominates.  Fix: retrain with "
            "classifier-free-guidance dropout on the weight vector "
            "(analog of SHELL_TARGET_DROPOUT)."
        )
    elif ratio_mean < 0.10:
        verdict = (
            "WEAK conditioning.  The weight vector affects predictions "
            "(mean ratio {0:.1%}) but only marginally.  Some signal is "
            "getting through but not enough to keep nanocrystalline "
            "grains intact over 50 inference iterations.  Likely "
            "compounds with iteration overshoot — recommend the "
            "MAX_ITER experiment as the next check, then retrain with "
            "stronger weight-channel conditioning if iteration count "
            "isn't the dominant cause.".format(ratio_mean)
        )
    else:
        verdict = (
            "Model IS using the weight vector (mean ratio {0:.1%}).  "
            "The amorphization drift is from a different cause — try "
            "the MAX_ITER experiments next to see whether iteration "
            "overshoot is responsible.".format(ratio_mean)
        )
    print(f"[VERDICT] {verdict}")


if __name__ == "__main__":
    main()
