"""Iterate per-system subdirs in a curated eval set, run the existing
``shelltgt_phys/evaluate.py`` pipeline on each, and save plots/XYZ to a
user-specified output root.

Workflow:
    1. ``curate_eval_set.py`` first builds an eval root with per-system
       subdirs and retrofits shell_target arrays.
    2. This script loads the model once, then loops over each system,
       running ``evaluate_one()`` per trajectory and writing plots to
       ``OUTPUT_ROOT / <system>/{plots,xyz}/``.

Reuses ``evaluate.py``'s functions (``_load_model``, ``_select_files``,
``evaluate_one``) so behavior matches running ``evaluate.py`` per-subdir
manually — but the model is loaded once across all systems and outputs
are organized per-system.

Edit the CONFIG block below, then run:
    python scripts/relaxml/eval_systems.py
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these (MUST be before torch/CUDA imports)
# ─────────────────────────────────────────────────────────────────────────────

# Single GPU is enough for evaluation.  Pick a free one.
GPU_ID = 0
NUM_THREADS = 2

# Path to the trained model checkpoint.
CHECKPOINT = (
    "/home/ehrdt/tricor/scripts/relaxml/shelltgt_phys/lightning_logs/"
    "516_struct/version_4/checkpoints/last.ckpt"
)

# Root of the curated eval set (output of curate_eval_set.py).  Must
# contain per-system subdirs ``<compound>_<mp_id>_trajectories/`` whose
# .npz files have shell_target arrays already retrofitted.
EVAL_SET_ROOT = "/wigeon/users/ehrdt/prod/eval_set_v1"

# Output root for plots and XYZ trajectories.  Each system gets a
# subdir under this root containing ``plots/`` and ``xyz/``.
OUTPUT_ROOT = "/home/ehrdt/tricor/eval_results/516_struct_v1"

# When True, evaluate one sample per regime (6 total per system, fast).
# When False, evaluate every .npz in the system subdir (20 per system,
# slower but full per-regime statistics).
SAMPLE_PER_REGIME = True

# When True, save .xyz multi-frame trajectories alongside the .png plots.
# Each .xyz animates initial → ML iter 1..N → tricor target in OVITO.
SAVE_XYZ = True

# Use EMA-averaged weights at inference (typically gives slightly better
# results than the raw last-step weights).
USE_EMA_WEIGHTS = True

# ─────────────────────────────────────────────────────────────────────────────
# Apply env BEFORE importing torch / evaluate.py
# ─────────────────────────────────────────────────────────────────────────────

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
_n = str(NUM_THREADS)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _n

import sys
import time
from pathlib import Path

# Add the shelltgt_phys script dir to sys.path so we can import evaluate.py.
_HERE = Path(__file__).resolve().parent
_EVAL_DIR = _HERE / "shelltgt_phys"
sys.path.insert(0, str(_EVAL_DIR))

import torch
torch.set_num_threads(NUM_THREADS)

# Import the existing eval pipeline.  evaluate.py reads several
# module-level constants (MAX_ITER, COMPUTE_G3, etc.) inside
# ``evaluate_one`` — its defaults are what we want, so we don't override.
# We only patch the ones that affect the per-call dispatch.
import evaluate as ev

# Patch the toggles the wrapper actually cares about.  All other
# behavior (PDF / ADF / g3 / XYZ stride / convergence tol / finetune
# steps / etc.) inherits from evaluate.py's defaults — keep them in
# sync by editing evaluate.py once.
ev.SAMPLE_PER_REGIME = bool(SAMPLE_PER_REGIME)
ev.SAVE_PLOTS = True
ev.SAVE_XYZ = bool(SAVE_XYZ)
ev.USE_EMA_WEIGHTS = bool(USE_EMA_WEIGHTS)


def main() -> None:
    eval_root = Path(EVAL_SET_ROOT).resolve()
    output_root = Path(OUTPUT_ROOT).resolve()
    ckpt = Path(CHECKPOINT).resolve()

    if not eval_root.is_dir():
        raise SystemExit(f"EVAL_SET_ROOT not found: {eval_root}")
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    system_dirs = sorted(
        d for d in eval_root.iterdir()
        if d.is_dir() and d.name.endswith("_trajectories")
    )
    if not system_dirs:
        raise SystemExit(
            f"No ``*_trajectories`` subdirs under {eval_root}; run "
            f"curate_eval_set.py first."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt}  (EMA={USE_EMA_WEIGHTS})")
    print(f"Eval set:   {eval_root}  ({len(system_dirs)} systems)")
    print(f"Output:     {output_root}")
    print(f"Mode:       {'one-per-regime' if SAMPLE_PER_REGIME else 'all-trajectories'}")
    print()

    # Load the model once for the whole sweep.
    model = ev._load_model(ckpt, device)

    grand_totals: list[dict] = []
    t_grand = time.perf_counter()

    for sys_idx, sys_dir in enumerate(system_dirs, start=1):
        sys_name = sys_dir.name[: -len("_trajectories")]
        out_sys = output_root / sys_name
        plot_dir = out_sys / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        xyz_dir = out_sys / "xyz" if SAVE_XYZ else None
        if xyz_dir is not None:
            xyz_dir.mkdir(parents=True, exist_ok=True)

        npz_files = ev._select_files(sys_dir)
        if not npz_files:
            print(f"[{sys_idx}/{len(system_dirs)}] {sys_name}: no .npz files, skipped")
            continue

        print(f"[{sys_idx}/{len(system_dirs)}] {sys_name}: {len(npz_files)} file(s)")
        t_sys = time.perf_counter()

        results = []
        for p in npz_files:
            try:
                r = ev.evaluate_one(
                    model, p, device, plot_dir=plot_dir, xyz_dir=xyz_dir,
                )
                r["system"] = sys_name
                results.append(r)
            except Exception as e:
                print(f"   {p.name}: FAILED: {type(e).__name__}: {e}")

        dt_sys = time.perf_counter() - t_sys
        if results:
            import numpy as np
            rmses = np.array([r["rmse_ang"] for r in results])
            print(
                f"   done in {dt_sys:.0f}s.  "
                f"RMSE mean={rmses.mean():.3f} Å  median={np.median(rmses):.3f} Å  "
                f"max={rmses.max():.3f} Å"
            )
        grand_totals.extend(results)

    dt_grand = time.perf_counter() - t_grand
    print()
    print(f"Grand total: {len(grand_totals)} trajectories in {dt_grand:.0f}s")
    if grand_totals:
        import numpy as np
        rmses = np.array([r["rmse_ang"] for r in grand_totals])
        print(
            f"Overall RMSE mean={rmses.mean():.3f} Å  median={np.median(rmses):.3f} Å  "
            f"max={rmses.max():.3f} Å"
        )
        # Per-system summary
        from collections import defaultdict
        by_sys: dict[str, list[float]] = defaultdict(list)
        for r in grand_totals:
            by_sys[r["system"]].append(r["rmse_ang"])
        print()
        print(f"Per-system RMSE (Å):")
        for sn in sorted(by_sys):
            arr = np.array(by_sys[sn])
            print(
                f"   {sn:<28s}  mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
                f"max={arr.max():.3f}  n={len(arr)}"
            )


if __name__ == "__main__":
    main()
