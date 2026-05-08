"""Generate small per-cell-size test sets for scale-transfer evaluation.

For each cell edge in CELL_SIZES, produces N_TEST_SAMPLES_PER_CELL tricor
relaxation trajectories with balanced regime quotas, written to a separate
directory.  These are the ground-truth references used to evaluate how well
a model trained on 50 Å cells generalizes to larger structures.

Edit the CONFIG section below, then run:
    python generate_test_cells.py

Outputs land in <OUTPUT_BASE>/cell<NNN>/ with their own manifest.csv.
Use evaluate.py with TARGET pointing at one of those directories per
cell size to see RMSE / PDF / ADF on the held-out scale.
"""

from __future__ import annotations

# Limit BLAS threads per worker to 1 so multiprocessing workers don't
# oversubscribe the CPU.  Must happen BEFORE numpy is imported.
import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.build import bulk

from tricor.shells import CoordinationShellTarget
from tricor.supercell import Supercell

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

# Cell edges (Å) to test.  Add or remove as desired.  At 200 Å each tricor
# relaxation already takes ~30 min, so be deliberate.
CELL_SIZES = [80.0, 100.0, 150.0]

OUTPUT_BASE = "./data/si_test_cells"   # creates si_test_cells/cell080/, cell100/, ...
SI_LATTICE = 5.431
REL_DENSITY = 0.96

# How many test trajectories per cell size.  12 = 2 per regime, enough for
# a per-regime visual check via evaluate.py's plotting.
N_TEST_SAMPLES_PER_CELL = 12

N_STEPS_DEFAULT = 200
TRAJECTORY_STRIDE = 5

# Seed range fully separate from v1_full (131000..132999) and v1_extras
# (200000..201499).  Each cell size gets its own 10k-seed window so they
# can never collide.
BASE_SEED_OFFSET = 300_000

NUM_WORKERS = 8

WEIGHT_JITTER_SIGMA = 0.20

# Balanced quotas — we want all regimes represented at every test cell
# size to see how generalization varies across the disorder spectrum.
# NOTE: "LRO" is the upstream-renamed key (was "MRO_more").
REGIME_STRATA = [
    {"name": "liquid",          "gs_range": (0.0,  0.0),   "quota": 0.20},
    {"name": "amorphous",       "gs_range": (4.0,  8.0),   "quota": 0.16},
    {"name": "SRO",             "gs_range": (8.0,  12.0),  "quota": 0.16},
    {"name": "MRO",             "gs_range": (12.0, 15.0),  "quota": 0.16},
    {"name": "LRO",             "gs_range": (15.0, 19.0),  "quota": 0.16},
    {"name": "nanocrystalline", "gs_range": (19.0, 25.0),  "quota": 0.16},
]

# ══════════════════════════════════════════════════════════════════════════════
# Sampling (inlined so this script is self-contained)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SampleConfig:
    idx: int
    source: str
    anchor_regime: str
    grain_size: float
    num_grains: int
    n_crystalline: int
    crystalline_fraction: float
    bond_weight: float
    angle_weight: float
    repulsion_weight: float
    hard_core_scale: float
    nonbond_push_scale: float
    displacement_sigma: float
    rng_seed: int
    num_steps: int


def _num_grains_for(grain_size: float, cell_size: float) -> int:
    if grain_size <= 0.0:
        return 0
    r = max(grain_size * 0.5, 2.0)
    v_grain = (4.0 / 3.0) * np.pi * r ** 3
    v_box = float(cell_size) ** 3
    return max(1, int(np.ceil(v_box / v_grain)))


def _jittered_weights(preset: dict, rng: np.random.Generator) -> dict:
    defaults = {
        "bond_weight": 1.0,
        "angle_weight": 0.5,
        "repulsion_weight": 3.0,
        "hard_core_scale": 1.0,
        "nonbond_push_scale": 1.0,
        "displacement_sigma": 0.0,
    }
    out: dict[str, float] = {}
    for key, default in defaults.items():
        base = float(preset.get(key, default))
        if key == "displacement_sigma":
            out[key] = base
            continue
        if base <= 0.0:
            out[key] = base
            continue
        factor = float(np.exp(rng.normal(0.0, WEIGHT_JITTER_SIGMA)))
        out[key] = base * factor
    return out


def _stratum_counts(total: int) -> list[int]:
    raw = [total * s["quota"] for s in REGIME_STRATA]
    base = [int(np.floor(x)) for x in raw]
    remainder = total - sum(base)
    frac_order = sorted(
        range(len(REGIME_STRATA)),
        key=lambda i: raw[i] - base[i],
        reverse=True,
    )
    for i in frac_order[:remainder]:
        base[i] += 1
    return base


def build_test_configs(
    rng: np.random.Generator,
    cell_size: float,
    base_seed: int,
    n_total: int,
) -> list[SampleConfig]:
    """Build a stratified list of SampleConfig for one cell size."""
    counts = _stratum_counts(n_total)
    configs: list[SampleConfig] = []
    idx = 0

    for stratum, n_samples in zip(REGIME_STRATA, counts):
        regime = stratum["name"]
        gs_lo, gs_hi = stratum["gs_range"]
        gs_hi_clipped = min(gs_hi, cell_size) if gs_hi > 0 else 0.0

        preset = dict(Supercell.PRESETS[regime])
        preset.pop("relative_density", None)
        preset_num_steps = int(preset.pop("num_steps", N_STEPS_DEFAULT))

        for _ in range(n_samples):
            if gs_lo == 0.0 and gs_hi == 0.0:
                grain_size = 0.0
                num_grains = 0
                n_crystalline = 0
                cf = 0.0
            else:
                if gs_hi_clipped <= gs_lo:
                    grain_size = float(gs_lo)
                else:
                    grain_size = float(rng.uniform(gs_lo, gs_hi_clipped))
                num_grains = _num_grains_for(grain_size, cell_size)
                n_crystalline = int(rng.integers(1, num_grains + 1))
                cf = n_crystalline / num_grains

            w = _jittered_weights(preset, rng)
            configs.append(SampleConfig(
                idx=idx,
                source="test_stratified",
                anchor_regime=regime,
                grain_size=grain_size,
                num_grains=num_grains,
                n_crystalline=n_crystalline,
                crystalline_fraction=cf,
                bond_weight=w["bond_weight"],
                angle_weight=w["angle_weight"],
                repulsion_weight=w["repulsion_weight"],
                hard_core_scale=w["hard_core_scale"],
                nonbond_push_scale=w["nonbond_push_scale"],
                displacement_sigma=w["displacement_sigma"],
                rng_seed=base_seed + idx,
                num_steps=preset_num_steps,
            ))
            idx += 1
    return configs


# ══════════════════════════════════════════════════════════════════════════════
# Worker side: builds and saves one trajectory
# ══════════════════════════════════════════════════════════════════════════════


_WORKER_REF = None
_WORKER_SHELL_TARGET = None
_WORKER_CELL_SIZE: float = 0.0
_WORKER_REL_DENSITY: float = 0.0
_WORKER_OUT_DIR: Path | None = None


def _init_worker(
    si_lattice: float, cell_size: float, rel_density: float, out_dir_str: str,
) -> None:
    global _WORKER_REF, _WORKER_SHELL_TARGET
    global _WORKER_CELL_SIZE, _WORKER_REL_DENSITY, _WORKER_OUT_DIR
    _WORKER_REF = bulk("Si", crystalstructure="diamond", a=si_lattice, cubic=True)
    _WORKER_SHELL_TARGET = CoordinationShellTarget.from_atoms(_WORKER_REF)
    _WORKER_CELL_SIZE = float(cell_size)
    _WORKER_REL_DENSITY = float(rel_density)
    _WORKER_OUT_DIR = Path(out_dir_str)


def _run_one_worker(cfg: SampleConfig) -> dict | None:
    try:
        return run_trajectory(
            cfg, _WORKER_REF, _WORKER_SHELL_TARGET,
            _WORKER_CELL_SIZE, _WORKER_REL_DENSITY, _WORKER_OUT_DIR,
        )
    except Exception as e:
        print(f"[{cfg.idx+1:4d}] FAILED: {type(e).__name__}: {e}", flush=True)
        return None


def run_trajectory(
    cfg: SampleConfig,
    ref,
    shell_target,
    cell_size: float,
    rel_density: float,
    out_dir: Path,
) -> dict[str, float | int | str]:
    t0 = time.perf_counter()
    sc = Supercell.from_atoms(
        ref, cell_dim_angstroms=cell_size, rng_seed=cfg.rng_seed,
        relative_density=rel_density,
    )
    initial_positions = np.asarray(sc.atoms.positions, dtype=np.float32).copy()

    kwargs = dict(
        bond_weight=cfg.bond_weight,
        angle_weight=cfg.angle_weight,
        repulsion_weight=cfg.repulsion_weight,
        hard_core_scale=cfg.hard_core_scale,
        nonbond_push_scale=cfg.nonbond_push_scale,
        displacement_sigma=cfg.displacement_sigma,
        crystalline_fraction=cfg.crystalline_fraction,
    )
    if cfg.grain_size > 0.0:
        kwargs["grain_size"] = cfg.grain_size

    summary = sc.generate(
        shell_target,
        num_steps=cfg.num_steps,
        show_progress=False,
        save_trajectory=True,
        trajectory_stride=TRAJECTORY_STRIDE,
        **kwargs,
    )
    runtime = time.perf_counter() - t0

    h = sc.shell_relax_history
    filename = (
        f"si_{cfg.anchor_regime}_cell{int(cell_size):03d}_idx{cfg.idx:05d}_"
        f"seed{cfg.rng_seed:09d}.npz"
    )
    outfile = out_dir / filename

    np.savez(
        outfile,
        positions=h["positions"],
        snapshot_steps=h["snapshot_steps"],
        initial_positions=initial_positions,
        best_positions=h["best_positions"],
        final_positions=sc.atoms.positions.astype(np.float32),
        species_numbers=sc.atoms.numbers.astype(np.int32),
        cell=np.asarray(sc.atoms.cell.array, dtype=np.float32),
        loss_history=h["loss"],
        idx=np.int64(cfg.idx),
        source=np.asarray(cfg.source),
        regime=np.asarray(cfg.anchor_regime),
        rng_seed=np.int64(cfg.rng_seed),
        grain_size=np.float32(cfg.grain_size),
        num_grains=np.int32(cfg.num_grains),
        n_crystalline=np.int32(cfg.n_crystalline),
        crystalline_fraction=np.float32(cfg.crystalline_fraction),
        bond_weight=np.float32(cfg.bond_weight),
        angle_weight=np.float32(cfg.angle_weight),
        repulsion_weight=np.float32(cfg.repulsion_weight),
        hard_core_scale=np.float32(cfg.hard_core_scale),
        nonbond_push_scale=np.float32(cfg.nonbond_push_scale),
        displacement_sigma=np.float32(cfg.displacement_sigma),
        num_steps=np.int32(cfg.num_steps),
        trajectory_stride=np.int32(TRAJECTORY_STRIDE),
        cell_size=np.float32(cell_size),
        rel_density=np.float32(rel_density),
        initial_loss=np.float64(summary["initial_loss"]),
        best_loss=np.float64(summary["best_loss"]),
        final_loss=np.float64(summary["final_loss"]),
    )

    n_atoms = len(sc.atoms)
    size_mb = outfile.stat().st_size / (1024 * 1024)
    print(
        f"[cell={cell_size:5.0f}Å {cfg.idx+1:3d}] "
        f"{cfg.anchor_regime:>16s} "
        f"gs={cfg.grain_size:5.1f} ng={cfg.num_grains:3d} nc={cfg.n_crystalline:3d} "
        f"atoms={n_atoms:6d} "
        f"loss {summary['initial_loss']:6.2f}→{summary['final_loss']:6.2f} "
        f"best={summary['best_loss']:6.2f}  {runtime:6.1f}s  {size_mb:5.1f}MB"
    )

    return {
        "idx": int(cfg.idx),
        "source": cfg.source,
        "regime": cfg.anchor_regime,
        "rng_seed": int(cfg.rng_seed),
        "grain_size": float(cfg.grain_size),
        "num_grains": int(cfg.num_grains),
        "n_crystalline": int(cfg.n_crystalline),
        "crystalline_fraction": float(cfg.crystalline_fraction),
        "bond_weight": float(cfg.bond_weight),
        "angle_weight": float(cfg.angle_weight),
        "repulsion_weight": float(cfg.repulsion_weight),
        "hard_core_scale": float(cfg.hard_core_scale),
        "nonbond_push_scale": float(cfg.nonbond_push_scale),
        "displacement_sigma": float(cfg.displacement_sigma),
        "num_steps": int(cfg.num_steps),
        "initial_loss": float(summary["initial_loss"]),
        "best_loss": float(summary["best_loss"]),
        "final_loss": float(summary["final_loss"]),
        "num_atoms": int(n_atoms),
        "runtime_s": float(runtime),
        "filename": outfile.name,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main: loop over cell sizes
# ══════════════════════════════════════════════════════════════════════════════


def run_one_cell_size(cell_size: float, cell_idx: int) -> None:
    out_dir = Path(OUTPUT_BASE) / f"cell{int(cell_size):03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_seed = BASE_SEED_OFFSET + cell_idx * 10_000
    rng = np.random.default_rng(base_seed)
    configs = build_test_configs(
        rng, cell_size=cell_size, base_seed=base_seed,
        n_total=N_TEST_SAMPLES_PER_CELL,
    )

    print(f"\n=== cell_size = {cell_size} Å  ({len(configs)} configs) ===")
    print(f"Output: {out_dir}")
    print(f"Seeds:  {base_seed}..{base_seed + len(configs) - 1}")

    t_start = time.perf_counter()
    manifest: list[dict] = []

    if NUM_WORKERS is None or NUM_WORKERS <= 1:
        ref = bulk("Si", crystalstructure="diamond", a=SI_LATTICE, cubic=True)
        shell_target = CoordinationShellTarget.from_atoms(ref)
        for cfg in configs:
            try:
                row = run_trajectory(
                    cfg, ref, shell_target, cell_size, REL_DENSITY, out_dir,
                )
                manifest.append(row)
            except Exception as e:
                print(f"[{cfg.idx+1:4d}] FAILED: {type(e).__name__}: {e}")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=NUM_WORKERS,
            initializer=_init_worker,
            initargs=(SI_LATTICE, cell_size, REL_DENSITY, str(out_dir)),
        ) as pool:
            for row in pool.imap_unordered(_run_one_worker, configs, chunksize=1):
                if row is not None:
                    manifest.append(row)

    elapsed = time.perf_counter() - t_start

    manifest_path = out_dir / "manifest.csv"
    if manifest:
        headers = list(manifest[0].keys())
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in manifest:
                writer.writerow(row)

    print(
        f"=== cell_size = {cell_size} Å done. "
        f"{len(manifest)}/{len(configs)} succeeded in {elapsed/60:.1f} min "
        f"({elapsed/max(len(configs), 1):.1f} s/config avg) ==="
    )


def main() -> None:
    print(f"Generating test sets for cell sizes: {CELL_SIZES}")
    print(f"Per cell: {N_TEST_SAMPLES_PER_CELL} configs, {NUM_WORKERS} workers")
    print(f"Output base: {OUTPUT_BASE}")

    overall_start = time.perf_counter()
    for cell_idx, cell_size in enumerate(CELL_SIZES):
        run_one_cell_size(cell_size, cell_idx)
    total = time.perf_counter() - overall_start
    print(f"\nAll cell sizes done.  Total wall: {total/60:.1f} min")


if __name__ == "__main__":
    main()
