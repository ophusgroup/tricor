"""Generate Si relaxation trajectories for training a GNN surrogate of Supercell.generate.

Uses the stratified sampler from diagnose_param_space (balanced regime coverage
with n_crystalline >= 1 for crystalline strata).  Each sample yields one .npz
with the full shell_relax trajectory, initial/final/best positions, species,
cell, all weight parameters, grain_size/cf/num_grains/n_crystalline, and the
initial/best/final losses as metadata.  A manifest CSV indexes every sample.

No quality gate is applied here: the surrogate should learn tricor's full
output distribution, including noisy/hard configs.  Downstream datasets that
use the surrogate for large-scale structure generation should filter by
best_loss (or a structure-based metric) at that stage instead.

Edit the CONFIG section below, then run:
    python generate_surrogate_trajectories.py
"""

from __future__ import annotations

# Limit BLAS threads per worker to 1 so that multiprocessing workers don't
# oversubscribe the CPU.  Must happen BEFORE numpy is imported, and applies
# to every forked/spawned worker as well since they inherit env vars.
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

# Supplemental extras run: distinct configs concentrated at the disorder
# extremes (liquid + nanocrystalline) where the surrogate currently shows
# the most regression-to-the-mean error.  Combined with the existing
# v1_full dataset this rebalances training mass toward the regimes the
# model needs the most help on.
OUTPUT_DIR = "./data/si_trajectories_v2_extras"
SI_LATTICE = 5.431
CELL_SIZE = 50.0                # Å — match the existing training data
REL_DENSITY = 0.96

# 1500 new structures, all stratified (no new preset anchors needed —
# we have those from the original run).
N_PRESET_SAMPLES = 0
N_STRATIFIED_SAMPLES = 1500

N_STEPS_DEFAULT = 200
TRAJECTORY_STRIDE = 5           # save every 5th step (≈ 40 snapshots / run)
# BASE_SEED bumped to a non-overlapping range so the new configs are
# genuinely distinct from the v1_full seeds (131000..132999).  Filenames
# include the seed, so the new .npz files won't collide with old ones
# even if they share the same regime + idx.
BASE_SEED = 200_000

# Multiprocessing.  None = serial (for debugging).  Integer = number of
# worker processes.  A good default is half the physical cores to leave
# room for other tasks.
NUM_WORKERS= 8 #: int | None = max(1, (os.cpu_count() or 2) // 2)

WEIGHT_JITTER_SIGMA = 0.20         # log-normal sigma on weight params

# 50/50 split between the two extremes — middle regimes get zero
# additional samples because they're already well-represented in v1_full.
REGIME_STRATA = [
    {"name": "liquid",          "gs_range": (0.0,  0.0),   "quota": 0.50}, #before: quota: 20
    {"name": "amorphous",       "gs_range": (4.0,  8.0),   "quota": 0.00}, #before: quoat: 16
    {"name": "SRO",             "gs_range": (8.0,  12.0),  "quota": 0.00}, #before: quoat: 16
    {"name": "MRO",             "gs_range": (12.0, 15.0),  "quota": 0.00}, #before: quoat: 16
    {"name": "LRO",        "gs_range": (15.0, 19.0),  "quota": 0.00}, #before: quoat: 16
    {"name": "nanocrystalline", "gs_range": (19.0, 25.0),  "quota": 0.50}, #before: quoat: 16
]

# ══════════════════════════════════════════════════════════════════════════════
# Sampling (inlined from diagnose_param_space so this script is standalone)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SampleConfig:
    idx: int
    source: str                     # "preset" or "stratified"
    anchor_regime: str
    grain_size: float               # 0.0 means None
    num_grains: int                 # ceil(V_box / V_grain); 0 when grain_size == 0
    n_crystalline: int              # number of grains labelled crystalline
    crystalline_fraction: float     # == n_crystalline / num_grains
    bond_weight: float
    angle_weight: float
    repulsion_weight: float
    hard_core_scale: float
    nonbond_push_scale: float
    displacement_sigma: float
    rng_seed: int
    num_steps: int


def _num_grains_for(grain_size: float, cell_size: float) -> int:
    """Replicate tricor's num_grains formula: ceil(V_box / V_grain)."""
    if grain_size <= 0.0:
        return 0
    r = max(grain_size * 0.5, 2.0)
    v_grain = (4.0 / 3.0) * np.pi * r ** 3
    v_box = float(cell_size) ** 3
    return max(1, int(np.ceil(v_box / v_grain)))


def _jittered_weights(preset: dict, rng: np.random.Generator) -> dict:
    """Take preset weights, apply log-normal jitter, return a dict usable by generate()."""
    defaults = {
        "bond_weight": 1.0,
        "angle_weight": 0.5,
        "repulsion_weight": 3.0,
        "hard_core_scale": 1.0,
        "nonbond_push_scale": 1.0,
        "displacement_sigma": 0.0,
    }
    out = {}
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
    """Distribute *total* samples across REGIME_STRATA by quota."""
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


def build_preset_samples(
    rng: np.random.Generator, cell_size: float,
) -> list[SampleConfig]:
    regimes = list(Supercell.PRESETS.keys())
    n_per = max(1, N_PRESET_SAMPLES // len(regimes))
    configs: list[SampleConfig] = []
    idx = 0
    for regime in regimes:
        preset = dict(Supercell.PRESETS[regime])
        preset.pop("relative_density", None)
        preset_num_steps = int(preset.pop("num_steps", N_STEPS_DEFAULT))
        for _ in range(n_per):
            gs = float(preset.get("grain_size") or 0.0)
            preset_cf = float(preset.get("crystalline_fraction", 1.0))
            num_grains = _num_grains_for(gs, cell_size)
            if num_grains == 0:
                n_crystalline = 0
                cf = 0.0
            else:
                n_crystalline = int(np.clip(
                    round(preset_cf * num_grains), 0, num_grains,
                ))
                cf = n_crystalline / num_grains
            w = _jittered_weights(preset, rng)
            configs.append(SampleConfig(
                idx=idx,
                source="preset",
                anchor_regime=regime,
                grain_size=gs,
                num_grains=num_grains,
                n_crystalline=n_crystalline,
                crystalline_fraction=cf,
                bond_weight=w["bond_weight"],
                angle_weight=w["angle_weight"],
                repulsion_weight=w["repulsion_weight"],
                hard_core_scale=w["hard_core_scale"],
                nonbond_push_scale=w["nonbond_push_scale"],
                displacement_sigma=w["displacement_sigma"],
                rng_seed=BASE_SEED + idx,
                num_steps=preset_num_steps,
            ))
            idx += 1
    return configs


def build_stratified_samples(
    rng: np.random.Generator, start_idx: int, cell_size: float,
) -> list[SampleConfig]:
    """Balanced per-regime sampling respecting the cf discretization.

    For each stratum:
      1. Compute n_samples from its quota.
      2. Liquid stratum (gs_range = (0, 0)): emit no-grain samples (cf=0).
      3. Otherwise: gs ~ Uniform[stratum_lo, min(stratum_hi, cell_size)],
         num_grains from tricor formula, n_crystalline ~ Uniform{1, ..., num_grains},
         cf = n_crystalline / num_grains, weights = stratum's preset + jitter.
    """
    configs: list[SampleConfig] = []
    counts = _stratum_counts(N_STRATIFIED_SAMPLES)
    idx = start_idx

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
                # n_crystalline >= 1 for crystalline strata (cf=0 belongs to liquid).
                n_crystalline = int(rng.integers(1, num_grains + 1))
                cf = n_crystalline / num_grains

            w = _jittered_weights(preset, rng)
            configs.append(SampleConfig(
                idx=idx,
                source="stratified",
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
                rng_seed=BASE_SEED + idx,
                num_steps=preset_num_steps,
            ))
            idx += 1
    return configs


# ══════════════════════════════════════════════════════════════════════════════


# Worker-global state populated once per process via _init_worker.
_WORKER_REF = None
_WORKER_SHELL_TARGET = None
_WORKER_CELL_SIZE: float = 0.0
_WORKER_REL_DENSITY: float = 0.0
_WORKER_OUT_DIR: Path | None = None


def _init_worker(
    si_lattice: float, cell_size: float, rel_density: float, out_dir_str: str,
) -> None:
    """Pool initializer: build the reference Si crystal + shell_target once."""
    global _WORKER_REF, _WORKER_SHELL_TARGET
    global _WORKER_CELL_SIZE, _WORKER_REL_DENSITY, _WORKER_OUT_DIR
    _WORKER_REF = bulk("Si", crystalstructure="diamond", a=si_lattice, cubic=True)
    _WORKER_SHELL_TARGET = CoordinationShellTarget.from_atoms(_WORKER_REF)
    _WORKER_CELL_SIZE = float(cell_size)
    _WORKER_REL_DENSITY = float(rel_density)
    _WORKER_OUT_DIR = Path(out_dir_str)


def _run_one_worker(cfg: "SampleConfig") -> dict | None:
    """Entry point for each pool task.  Returns manifest row or None on failure."""
    try:
        return run_trajectory(
            cfg, _WORKER_REF, _WORKER_SHELL_TARGET,
            _WORKER_CELL_SIZE, _WORKER_REL_DENSITY, _WORKER_OUT_DIR,
        )
    except Exception as e:
        print(f"[{cfg.idx+1:4d}] FAILED: {type(e).__name__}: {e}", flush=True)
        return None


def run_trajectory(
    cfg: "SampleConfig",
    ref,
    shell_target,
    cell_size: float,
    rel_density: float,
    out_dir: Path,
) -> dict[str, float | int | str]:
    """Run Supercell.generate for one sampled config; dump a .npz."""
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
        positions=h["positions"],                         # (S, N, 3) float32
        snapshot_steps=h["snapshot_steps"],               # (S,) int32
        initial_positions=initial_positions,              # (N, 3) float32
        best_positions=h["best_positions"],               # (N, 3) float32
        final_positions=sc.atoms.positions.astype(np.float32),
        species_numbers=sc.atoms.numbers.astype(np.int32),
        cell=np.asarray(sc.atoms.cell.array, dtype=np.float32),
        loss_history=h["loss"],                           # (num_steps+1,) float64
        # Sampling metadata
        idx=np.int64(cfg.idx),
        source=np.asarray(cfg.source),
        regime=np.asarray(cfg.anchor_regime),
        rng_seed=np.int64(cfg.rng_seed),
        # Structural parameters
        grain_size=np.float32(cfg.grain_size),
        num_grains=np.int32(cfg.num_grains),
        n_crystalline=np.int32(cfg.n_crystalline),
        crystalline_fraction=np.float32(cfg.crystalline_fraction),
        # Weight parameters
        bond_weight=np.float32(cfg.bond_weight),
        angle_weight=np.float32(cfg.angle_weight),
        repulsion_weight=np.float32(cfg.repulsion_weight),
        hard_core_scale=np.float32(cfg.hard_core_scale),
        nonbond_push_scale=np.float32(cfg.nonbond_push_scale),
        displacement_sigma=np.float32(cfg.displacement_sigma),
        # Run settings
        num_steps=np.int32(cfg.num_steps),
        trajectory_stride=np.int32(TRAJECTORY_STRIDE),
        cell_size=np.float32(cell_size),
        rel_density=np.float32(rel_density),
        # Quality metrics (for downstream filtering, not used in surrogate training)
        initial_loss=np.float64(summary["initial_loss"]),
        best_loss=np.float64(summary["best_loss"]),
        final_loss=np.float64(summary["final_loss"]),
    )

    n_atoms = len(sc.atoms)
    size_mb = outfile.stat().st_size / (1024 * 1024)
    print(
        f"[{cfg.idx+1:4d}] {cfg.source:>10s} {cfg.anchor_regime:>16s} "
        f"gs={cfg.grain_size:5.1f} ng={cfg.num_grains:3d} nc={cfg.n_crystalline:3d} "
        f"cf={cfg.crystalline_fraction:.3f} atoms={n_atoms:5d} "
        f"loss {summary['initial_loss']:6.2f}→{summary['final_loss']:6.2f} "
        f"best={summary['best_loss']:6.2f}  {runtime:5.1f}s  {size_mb:4.1f}MB"
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


def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = bulk("Si", crystalstructure="diamond", a=SI_LATTICE, cubic=True)
    shell_target = CoordinationShellTarget.from_atoms(ref)

    rng = np.random.default_rng(BASE_SEED)
    preset_configs = build_preset_samples(rng, cell_size=CELL_SIZE)
    stratified_configs = build_stratified_samples(
        rng, start_idx=len(preset_configs), cell_size=CELL_SIZE,
    )
    configs = preset_configs + stratified_configs
    total = len(configs)
    print(f"Si reference: diamond a={SI_LATTICE} Å  cell_size={CELL_SIZE} Å")
    print(f"Running {total} configs  (presets={len(preset_configs)}, "
          f"stratified={len(stratified_configs)})")
    counts = _stratum_counts(N_STRATIFIED_SAMPLES)
    print("Stratum quotas (stratified tier only):")
    for s, c in zip(REGIME_STRATA, counts):
        print(f"  {s['name']:18s} range={s['gs_range']}  quota={s['quota']:.2f} -> {c}")
    print(f"Output dir: {out_dir}\n")

    manifest: list[dict] = []
    t_start = time.perf_counter()

    if NUM_WORKERS is None or NUM_WORKERS <= 1:
        print("Running serially (NUM_WORKERS <= 1)")
        for cfg in configs:
            try:
                row = run_trajectory(
                    cfg, ref, shell_target, CELL_SIZE, REL_DENSITY, out_dir,
                )
                manifest.append(row)
            except Exception as e:
                print(f"[{cfg.idx+1:4d}] FAILED: {type(e).__name__}: {e}")
    else:
        print(f"Running in parallel with {NUM_WORKERS} workers")
        # Using "spawn" for portability (macOS default, works everywhere).
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=NUM_WORKERS,
            initializer=_init_worker,
            initargs=(SI_LATTICE, CELL_SIZE, REL_DENSITY, str(out_dir)),
        ) as pool:
            # imap_unordered to let fast configs return without blocking on
            # slower ones.  chunksize=1 keeps load-balancing responsive for
            # heterogeneous per-config runtimes.
            for i, row in enumerate(
                pool.imap_unordered(_run_one_worker, configs, chunksize=1)
            ):
                if row is not None:
                    manifest.append(row)

    elapsed = time.perf_counter() - t_start

    # Write manifest CSV
    manifest_path = out_dir / "manifest.csv"
    if manifest:
        headers = list(manifest[0].keys())
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in manifest:
                writer.writerow(row)

    print(f"\nDone. Wrote {len(manifest)}/{total} trajectories to {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Total wall time: {elapsed:.1f}s  ({elapsed/max(total,1):.1f}s/config avg)")


if __name__ == "__main__":
    main()
