"""Generate relaxation trajectories for training a multi-species GNN surrogate.

Version 2 (2026-05-13): updated for the upstream tricor merge of May 2026.
Same behavior as v1 EXCEPT:

  * Passes the new ``refine_orientations`` kwarg to ``Supercell.generate``
    (default True in this script — runs a cheap topology-free coordinate
    descent over per-grain rotations before the FIRE quench, recommended
    for directional-bond materials like Si and Si3N4).  No-op when
    ``grain_size`` is 0 (liquid/amorphous regimes).

  * Passes the new ``k_restraint`` kwarg (position-tether spring strength,
    eV/Å²) — defaults to 0.0 (off) but can be raised to preserve regime
    character (grain layout, amorphous topology) during relaxation when
    grains tend to drift apart.

  * Default ``DATASET_ROOT`` bumped to ``data/multi_species_v2`` so v2
    outputs don't collide with v1 trajectories on disk.

All other parameters (bond_weight, angle_weight, etc.) come from the same
stratified sampler as v1.  The relax inner loop is whatever the merged
upstream tricor provides.

Uses the stratified sampler from diagnose_param_space (balanced regime coverage
with n_crystalline >= 1 for crystalline strata).  Each sample yields one .npz
with the full shell_relax trajectory, initial/final/best positions, species,
cell, all weight parameters, grain_size/cf/num_grains/n_crystalline, and the
initial/best/final losses as metadata.  A manifest CSV indexes every sample.

The compound to generate is selected by COMPOUND_NAME below.  Each compound
gets its own output dir + filename prefix, so multiple runs for different
compounds can later be merged via scripts/relaxml/merge_manifests.py.

No quality gate is applied here: the surrogate should learn tricor's full
output distribution, including noisy/hard configs.  Downstream datasets that
use the surrogate for large-scale structure generation should filter by
best_loss (or a structure-based metric) at that stage instead.

Edit the CONFIG section below, then run:
    python generate_surrogate_trajectories_multicomp_v2.py
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
import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from ase.io import read as ase_read

from tricor.shells import CoordinationShellTarget
from tricor.supercell import Supercell


# Materials Project API key (used only when multiple polymorphs match a
# compound's pattern and we need to pick the hull-stable one).  Same key
# file as scripts/run_mp_nos.py in mc_structgen.
MP_API_KEY_FILE = Path("/home/ehrdt/materials_project_api.txt")

# Hull-pick cache filename (lives next to the CIFs).  Maps compound name
# → chosen CIF filename so we don't requery MP on every script run.
_HULL_CACHE_FILE = "hull_picks.json"


def _load_hull_cache(cif_dir: Path) -> dict[str, str]:
    p = cif_dir / _HULL_CACHE_FILE
    return json.loads(p.read_text()) if p.is_file() else {}


def _save_hull_cache(cif_dir: Path, cache: dict[str, str]) -> None:
    (cif_dir / _HULL_CACHE_FILE).write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n"
    )


def _pick_hull_polymorph(compound_name: str, candidates: list[Path]) -> Path:
    """Query MP for e_above_hull of each candidate and return the lowest.

    Filenames are assumed to start with the MP ID (mp-NNN_Formula.cif),
    matching what scripts/run_mp_nos.py in mc_structgen writes.
    """
    from mp_api.client import MPRester
    if not MP_API_KEY_FILE.is_file():
        raise FileNotFoundError(
            f"MP API key not found at {MP_API_KEY_FILE}; cannot auto-pick "
            f"hull polymorph for {compound_name}.  Either drop the key file "
            f"there or pin a polymorph by replacing the glob with an exact "
            f"filename in COMPOUND_PRESETS."
        )
    api_key = MP_API_KEY_FILE.read_text().strip()

    id_to_path = {p.name.split("_", 1)[0]: p for p in candidates}
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=list(id_to_path),
            fields=["material_id", "energy_above_hull"],
        )
    if not docs:
        raise RuntimeError(
            f"MP query returned no results for {list(id_to_path)} "
            f"({compound_name})."
        )
    best = min(docs, key=lambda d: float(d.energy_above_hull))
    chosen = id_to_path[str(best.material_id)]
    print(
        f"[hull-pick] {compound_name}: {chosen.name} "
        f"(e_above_hull={float(best.energy_above_hull):.4f} eV/atom; "
        f"chose 1 of {len(candidates)})"
    )
    return chosen


def _resolve_cif(
    cif_dir: Path,
    pattern: str,
    compound_name: str,
    mp_id: str | None = None,
) -> Path:
    """Resolve the CIF for a compound under ``cif_dir``.

    If ``mp_id`` is given, target ``{mp_id}_{compound_name}.cif`` exactly
    and raise if it doesn't exist — pins a specific polymorph (e.g.
    ``mp-1143`` for Al2O3 corundum, distinguishing it from other Al2O3
    phases in MP).

    Otherwise glob ``cif_dir`` for ``pattern``.  For a single match,
    return it directly.  For multiple matches, look up the cached hull
    pick or call MP to pick the lowest e_above_hull.
    """
    if mp_id is not None:
        target = cif_dir / f"{mp_id}_{compound_name}.cif"
        if not target.is_file():
            raise FileNotFoundError(
                f"Pinned CIF not found: {target}.  Verify mp_id={mp_id!r} "
                f"is correct for compound {compound_name!r} (check "
                f"`ls {cif_dir}/*_{compound_name}.cif`)."
            )
        return target

    matches = sorted(cif_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No CIF in {cif_dir} matching {pattern!r}.  Check that the MP "
            f"download includes this compound."
        )
    if len(matches) == 1:
        return matches[0]

    cache = _load_hull_cache(cif_dir)
    cached_name = cache.get(compound_name)
    if cached_name is not None:
        cached_path = cif_dir / cached_name
        if cached_path.is_file():
            return cached_path
        # Fall through and re-query if the cached file vanished.

    chosen = _pick_hull_polymorph(compound_name, matches)
    cache[compound_name] = chosen.name
    _save_hull_cache(cif_dir, cache)
    return chosen


def _build_reference(spec: "CompoundSpec", cif_dir: Path):
    """Read the reference structure for a compound spec from a CIF file."""
    cif_path = _resolve_cif(cif_dir, spec.cif_pattern, spec.name, spec.mp_id)
    return ase_read(str(cif_path), format="cif")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CompoundSpec:
    """A reference compound loaded from a CIF file under CIF_DIR."""
    name: str           # short tag used in filenames + output paths
    cif_pattern: str    # glob (relative to CIF_DIR); must match exactly one file
    mp_id: str | None = None  # if set (e.g. "mp-1143"), pin to {mp_id}_{name}.cif
                              # exactly and skip the hull-pick fallback.  Used
                              # to nail down a specific polymorph when a formula
                              # has several (e.g. corundum vs other Al2O3 phases)


# CIF library on mallard, populated by mc_structgen/test/run_mp_nos.py
# (Materials Project: 1-2-element materials containing C/N/O/S +
# all monatomic, e_above_hull <= 500 meV/atom).  Filenames look like
# "mp-149_Si.cif".
CIF_DIR = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos")

# Initial multi-species glass-former set.  Patterns use formula globs;
# if MP returned multiple polymorphs for a formula, _resolve_cif will
# raise with the candidate list — replace the glob with the chosen
# filename to pin one polymorph (one phase per structure type for v1).
COMPOUND_PRESETS: dict[str, CompoundSpec] = {
    "Si":     CompoundSpec("Si",     "*_Si.cif"),
    "Ge":     CompoundSpec("Ge",     "*_Ge.cif"),
    "SiC":    CompoundSpec("SiC",    "*_SiC.cif"),
    "BN":     CompoundSpec("BN",     "*_BN.cif"),
    "AlN":     CompoundSpec("AlN",     "*_AlN.cif"),
    "Si3N4":  CompoundSpec("Si3N4",  "*_Si3N4.cif"),
    "SiO2":   CompoundSpec("SiO2",   "*_SiO2.cif"),
    "GeO2":   CompoundSpec("GeO2",   "*_GeO2.cif"),
    "B2O3":   CompoundSpec("B2O3",   "*_B2O3.cif"),
    "Al2O3":  CompoundSpec("Al2O3",  "*_Al2O3.cif"),
    "Ga2O3":  CompoundSpec("Ga2O3",  "*_Ga2O3.cif"),
    "TiO2":   CompoundSpec("TiO2",  "*_TiO2.cif"),
    "As2S3":  CompoundSpec("As2S3",  "*_As2S3.cif"),
}

# Pick which compound this run generates.  Re-run with a different name to
# build the next compound's dataset; outputs are siloed by name so they
# don't collide.
COMPOUND_NAME = "Si3N4" #next: Al2O3?

# Optional: pin to a specific MP polymorph by mp-id (e.g. "mp-1143" for
# Al2O3 corundum, "mp-2657" for TiO2 rutile, "mp-390" for TiO2 anatase).
# Use this when a formula has multiple stable polymorphs and you need a
# specific one (e.g. for the cross-coordination test, you want corundum
# Al2O3 specifically, not other Al2O3 phases).
#
# None  -> use the preset's default mp_id (if any) or fall back to the
#          MP hull pick (lowest e_above_hull).
# "mp-X" -> pin to {mp_id}_{compound}.cif exactly; raises if missing.
#
# When set, the chosen mp-id is appended to OUTPUT_DIR below so re-runs
# with different mp_ids don't collide (e.g. TiO2 rutile and TiO2 anatase
# can coexist as ./data/multi_species_v1/TiO2_mp-2657_trajectories_150
# and ./data/multi_species_v1/TiO2_mp-390_trajectories_150).
MP_ID: str | None = None

COMPOUND = COMPOUND_PRESETS[COMPOUND_NAME]
if MP_ID is not None:
    # Override the preset's default mp_id with the run-time choice.
    COMPOUND = CompoundSpec(
        name=COMPOUND.name,
        cif_pattern=COMPOUND.cif_pattern,
        mp_id=MP_ID,
    )

# Each compound gets its own subdir under a shared root so the merge step
# can union them all into one training manifest.  When MP_ID is pinned,
# the mp-id goes into the dir name so different polymorphs don't collide.
#
# v2 default points at a separate root so v2 outputs don't overwrite v1
# trajectories on disk.  Change to "./data/multi_species_v1" if you want
# to add v2 trajectories alongside v1 (be aware they'll have different
# relax behavior — the model's training distribution would be mixed).
DATASET_ROOT = "./data/multi_species_v2"
_dir_tag = f"{COMPOUND_NAME}_{COMPOUND.mp_id}" if COMPOUND.mp_id else COMPOUND_NAME
OUTPUT_DIR = f"{DATASET_ROOT}/{_dir_tag}_trajectories_150"

CELL_SIZE = 50.0                # Å — match the existing training data
REL_DENSITY = 0.96

# --- v2 additions: forwarded to Supercell.generate ---
# Build-time per-grain orientation refinement.  Runs a cheap topology-free
# coordinate-descent over per-grain rotations BEFORE the FIRE quench, so
# FIRE starts from a better basin.  Only meaningful when grains exist
# (no-op for liquid/amorphous strata where grain_size = 0).  Default True
# in v2 because it helps directional-bond materials (Si, Si3N4, Si4-N3).
REFINE_ORIENTATIONS = True

# Position-tether spring strength (eV/Å²).  0.0 (default) disables.
# Small values (~0.1-1.0) preserve regime character (grain layout,
# amorphous topology) while permitting local relaxation.  Large values
# (≫ 10) hold the structure rigid.  Useful when grains drift apart
# during relaxation.
K_RESTRAINT = 0.0

# 750 stratified samples per compound × 5 compounds ≈ 3750 total — same
# order of magnitude as the merged Si v2 dataset.
N_PRESET_SAMPLES = 0
N_STRATIFIED_SAMPLES = 50

N_STEPS_DEFAULT = 200
TRAJECTORY_STRIDE = 5           # save every 5th step (≈ 40 snapshots / run)
# Per-compound BASE_SEED so different compounds use disjoint seed ranges
# (compound_idx * 100k offset).  Keeps filenames unique across compounds
# even when regimes + idx coincide.
_COMPOUND_SEED_OFFSET = list(COMPOUND_PRESETS).index(COMPOUND_NAME) * 100_000
BASE_SEED = 400_000 + _COMPOUND_SEED_OFFSET

# Multiprocessing.  None = serial (for debugging).  Integer = number of
# worker processes.  A good default is half the physical cores to leave
# room for other tasks.
NUM_WORKERS= 8 #: int | None = max(1, (os.cpu_count() or 2) // 2)

WEIGHT_JITTER_SIGMA = 0.20         # log-normal sigma on weight params

# Balanced-regime sweep covering all six strata.  Per-compound generation
# wants the full disorder spectrum, not just the extremes.
# NOTE: upstream renamed "MRO_more" -> "LRO" in Supercell.PRESETS.
REGIME_STRATA = [
    {"name": "liquid",          "gs_range": (0.0,  0.0),   "quota": 0.18},
    {"name": "amorphous",       "gs_range": (4.0,  8.0),   "quota": 0.16},
    {"name": "SRO",             "gs_range": (8.0,  12.0),  "quota": 0.16},
    {"name": "MRO",             "gs_range": (12.0, 15.0),  "quota": 0.16},
    {"name": "LRO",             "gs_range": (15.0, 19.0),  "quota": 0.16},
    {"name": "nanocrystalline", "gs_range": (19.0, 25.0),  "quota": 0.18},
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
    spec: "CompoundSpec",
    cif_dir_str: str,
    cell_size: float,
    rel_density: float,
    out_dir_str: str,
) -> None:
    """Pool initializer: build the reference crystal + shell_target once."""
    global _WORKER_REF, _WORKER_SHELL_TARGET
    global _WORKER_CELL_SIZE, _WORKER_REL_DENSITY, _WORKER_OUT_DIR
    _WORKER_REF = _build_reference(spec, Path(cif_dir_str))
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
        # v2 additions — see CONFIG block.  refine_orientations is a
        # no-op when grain_size <= 0; k_restraint defaults to 0 (off).
        refine_orientations=REFINE_ORIENTATIONS,
        k_restraint=K_RESTRAINT,
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
        f"{COMPOUND.name}_{cfg.anchor_regime}_cell{int(cell_size):03d}_"
        f"idx{cfg.idx:05d}_seed{cfg.rng_seed:09d}.npz"
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
        "compound": COMPOUND.name,
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

    if not CIF_DIR.is_dir():
        raise SystemExit(f"CIF_DIR not found: {CIF_DIR}")

    ref = _build_reference(COMPOUND, CIF_DIR)
    shell_target = CoordinationShellTarget.from_atoms(ref)

    rng = np.random.default_rng(BASE_SEED)
    preset_configs = build_preset_samples(rng, cell_size=CELL_SIZE)
    stratified_configs = build_stratified_samples(
        rng, start_idx=len(preset_configs), cell_size=CELL_SIZE,
    )
    configs = preset_configs + stratified_configs
    total = len(configs)
    species_present = sorted({int(z) for z in ref.numbers})
    cif_path = _resolve_cif(CIF_DIR, COMPOUND.cif_pattern, COMPOUND.name,
                            COMPOUND.mp_id)
    print(f"Compound: {COMPOUND.name}  CIF: {cif_path.name}")
    print(f"Reference cell: {len(ref)} atoms, species Z={species_present}")
    print(f"Cell volume: {ref.cell.volume:.2f} Å³  PBC={ref.pbc.tolist()}")
    print(f"Supercell target: {CELL_SIZE} Å  base seed: {BASE_SEED}")
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
            initargs=(COMPOUND, str(CIF_DIR), CELL_SIZE, REL_DENSITY, str(out_dir)),
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
