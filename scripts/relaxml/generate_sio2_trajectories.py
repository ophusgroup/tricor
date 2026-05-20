"""Generate SiO₂ multi-polymorph trajectories for phase-conditioning training.

Variant of generate_surrogate_trajectories.py specialized for SiO₂.  In a
single run it iterates over a list of SiO₂ polymorphs (α-quartz, cristobalite,
tridymite, etc.) and produces an independent trajectory dataset for each,
each with that polymorph's own shell_target written into the .npz.  The
intent is to break the species-↔-shell_target confound: with only one
polymorph per compound, the model can shortcut around the shell_target
encoder; with several polymorphs sharing species (Si, O) but differing
shell_targets, it has to actually use the conditioning input.

Each polymorph's output goes into its own subdir + uses a polymorph-
distinct filename prefix + seed offset, so they don't collide and so the
merge step can union or hold them out individually.

To find available SiO₂ CIFs in the library:
    ls /data/users/ehrdt/prod/cifs_mp_cnos/*_SiO2.cif

Pick polymorphs by inspecting the structures (different space groups +
densities ≈ different polymorphs); fill POLYMORPHS below; run:
    python generate_surrogate_trajectories_sio2_polymorphs.py

Held-out evaluation strategy: leave one polymorph out of POLYMORPHS during
training, generate it separately for testing — that's the real probe of
whether the shell_target conditioning is doing anything useful.
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

from tricor.relaxml.shell_target import extract_shell_target_arrays
from tricor.shells import CoordinationShellTarget
from tricor.supercell import Supercell


# Materials Project API key (used only when multiple polymorphs match a
# compound's pattern and we need to pick the hull-stable one).  Same key
# file as scripts/run_mp_nos.py in mc_structgen.
MP_API_KEY_FILE = Path("/home/ehrdt/misc/materials_project_api.txt")

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


def _resolve_cif(cif_dir: Path, pattern: str, compound_name: str) -> Path:
    """Glob CIF_DIR for `pattern` and return a single match.

    For a single match, return it directly.  For multiple matches, look up
    the cached hull pick or call MP to pick the lowest e_above_hull.
    """
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


def _build_polymorph_reference(polymorph: "PolymorphSpec", cif_dir: Path):
    """Read the reference structure for a polymorph spec from its CIF file."""
    cif_path = cif_dir / polymorph.cif_filename
    if not cif_path.is_file():
        raise FileNotFoundError(
            f"CIF for polymorph {polymorph.tag!r} not found: {cif_path}"
        )
    return ase_read(str(cif_path), format="cif")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PolymorphSpec:
    """One SiO₂ polymorph: a tag + an exact CIF filename under CIF_DIR.

    The tag becomes the per-polymorph subdir name and the filename
    prefix, so it should be filesystem-safe (letters, digits, underscores,
    hyphens — no spaces or slashes).
    """
    tag: str            # e.g. "alpha_quartz"
    cif_filename: str   # exact filename under CIF_DIR (no glob); MP IDs
                        # disambiguate polymorphs since they all share
                        # the formula SiO₂.


# CIF library on mallard, populated by mc_structgen/test/run_mp_nos.py.
CIF_DIR = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos")

# Polymorphs to generate this run.  Replace the placeholder mp-IDs below
# with the actual filenames you have in CIF_DIR — find them with:
#     ls /data/users/ehrdt/prod/cifs_mp_cnos/*_SiO2.cif
# and pick polymorphs by looking at their structures (different space
# groups + densities = different phases).  Common SiO₂ polymorphs in MP:
#   α-quartz       (P3₁2₁)  — ground state
#   β-quartz       (P6₂22)
#   α-cristobalite (P4₁2₁2)
#   β-cristobalite (Fd-3m)
#   α-tridymite    (P2₁/c)
#   coesite        (C2/c)   — high-pressure
#   stishovite     (P4₂/mnm) — high-pressure (6-coord Si, very different!)
#
# For the held-out-phase test: leave one of these out of POLYMORPHS during
# training, generate it separately afterwards, and evaluate on it.
POLYMORPHS: list[PolymorphSpec] = [
    # PolymorphSpec("alpha_quartz",       "mp-7000_SiO2.cif"),
    # PolymorphSpec("alpha_cristobalite", "mp-6945_SiO2.cif"),
    # PolymorphSpec("beta_cristobalite",  "mp-546794_SiO2.cif"),
    # PolymorphSpec("coesite",            "mp-6930_SiO2.cif"),
    PolymorphSpec("stishovite",         "mp-6947_SiO2.cif"),
]

# Composition tag — used as the manifest's `compound` column and as the
# filename prefix.  All polymorphs share it so the merged manifest knows
# they're the same composition; the polymorph distinction is in the
# `polymorph` column we add below.
COMPOUND_NAME = "SiO2"

# Each polymorph gets its own subdir under DATASET_ROOT/COMPOUND_NAME/
# so they can be merged or held out individually.
DATASET_ROOT = "./data/sio2_polymorphs_v1"

CELL_SIZE = 50.0                # Å
REL_DENSITY = 0.96

# Per-polymorph stratified samples.  Default 150 × 4 polymorphs = 600
# trajectories total — enough to break the species-↔-shell_target
# confound without blowing the generation budget.
N_PRESET_SAMPLES = 0
N_STRATIFIED_SAMPLES = 150

N_STEPS_DEFAULT = 200
TRAJECTORY_STRIDE = 5           # save every 5th step (≈ 40 snapshots / run)

# BASE_SEED is set per-polymorph at runtime via _polymorph_seed() so
# different polymorphs use disjoint seed ranges and filenames are unique.
# Polymorph idx i gets BASE_SEED_ROOT + i * 50_000.
BASE_SEED_ROOT = 800_000

# This module-level placeholder is overwritten per polymorph in main();
# kept around because build_preset_samples / build_stratified_samples
# read it directly.  See _set_polymorph_seed().
BASE_SEED = BASE_SEED_ROOT

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
_WORKER_POLYMORPH_TAG: str = ""


def _init_worker(
    polymorph: "PolymorphSpec",
    cif_dir_str: str,
    cell_size: float,
    rel_density: float,
    out_dir_str: str,
) -> None:
    """Pool initializer: build the polymorph reference + shell_target once."""
    global _WORKER_REF, _WORKER_SHELL_TARGET
    global _WORKER_CELL_SIZE, _WORKER_REL_DENSITY, _WORKER_OUT_DIR
    global _WORKER_POLYMORPH_TAG
    _WORKER_REF = _build_polymorph_reference(polymorph, Path(cif_dir_str))
    _WORKER_SHELL_TARGET = CoordinationShellTarget.from_atoms(_WORKER_REF)
    _WORKER_CELL_SIZE = float(cell_size)
    _WORKER_REL_DENSITY = float(rel_density)
    _WORKER_OUT_DIR = Path(out_dir_str)
    _WORKER_POLYMORPH_TAG = polymorph.tag


def _run_one_worker(cfg: "SampleConfig") -> dict | None:
    """Entry point for each pool task.  Returns manifest row or None on failure."""
    try:
        return run_trajectory(
            cfg, _WORKER_REF, _WORKER_SHELL_TARGET,
            _WORKER_CELL_SIZE, _WORKER_REL_DENSITY, _WORKER_OUT_DIR,
            _WORKER_POLYMORPH_TAG,
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
    polymorph_tag: str,
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
        f"{COMPOUND_NAME}_{polymorph_tag}_{cfg.anchor_regime}_"
        f"cell{int(cell_size):03d}_idx{cfg.idx:05d}_"
        f"seed{cfg.rng_seed:09d}.npz"
    )
    outfile = out_dir / filename

    # Flatten the shell_target driving this trajectory into the four-array
    # schema the relaxml model conditions on.  Same target for every pair
    # in this trajectory (it's a per-trajectory constant), so we store it
    # once per file.
    shell_target_arrays = extract_shell_target_arrays(shell_target)

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
        # Phase conditioning: shell_target arrays consumed by the
        # ShellTargetEncoder in the relaxml model.
        **shell_target_arrays,
        # Sampling metadata
        idx=np.int64(cfg.idx),
        source=np.asarray(cfg.source),
        regime=np.asarray(cfg.anchor_regime),
        compound=np.asarray(COMPOUND_NAME),
        polymorph=np.asarray(polymorph_tag),
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
        "compound": COMPOUND_NAME,
        "polymorph": polymorph_tag,
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


def _run_one_polymorph(polymorph: "PolymorphSpec", polymorph_idx: int) -> int:
    """Generate trajectories for a single polymorph.  Returns the number
    of successful trajectories written."""
    out_dir = Path(DATASET_ROOT) / COMPOUND_NAME / f"{polymorph.tag}_trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-polymorph base seed so the seed ranges (and therefore filenames)
    # don't collide across polymorphs.
    polymorph_base_seed = BASE_SEED_ROOT + polymorph_idx * 50_000
    # build_*_samples reads the module-level BASE_SEED, so we point that
    # at the per-polymorph value for the duration of this call.
    global BASE_SEED
    BASE_SEED = polymorph_base_seed

    ref = _build_polymorph_reference(polymorph, CIF_DIR)
    shell_target = CoordinationShellTarget.from_atoms(ref)

    rng = np.random.default_rng(BASE_SEED)
    preset_configs = build_preset_samples(rng, cell_size=CELL_SIZE)
    stratified_configs = build_stratified_samples(
        rng, start_idx=len(preset_configs), cell_size=CELL_SIZE,
    )
    configs = preset_configs + stratified_configs
    total = len(configs)
    species_present = sorted({int(z) for z in ref.numbers})
    print()
    print(f"━━━ Polymorph: {polymorph.tag}  CIF: {polymorph.cif_filename} ━━━")
    print(f"  Reference cell: {len(ref)} atoms, species Z={species_present}")
    print(f"  Cell volume: {ref.cell.volume:.2f} Å³  PBC={ref.pbc.tolist()}")
    print(f"  shell_target: {len(shell_target.pair_labels)} pairs, "
          f"{len(shell_target.angle_labels)} triplets (pre-filter)")
    print(f"  Supercell target: {CELL_SIZE} Å  base seed: {BASE_SEED}")
    print(f"  Running {total} configs  (presets={len(preset_configs)}, "
          f"stratified={len(stratified_configs)})")
    print(f"  Output dir: {out_dir}")

    manifest: list[dict] = []
    t_start = time.perf_counter()

    if NUM_WORKERS is None or NUM_WORKERS <= 1:
        for cfg in configs:
            try:
                row = run_trajectory(
                    cfg, ref, shell_target, CELL_SIZE, REL_DENSITY, out_dir,
                    polymorph.tag,
                )
                manifest.append(row)
            except Exception as e:
                print(f"[{cfg.idx+1:4d}] FAILED: {type(e).__name__}: {e}")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=NUM_WORKERS,
            initializer=_init_worker,
            initargs=(polymorph, str(CIF_DIR), CELL_SIZE, REL_DENSITY, str(out_dir)),
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

    print(f"  → {len(manifest)}/{total} trajectories  "
          f"wall {elapsed:.1f}s  ({elapsed/max(total,1):.1f}s/config avg)")
    print(f"  Manifest: {manifest_path}")
    return len(manifest)


def main() -> None:
    if not CIF_DIR.is_dir():
        raise SystemExit(f"CIF_DIR not found: {CIF_DIR}")
    if not POLYMORPHS:
        raise SystemExit("POLYMORPHS is empty — nothing to generate.")

    # Validate every polymorph's CIF up front so we don't crash partway
    # through a long run with a typo'd MP ID.
    for p in POLYMORPHS:
        cif_path = CIF_DIR / p.cif_filename
        if not cif_path.is_file():
            raise SystemExit(
                f"Polymorph {p.tag!r}: CIF not found at {cif_path}"
            )

    counts = _stratum_counts(N_STRATIFIED_SAMPLES)
    print(f"Compound: {COMPOUND_NAME}")
    print(f"Polymorphs to generate: {len(POLYMORPHS)}")
    for p in POLYMORPHS:
        print(f"  {p.tag:24s}  {p.cif_filename}")
    print(f"Per-polymorph: {N_STRATIFIED_SAMPLES} stratified samples")
    print("Stratum quotas:")
    for s, c in zip(REGIME_STRATA, counts):
        print(f"  {s['name']:18s} range={s['gs_range']}  quota={s['quota']:.2f} -> {c}")
    print(f"Cell size: {CELL_SIZE} Å  workers: {NUM_WORKERS}")

    t_global = time.perf_counter()
    total_rows = 0
    for i, polymorph in enumerate(POLYMORPHS):
        total_rows += _run_one_polymorph(polymorph, i)
    elapsed = time.perf_counter() - t_global

    print()
    print(f"━━━ All polymorphs done.  Total: {total_rows} trajectories  "
          f"wall {elapsed:.1f}s ({elapsed/3600:.2f} hr)")


if __name__ == "__main__":
    main()
