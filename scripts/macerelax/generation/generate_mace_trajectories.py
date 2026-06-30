"""CIF-list-driven MACE+wall trajectory generation for the big dataset.

Walks a CIF directory (or a list file restricting it to a subset), generates
N_TRAJ_PER_CIF stratified-regime trajectories per CIF, and writes them under
``DATASET_ROOT/{compound}_{mp_id}_trajectories/`` with per-CIF manifests.
Mirrors ``scripts/relaxml/generate_big_dataset.py`` in shape, but each
trajectory is a MACE-MPA + min-distance wall FIRE relaxation (not tricor
shell_relax) — see ``MACE_RELAX_PILOT.md`` §14 and the 2026-05-28 session
log for the design rationale.

Pipeline per trajectory:
  1. tricor pack (regime preset, no jitter, num_steps=0).
  2. bond_relax cleanup (geometric overlap removal — not shell_relax).
  3. MACE-MPA + min-distance wall, N_STEPS FIRE steps at OPT_MAXSTEP Å.
     FIRE is memory-less, so per-step displacement is a learnable function
     of current state.  Settings validated by the v3_fire SiO2 sweep.
  4. save NPZ.

OOM-driven cell-size calibration (keeps CELL=50 as the target; lets actual
GPU memory be the arbiter instead of an a-priori atom-count budget):
  * Each CIF starts at CELL_SIZE (or its cached calibrated value from a
    previous run — see DATASET_ROOT/calibrated_cells.csv).
  * build_configs orders trajectories densest-regime-first, so the most
    OOM-prone regime runs first and any shrink happens fast.
  * If a trajectory raises torch.OutOfMemoryError, the cell is shrunk by
    SHRINK_FACTOR (0.95×), partial NPZs for this CIF are wiped (they
    have the now-wrong cell baked into filename + metadata), and the
    20-trajectory loop restarts.
  * Floor at MIN_CELL — a CIF that OOMs even there is logged and skipped.
  * Successful calibrations persist to ``DATASET_ROOT/calibrated_cells.csv``
    so re-runs start at the known-good cell without re-probing.

Per-trajectory failure tolerance:
  * Any exception inside ``run_trajectory`` is caught + logged to
    ``DATASET_ROOT/failures.csv`` (CIF, regime, seed, exception class +
    message); the loop continues.
  * A whole-CIF failure (e.g. CIF parse error, cache build OOM) is
    similarly caught + logged; subsequent CIFs continue.

Resumable on every level:
  * Existing per-trajectory NPZ → SKIP without re-running.
  * Existing per-CIF manifest with the expected row count → SKIP the cache
    build entirely (saves ~20s/CIF).

Multi-GPU partitioning:
  * Set GPU_IDS = [0, 1, 2, 3] to run on 4 GPUs in parallel.  Each gets a
    round-robin slice of the CIF index range via PILOT_CIF_INDICES env
    var, and its own log under DATASET_ROOT/logs/gpuN.log.

Perlmutter A100 40 GB note:
  MACE 0.3.16 only dispatches "float32" / "float64" through its
  default_dtype API, so MACE_DEFAULT_DTYPE stays at "float32" here.
  The memory savings come from cuequivariance instead: install
  `cuequivariance + cuequivariance-torch` in the env (pip install — no
  CUDA toolkit linking needed) for ~2-3× speedup and ~30 % peak memory
  reduction via fused tensor-product kernels.  MACE auto-detects on
  import.  Without cueq the script still runs at higher peak memory;
  the OOM-driven calibration catches anything that overflows.

NPZ schema additions vs the 8-system pilot v1:
  * ``compound``        string   parsed from CIF filename, e.g. "SiO2"
  * ``mp_id``           string   parsed from CIF filename, e.g. "mp-7000"
  * ``system_id``       string   = f"{compound}_{mp_id}"  (kept for
                                 downstream backward compatibility)
  * ``role``            DROPPED  (now a split-time concept; see
                                 make_experiment_manifests.py)
  * ``cell_overridden`` bool     True when auto-shrink kicked in

Run: python scripts/macerelax/generation/generate_mace_trajectories.py
"""
from __future__ import annotations

# Thread caps, set BEFORE numpy/torch imports so they take effect.
import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "8")

import csv
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ase.io import read as ase_read
from ase.optimize import FIRE
from mace.calculators import mace_mp

torch.set_num_threads(4)

import tricor as tc
from tricor.g3 import G3Distribution

sys.path.insert(0, str(Path(__file__).parent))
from wall_calculator import MinDistanceWallCalculator, per_pair_min_from_atoms


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# --- CIF source ---
# Either point at the full library (and rely on CIF_LIST_FILE to restrict),
# or point at the already-subsetted training symlink dir.
CIF_DIR       = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos_le100meV_training")
CIF_LIST_FILE = None   # None = use every *.cif in CIF_DIR

# --- Output ---
# Set per-run.  On Perlmutter this will be a path under $SCRATCH or $CFS.
DATASET_ROOT  = Path("/home/ehrdt/tricor/mace/data/big_v1")

# --- Per-CIF generation ---
N_TRAJ_PER_CIF = 12             # 2 trajectories per regime × 6 regimes (post-
                                  # liquid-drop, see REGIME_STRATA below).
                                  # Was 20 (matching relaxml-big) before the
                                  # n=60 step + drop-liquid economy decision.
BASE_SEED      = 1_000_000
SEED_STEP      = 1000            # gap between consecutive CIFs' seed blocks;
                                  # must exceed N_TRAJ_PER_CIF so they don't
                                  # overlap.
SOURCE_TAG     = "mace_big_v1"   # written into every NPZ's `source` field

# --- Optional: stop after this many CIFs (smoke testing) ---
MAX_CIFS = 30                  # None = run them all

# --- Multi-GPU spawning ---
# Round-robin partition over CIF indices.  Set GPU_IDS = [<id>] for single
# GPU; GPU_IDS = [] disables pinning entirely (and the workers run in the
# parent process — handy for debugging).
GPU_IDS = [1]

# --- Packing (regime knobs) ---
CELL_SIZE = 50.0                 # target supercell edge length (Å).  Held
                                  # unless an actual CUDA OOM forces a shrink.
                                  # See "OOM-driven calibration" below.

# OOM-driven cell calibration.  Instead of guessing an ATOM_BUDGET that
# corresponds to the GPU memory ceiling, we let actual GPU memory be the
# arbiter: try CELL_SIZE, retry at CELL_SIZE * SHRINK_FACTOR**n on
# torch.OutOfMemoryError, persist the calibrated value to a sidecar CSV
# so re-runs skip the probe.  build_configs() puts crystalline_30 (the
# densest regime) first so an OOM-prone CIF fails fast on trajectory 0
# rather than after several successful regimes worth of work.
SHRINK_FACTOR = 0.95             # multiplicative shrink per OOM retry.
                                  # 0.95**n → atom count × 0.95**(3n).
MIN_CELL = 40.0                  # don't shrink below this — at smaller cells,
                                  # the crystalline_30 grains (~21 Å diameter,
                                  # two-grain target) start being unphysical.
                                  # A CIF that OOMs at MIN_CELL is logged
                                  # + skipped.
CALIBRATION_CACHE_FILENAME = "calibrated_cells.csv"

REGIME_STRATA = ["amorphous", "SRO", "MRO", "LRO",
                  "nanocrystalline", "crystalline_30"]
# `liquid` was dropped from the production sweep — the step-60 PDF/ADF
# overlay showed liquid endpoints were nearly indistinguishable from
# amorphous, so removing it bought a ~14% N_TRAJ savings without much
# loss in regime breadth.  See mace/compare_steps_pdfadf.py output.
DENSITY_BY_REGIME = {
    "amorphous":        0.92,
    "SRO":              0.92,
    "MRO":              0.88,
    "LRO":              0.92,
    "nanocrystalline":  0.96,
    "crystalline_30":   0.98,
}
# Display-only: estimate_atom_count() uses this to print expected atom
# counts per CIF.  No longer gates anything.
MAX_REL_DENSITY = max(DENSITY_BY_REGIME.values())


def _build_local_presets() -> dict:
    """tricor PRESETS with displacement_sigma=0 + a custom crystalline_30.

    Only builds presets for regimes that are actually in REGIME_STRATA;
    drops liquid because it was removed from the production sweep.
    crystalline_30 isn't in tricor's PRESETS — we synthesize it inline.
    """
    base = {}
    for name in REGIME_STRATA:
        if name == "crystalline_30":
            continue   # synthesized below; not in tricor's PRESETS
        d = dict(tc.Supercell.PRESETS[name])
        d["displacement_sigma"] = 0.0
        base[name] = d
    base["crystalline_30"] = dict(
        num_steps=0, grain_size=30.0, displacement_sigma=0.0,
        bond_weight=3.0, angle_weight=1.5,
    )
    return base


LOCAL_PRESETS = _build_local_presets()

# --- Geometric overlap cleanup (before MACE) ---
CLEANUP_METHOD          = "bond_relax"
BOND_RELAX_N_ITER       = 80
BOND_RELAX_MAX_STEP     = 0.1
HARDCORE_N_ITER         = 40
HARDCORE_PUSH_FRACTION  = 0.5

# --- MACE+wall relaxation ---
MACE_MODEL          = "medium-mpa-0"
MACE_DEVICE         = "cuda"       # "cuda", "cpu", or "auto"; "cuda"/"auto"
                                    # fall back to CPU when no GPU is visible
                                    # (resolved at the mace_mp call below)
MACE_DEFAULT_DTYPE  = "float32"    # MACE 0.3.16 only dispatches "float32"
                                    # and "float64" through its default_dtype
                                    # knob (mace/tools/torch_tools.py:80).
                                    # Memory savings on A100 40 GB come
                                    # from cuequivariance instead (pip
                                    # install cuequivariance + cueq-torch).
N_STEPS             = 60         # was 80.  mace/compare_steps_pdfadf.py
                                  # overlay validated that step-60 and
                                  # step-80 PDF/ADF curves are visually
                                  # indistinguishable for the disordered
                                  # regimes, with only small differences
                                  # at crystalline_30.  Drop to 60 buys
                                  # ~18% per-trajectory savings.
OPT_MAXSTEP         = 0.3
FMAX_TARGET         = 0.05
TRAJECTORY_STRIDE   = 1

WALL_K        = 1000.0
WALL_EXPONENT = 4
WALL_MARGIN   = 0.0

# ══════════════════════════════════════════════════════════════════════════════


_MP_ID_RE = re.compile(r"^(mp-\d+)_(.+)$")


@dataclass(frozen=True)
class CifSpec:
    """Per-CIF identity + chosen cell size."""
    cif_path:    Path
    compound:    str          # e.g. "SiO2"
    mp_id:       str          # e.g. "mp-7000"
    cell_size:   float        # the cell edge actually used (possibly auto-shrunk)
    cell_overridden: bool     # True iff cell_size < CELL_SIZE

    @property
    def system_id(self) -> str:
        return f"{self.compound}_{self.mp_id}"


@dataclass(frozen=True)
class SampleConfig:
    idx: int
    regime: str
    rng_seed: int
    relative_density: float


@dataclass
class _CifCache:
    """Per-CIF data computed ONCE and reused across all trajectories.
    See generate_mace_trajectories.py history / MACE_RELAX_PILOT.md §14.5
    for why measure_g3 lives here (called per-CIF, not per-trajectory)."""
    ref_atoms: object
    shell:     object
    distribution: G3Distribution


def parse_cif_name(cif_path: Path) -> tuple[str, str]:
    """Parse ``mp-XXX_Formula.cif`` filename → (compound, mp_id).

    Raises ValueError on a filename that doesn't match the convention.
    Used by both CIF discovery and the per-trajectory provenance stamp.
    """
    m = _MP_ID_RE.match(cif_path.stem)
    if not m:
        raise ValueError(
            f"could not parse mp-id + formula from {cif_path.name}; "
            "expected pattern 'mp-NNNN_Formula.cif'."
        )
    return m.group(2), m.group(1)  # (compound, mp_id) — caller convention


def estimate_atom_count(ref_atoms, cell_size: float,
                          rel_density: float = MAX_REL_DENSITY) -> int:
    """Atom-count estimate at the worst-case (densest) regime.

    n_atoms ≈ (N_ref / V_ref_unit_cell) * cell_size³ * rel_density
    """
    n_ref = len(ref_atoms)
    v_ref = float(ref_atoms.cell.volume)
    if v_ref <= 0:
        # Degenerate cell (rare; flag it for the failure log to inspect).
        return int(1e9)
    return int(round(n_ref / v_ref * cell_size**3 * rel_density))


def _is_cuda_oom(exc: BaseException) -> bool:
    """True if `exc` is a CUDA out-of-memory error.

    PyTorch ≥ 2 raises ``torch.OutOfMemoryError`` (subclass of RuntimeError).
    Be permissive about RuntimeErrors with "out of memory" in the message
    in case a path through MACE/e3nn wraps it.
    """
    if isinstance(exc, getattr(torch, "OutOfMemoryError", RuntimeError)):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def build_configs(n_traj: int, base_seed: int) -> list[SampleConfig]:
    """One SampleConfig per (regime × idx) for this CIF.

    Regime order is by descending density so the densest regime (which
    consumes the most GPU memory) runs first.  Two benefits:
      1. An OOM-prone CIF fails fast on trajectory 0 (≈2-3 min), not
         after 6 successful regimes' worth of work.
      2. After the first crystalline_30 trajectory succeeds at the chosen
         cell, every subsequent (less-dense) regime is guaranteed to fit.

    At the default n_traj=12 with 6 regimes the distribution is
    [2, 2, 2, 2, 2, 2] — exactly 2 trajectories per regime.  Liquid was
    dropped from REGIME_STRATA on the economy pass since its endpoint
    structure was nearly indistinguishable from amorphous.
    """
    # Densest first, ties broken by REGIME_STRATA's original order for
    # determinism.  We need an explicit stable sort because Python's sort
    # is stable on equal keys.
    density_order = sorted(
        REGIME_STRATA,
        key=lambda r: -DENSITY_BY_REGIME[r],
    )
    configs: list[SampleConfig] = []
    for idx in range(n_traj):
        regime = density_order[idx % len(density_order)]
        configs.append(SampleConfig(
            idx=idx,
            regime=regime,
            rng_seed=base_seed + idx,
            relative_density=DENSITY_BY_REGIME[regime],
        ))
    return configs


def _build_cif_cache(cif_spec: CifSpec) -> _CifCache:
    """Build the cached shell_target + measured-g3 distribution for a CIF.

    Called once per CIF (not per trajectory) — see MACE_RELAX_PILOT.md
    §14.5 for why measure_g3 is the expensive step we cache here.
    """
    ref = ase_read(str(cif_spec.cif_path), format="cif")
    shell = tc.CoordinationShellTarget.from_atoms(ref, phi_num_bins=90)
    dist = G3Distribution(ref, label="ref")
    dist.measure_g3(r_max=10.0, r_step=0.1, phi_num_bins=90,
                    show_progress=False)
    return _CifCache(ref_atoms=ref, shell=shell, distribution=dist)


def pack_supercell(cif_spec: CifSpec, cache: _CifCache, regime: str,
                    rho: float, seed: int, label: str):
    """Tricor packing + grain construction; no shell_relax."""
    cell = tc.Supercell(
        cache.distribution,
        cell_dim_angstroms=cif_spec.cell_size,
        relative_density=rho,
        rng_seed=seed,
        label=label,
    )
    preset = LOCAL_PRESETS[regime]
    summary = cell.generate(
        cache.shell,
        **{**preset, "num_steps": 0},
        refine_orientations=False,
        show_progress=False,
    )
    return cell, cache.shell, summary


def apply_cleanup(cell, shell):
    if CLEANUP_METHOD == "none":
        return
    if CLEANUP_METHOD == "bond_relax":
        cell.bond_relax(shell, n_iter=BOND_RELAX_N_ITER,
                        max_step=BOND_RELAX_MAX_STEP)
    elif CLEANUP_METHOD == "enforce_hard_core":
        cell.enforce_hard_core(shell, n_iter=HARDCORE_N_ITER,
                                push_fraction=HARDCORE_PUSH_FRACTION)
    else:
        raise ValueError(f"Unknown CLEANUP_METHOD: {CLEANUP_METHOD!r}")


def run_mace_relax(atoms, base_calc):
    """FIRE on MACE+wall PES. Wall thresholds derived per-trajectory from
    the bond_relax-cleaned initial structure."""
    r_min_per_pair = per_pair_min_from_atoms(atoms, margin=WALL_MARGIN)
    atoms.calc = MinDistanceWallCalculator(
        base_calc=base_calc, r_min_per_pair=r_min_per_pair,
        k=WALL_K, exponent=WALL_EXPONENT,
    )

    snapshots: list[np.ndarray] = []
    snapshot_steps: list[int] = []
    energy_history: list[float] = []
    best = {"E": float("inf"), "pos": atoms.positions.copy()}

    e0 = float(atoms.get_potential_energy())
    f0 = atoms.get_forces()
    fmax_initial = float(np.abs(f0).max())
    snapshots.append(atoms.positions.copy().astype(np.float32))
    snapshot_steps.append(0)
    energy_history.append(e0)
    best["E"] = e0
    best["pos"] = atoms.positions.copy()

    opt = FIRE(atoms, maxstep=OPT_MAXSTEP, logfile=None)

    def per_step_callback():
        e = float(atoms.get_potential_energy())
        energy_history.append(e)
        if e < best["E"]:
            best["E"] = e
            best["pos"] = atoms.positions.copy()
        if opt.nsteps % TRAJECTORY_STRIDE == 0:
            snapshots.append(atoms.positions.copy().astype(np.float32))
            snapshot_steps.append(int(opt.nsteps))

    opt.attach(per_step_callback, interval=1)
    opt.run(fmax=FMAX_TARGET, steps=N_STEPS)

    if snapshot_steps[-1] != int(opt.nsteps):
        snapshots.append(atoms.positions.copy().astype(np.float32))
        snapshot_steps.append(int(opt.nsteps))

    return {
        "positions":      np.stack(snapshots, axis=0),
        "snapshot_steps": np.asarray(snapshot_steps, dtype=np.int32),
        "loss":           np.asarray(energy_history, dtype=np.float64),
        "best_positions": best["pos"].astype(np.float32),
        "initial_loss":   float(energy_history[0]),
        "best_loss":      float(best["E"]),
        "final_loss":     float(energy_history[-1]),
        "fmax_initial":   fmax_initial,
        "fmax_final":     float(np.abs(atoms.get_forces()).max()),
        "n_opt_steps":    int(opt.nsteps),
        "wall_thresholds": r_min_per_pair,
    }


def _trajectory_filename(cif_spec: CifSpec, cfg: SampleConfig) -> str:
    return (
        f"{cif_spec.system_id}_{cfg.regime}_cell{int(cif_spec.cell_size):03d}_"
        f"idx{cfg.idx:05d}_seed{cfg.rng_seed:09d}.npz"
    )


def run_trajectory(cif_spec: CifSpec, cfg: SampleConfig, cache: _CifCache,
                    calc, out_dir: Path,
                    failure_log: Path) -> dict | None:
    outfile = out_dir / _trajectory_filename(cif_spec, cfg)
    if outfile.is_file():
        print(f"  [{cfg.idx+1:4d}] SKIP (exists)  {outfile.name}")
        return None

    t0 = time.perf_counter()
    label = (f"{cif_spec.system_id}_{cfg.regime}_idx{cfg.idx:05d}_"
              f"seed{cfg.rng_seed:09d}")
    try:
        cell, shell, summary = pack_supercell(
            cif_spec, cache, cfg.regime, cfg.relative_density, cfg.rng_seed,
            label,
        )
        initial_positions = cell.atoms.positions.astype(np.float32).copy()
        apply_cleanup(cell, shell)
        h = run_mace_relax(cell.atoms, calc)
    except Exception as exc:
        # CUDA OOM bubbles up to run_cif so it can shrink + retry.  Free
        # GPU memory first so the retry's forward pass starts from a
        # clean slate.
        if _is_cuda_oom(exc):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        # Non-OOM failure: log and move on — one bad trajectory shouldn't
        # kill a long CIF let alone a multi-day generation.  Persist the
        # traceback to a sidecar CSV for post-hoc audit.
        print(f"  [{cfg.idx+1:4d}] FAILED  {cfg.regime}  seed={cfg.rng_seed}  "
              f"{type(exc).__name__}: {exc}")
        _append_failure(
            failure_log,
            cif_spec=cif_spec, regime=cfg.regime, seed=cfg.rng_seed,
            stage="trajectory",
            exc_type=type(exc).__name__, exc_msg=str(exc),
            tb=traceback.format_exc(),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    runtime = time.perf_counter() - t0

    grain_size_actual = float(summary.get("grain_size") or 0.0)
    num_grains = int(summary.get("n_grains") or 0)
    crystalline_fraction = float(summary.get("crystalline_fraction") or 0.0)
    n_crystalline = int(round(num_grains * crystalline_fraction))

    wall_thresholds = h["wall_thresholds"]
    wall_global_min = (float(min(wall_thresholds.values()))
                       if wall_thresholds else 0.0)

    np.savez(
        outfile,
        # --- Core trajectory state ---
        positions=h["positions"],
        snapshot_steps=h["snapshot_steps"],
        initial_positions=initial_positions,
        best_positions=h["best_positions"],
        final_positions=cell.atoms.positions.astype(np.float32),
        species_numbers=cell.atoms.numbers.astype(np.int32),
        cell=np.asarray(cell.atoms.cell.array, dtype=np.float32),
        loss_history=h["loss"],
        # --- Sampling metadata ---
        idx=np.int64(cfg.idx),
        source=np.asarray(SOURCE_TAG),
        regime=np.asarray(cfg.regime),
        rng_seed=np.int64(cfg.rng_seed),
        # `system_id` = "{compound}_{mp_id}" kept for downstream
        # backward-compat (make_experiment_manifests.py reads it).
        system_id=np.asarray(cif_spec.system_id),
        compound=np.asarray(cif_spec.compound),
        mp_id=np.asarray(cif_spec.mp_id),
        # --- Model conditioning (trajectory-level) ---
        grain_size=np.float32(grain_size_actual),
        num_grains=np.int32(num_grains),
        crystalline_fraction=np.float32(crystalline_fraction),
        rel_density=np.float32(cfg.relative_density),
        wall_global_min=np.float32(wall_global_min),
        fmax_initial=np.float32(h["fmax_initial"]),
        # --- Run settings ---
        num_steps=np.int32(h["n_opt_steps"]),
        trajectory_stride=np.int32(TRAJECTORY_STRIDE),
        cell_size=np.float32(cif_spec.cell_size),
        cell_overridden=np.bool_(cif_spec.cell_overridden),
        n_crystalline=np.int32(n_crystalline),
        # --- Quality (filtering only, not model input) ---
        initial_loss=np.float64(h["initial_loss"]),
        best_loss=np.float64(h["best_loss"]),
        final_loss=np.float64(h["final_loss"]),
        fmax_final=np.float32(h["fmax_final"]),
        # --- Provenance ---
        backend=np.asarray("mace+wall"),
        optimizer=np.asarray("FIRE"),
        mace_model=np.asarray(MACE_MODEL),
        mace_dtype=np.asarray(MACE_DEFAULT_DTYPE),
        cleanup_method=np.asarray(CLEANUP_METHOD),
        wall_k=np.float32(WALL_K),
        wall_exponent=np.int32(WALL_EXPONENT),
    )

    n_atoms = len(cell.atoms)
    size_mb = outfile.stat().st_size / (1024 * 1024)
    print(
        f"  [{cfg.idx+1:4d}] {cfg.regime:>17s} seed={cfg.rng_seed:9d}  "
        f"atoms={n_atoms:5d}  E {h['initial_loss']:9.2f}→{h['final_loss']:9.2f} "
        f"fmax {h['fmax_initial']:5.2f}→{h['fmax_final']:5.2f}  "
        f"wall_min={wall_global_min:.2f}  "
        f"{runtime:6.1f}s  {size_mb:4.1f}MB"
    )

    return {
        "system_id": cif_spec.system_id,
        "compound": cif_spec.compound,
        "mp_id": cif_spec.mp_id,
        "idx": int(cfg.idx),
        "regime": cfg.regime,
        "rng_seed": int(cfg.rng_seed),
        "n_atoms": int(n_atoms),
        "cell_size": float(cif_spec.cell_size),
        "cell_overridden": bool(cif_spec.cell_overridden),
        "grain_size": float(grain_size_actual),
        "num_grains": int(num_grains),
        "crystalline_fraction": float(crystalline_fraction),
        "rel_density": float(cfg.relative_density),
        "wall_global_min": float(wall_global_min),
        "fmax_initial": float(h["fmax_initial"]),
        "initial_loss": float(h["initial_loss"]),
        "best_loss": float(h["best_loss"]),
        "final_loss": float(h["final_loss"]),
        "fmax_final": float(h["fmax_final"]),
        "num_opt_steps": int(h["n_opt_steps"]),
        "runtime_s": float(runtime),
        "filename": outfile.name,
    }


def write_manifest(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── Sidecar CSV loggers (append-only, header lazy-written) ──────────────────


_FAILURE_HEADERS = (
    "cif_filename", "compound", "mp_id", "cell_size", "regime", "rng_seed",
    "stage", "exception_class", "exception_message", "traceback",
)


def _append_failure(path: Path, *, cif_spec: CifSpec | None,
                     regime: str | None, seed: int | None,
                     stage: str, exc_type: str, exc_msg: str,
                     tb: str) -> None:
    """One row per failed trajectory or per-CIF abort.  Robust to missing
    cif_spec (whole-CIF failures before the CifSpec is built)."""
    new_file = not path.is_file()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FAILURE_HEADERS)
        if new_file:
            w.writeheader()
        w.writerow({
            "cif_filename": cif_spec.cif_path.name if cif_spec else "",
            "compound":     cif_spec.compound if cif_spec else "",
            "mp_id":        cif_spec.mp_id if cif_spec else "",
            "cell_size":    f"{cif_spec.cell_size:.2f}" if cif_spec else "",
            "regime":       regime or "",
            "rng_seed":     str(seed) if seed is not None else "",
            "stage":        stage,
            "exception_class": exc_type,
            "exception_message": exc_msg,
            "traceback":    tb,
        })


_CAL_HEADERS = ("cif_filename", "compound", "mp_id", "estimated_n_atoms",
                 "target_cell", "calibrated_cell", "n_shrinks", "reason")


def _load_calibration_cache(path: Path) -> dict[str, float]:
    """Map ``cif_filename → calibrated_cell`` from the on-disk cache.

    Used at run_cif start so a CIF that's been calibrated in a previous
    run starts at its known-good cell size, skipping the OOM probe.
    Returns {} if the file doesn't exist yet (first run).  Entries with
    empty calibrated_cell (CIFs that hit MIN_CELL on probe) are skipped
    by the consumer.
    """
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                out[row["cif_filename"]] = float(row["calibrated_cell"])
            except (KeyError, ValueError):
                continue  # MIN_CELL-skipped rows have empty calibrated_cell
    return out


def _save_calibration_entry(path: Path, *, cif_path: Path, compound: str,
                              mp_id: str, estimated_n_atoms: int,
                              target_cell: float,
                              calibrated_cell: float | None,
                              n_shrinks: int, reason: str) -> None:
    """Append one calibration result.  ``calibrated_cell=None`` records a
    MIN_CELL skip (CIF couldn't be fit at any cell ≥ MIN_CELL)."""
    new_file = not path.is_file()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_CAL_HEADERS)
        if new_file:
            w.writeheader()
        w.writerow({
            "cif_filename": cif_path.name,
            "compound": compound,
            "mp_id": mp_id,
            "estimated_n_atoms": estimated_n_atoms,
            "target_cell": f"{target_cell:.2f}",
            "calibrated_cell": (f"{calibrated_cell:.2f}"
                                  if calibrated_cell is not None else ""),
            "n_shrinks": str(n_shrinks),
            "reason": reason,
        })


# ── CIF discovery + per-CIF orchestration ────────────────────────────────────


def list_cifs() -> list[Path]:
    """Resolve CIF_DIR (+ optional CIF_LIST_FILE) into a sorted list of paths."""
    if not CIF_DIR.is_dir():
        raise SystemExit(f"CIF_DIR not found: {CIF_DIR}")
    if CIF_LIST_FILE is not None and Path(CIF_LIST_FILE).is_file():
        names = [
            line.strip() for line in Path(CIF_LIST_FILE).read_text().splitlines()
            if line.strip()
        ]
        paths = [CIF_DIR / n for n in names]
    else:
        paths = sorted(CIF_DIR.glob("*.cif"))
    paths = [p for p in paths if p.is_file()]
    if MAX_CIFS is not None:
        paths = paths[:MAX_CIFS]
    return paths


def _per_cif_out_dir(compound: str, mp_id: str) -> Path:
    return DATASET_ROOT / f"{compound}_{mp_id}_trajectories"


def _is_cif_already_done(out_dir: Path, n_expected: int) -> bool:
    """Per-CIF resume shortcut: skip the cache build entirely when the
    per-CIF manifest already lists `n_expected` rows and every NPZ exists."""
    manifest = out_dir / "manifest.csv"
    if not manifest.is_file():
        return False
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) < n_expected:
        return False
    return all((out_dir / r["filename"]).is_file() for r in rows)


def _read_existing_manifest(out_dir: Path) -> list[dict]:
    p = out_dir / "manifest.csv"
    if not p.is_file():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def _clear_partial_trajectories(out_dir: Path) -> int:
    """Delete any NPZs in `out_dir` whose cell-size doesn't match the new
    calibrated cell.  Called when an OOM forces a shrink mid-CIF — the
    already-written NPZs have the now-wrong cell baked into their
    filename and metadata, so we can't keep them.

    Returns the number of files deleted.  This is destructive but bounded
    to ONE CIF; per-CIF resume + cross-CIF manifest stays intact.
    """
    n = 0
    for p in out_dir.glob("*.npz"):
        p.unlink()
        n += 1
    # Manifest is rewritten at end-of-CIF; remove stale one too.
    mp = out_dir / "manifest.csv"
    if mp.is_file():
        mp.unlink()
    return n


def run_cif(cif_path: Path, cif_idx: int, n_cifs: int, calc,
             failure_log: Path, calibration_path: Path,
             calibration_cache: dict[str, float]) -> list[dict]:
    """Generate all trajectories for one CIF.  Returns per-trajectory rows.

    OOM-driven cell calibration:
      * Start at CELL_SIZE (or the cached calibrated value if known).
      * If a trajectory raises CUDA OOM, shrink by SHRINK_FACTOR, wipe any
        partial NPZs (they have the wrong cell baked into filename +
        metadata), and restart the loop.
      * Floor at MIN_CELL — a CIF that OOMs at MIN_CELL gets logged and
        skipped (no more shrinks; would compromise crystalline_30 fidelity).

    All other exceptions are caught at the trajectory level inside
    run_trajectory; they don't trigger the shrink loop.
    """
    try:
        compound, mp_id = parse_cif_name(cif_path)
    except ValueError as exc:
        print(f"[{cif_idx+1}/{n_cifs}] SKIP unparseable: "
              f"{cif_path.name} ({exc})")
        _append_failure(failure_log, cif_spec=None, regime=None, seed=None,
                         stage="parse_cif_name",
                         exc_type=type(exc).__name__, exc_msg=str(exc),
                         tb="")
        return []

    out_dir = _per_cif_out_dir(compound, mp_id)
    if _is_cif_already_done(out_dir, N_TRAJ_PER_CIF):
        print(f"[{cif_idx+1}/{n_cifs}] SKIP done: {compound}/{mp_id} "
              f"({N_TRAJ_PER_CIF} trajectories)")
        return _read_existing_manifest(out_dir)

    # Read CIF — only needed for the atom-count estimate display + cache build.
    try:
        ref_atoms = ase_read(str(cif_path), format="cif")
    except Exception as exc:
        print(f"[{cif_idx+1}/{n_cifs}] FAIL CIF parse: {cif_path.name} "
              f"({type(exc).__name__}: {exc})")
        _append_failure(failure_log, cif_spec=None, regime=None, seed=None,
                         stage="ase_read",
                         exc_type=type(exc).__name__, exc_msg=str(exc),
                         tb=traceback.format_exc())
        return []

    # Pick initial cell: cached calibration if known, else CELL_SIZE.
    n_est = estimate_atom_count(ref_atoms, CELL_SIZE)
    cached_cell = calibration_cache.get(cif_path.name)
    if cached_cell is not None:
        current_cell = cached_cell
        cache_hit_note = f"  (calibration cached: cell={cached_cell:.1f})"
    else:
        current_cell = CELL_SIZE
        cache_hit_note = ""

    per_cif_seed = BASE_SEED + cif_idx * SEED_STEP
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n══ CIF {cif_idx+1}/{n_cifs}: {compound} / {mp_id} ══")
    print(f"  ref atoms={len(ref_atoms)}  V={ref_atoms.cell.volume:.1f} Å³  "
          f"est_n_atoms@cell={CELL_SIZE}={n_est}  seed_base={per_cif_seed}  "
          f"→ {out_dir}{cache_hit_note}")

    # Cache build is cell-size-independent (just ref_atoms + shell_target +
    # measure_g3 on the unit cell), so it survives shrinks unchanged.
    try:
        t_cache0 = time.perf_counter()
        cache = _build_cif_cache(CifSpec(
            cif_path=cif_path, compound=compound, mp_id=mp_id,
            cell_size=current_cell, cell_overridden=False,
        ))
        cache_dt = time.perf_counter() - t_cache0
        species = sorted({int(z) for z in cache.ref_atoms.numbers})
        print(f"  cache built in {cache_dt:.1f}s  species Z={species}")
    except Exception as exc:
        print(f"[{cif_idx+1}/{n_cifs}] FAIL cache build: {compound}/{mp_id} "
              f"({type(exc).__name__}: {exc})")
        _append_failure(failure_log,
                         cif_spec=None, regime=None, seed=None,
                         stage="build_cache",
                         exc_type=type(exc).__name__, exc_msg=str(exc),
                         tb=traceback.format_exc())
        return []

    # ── OOM-driven retry loop ────────────────────────────────────────────────
    n_shrinks = 0
    while True:
        cif_spec = CifSpec(
            cif_path=cif_path, compound=compound, mp_id=mp_id,
            cell_size=current_cell,
            cell_overridden=(abs(current_cell - CELL_SIZE) > 1e-6),
        )
        configs = build_configs(N_TRAJ_PER_CIF, per_cif_seed)

        try:
            rows: list[dict] = []
            for cfg in configs:
                row = run_trajectory(
                    cif_spec, cfg, cache, calc, out_dir, failure_log,
                )
                if row is not None:
                    rows.append(row)
            # The loop completed without OOM, but that's only weak
            # evidence the cell fits — every trajectory might have failed
            # with a non-OOM error (run_trajectory catches those and
            # returns None).  Require at least one actual success before
            # persisting calibration; otherwise the cache fills up with
            # false positives and any bug that kills every trajectory
            # gets silently locked in.
            if rows and cached_cell is None:
                _save_calibration_entry(
                    calibration_path, cif_path=cif_path,
                    compound=compound, mp_id=mp_id,
                    estimated_n_atoms=n_est, target_cell=CELL_SIZE,
                    calibrated_cell=current_cell, n_shrinks=n_shrinks,
                    reason=("CELL_SIZE fits" if n_shrinks == 0 else
                             f"OOM-shrunk {n_shrinks}× to {current_cell:.2f}"),
                )
                calibration_cache[cif_path.name] = current_cell
            write_manifest(rows, out_dir / "manifest.csv")
            if not rows:
                print(f"  WARN: {len(configs)} trajectories failed at "
                      f"cell={current_cell:.1f} (none were OOM; see "
                      f"failures CSV).  Skipping calibration persist.")
            else:
                print(f"  done: {len(rows)}/{len(configs)} written "
                      f"at cell={current_cell:.1f} (skips/failures "
                      f"excluded; OOM shrinks: {n_shrinks})")
            return rows

        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            # OOM bubbled up from run_trajectory.  Free GPU state, shrink,
            # nuke any NPZs that got written before the OOM (they have the
            # now-wrong cell in their filename + metadata), and retry.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            new_cell = current_cell * SHRINK_FACTOR
            n_partial = _clear_partial_trajectories(out_dir)
            if new_cell < MIN_CELL:
                print(f"  OOM at cell={current_cell:.1f} Å, would shrink to "
                      f"{new_cell:.1f} Å (< MIN_CELL={MIN_CELL}).  "
                      f"SKIP CIF.")
                _append_failure(failure_log, cif_spec=cif_spec,
                                 regime=None, seed=None,
                                 stage="oom_below_min_cell",
                                 exc_type=type(exc).__name__,
                                 exc_msg=str(exc),
                                 tb=traceback.format_exc())
                _save_calibration_entry(
                    calibration_path, cif_path=cif_path,
                    compound=compound, mp_id=mp_id,
                    estimated_n_atoms=n_est, target_cell=CELL_SIZE,
                    calibrated_cell=None, n_shrinks=n_shrinks + 1,
                    reason=(f"OOM at cell={current_cell:.2f}; next "
                             f"shrink {new_cell:.2f} < MIN_CELL={MIN_CELL}"),
                )
                return []
            n_shrinks += 1
            print(f"  OOM at cell={current_cell:.1f}; shrink #{n_shrinks} "
                  f"→ {new_cell:.1f} Å  (wiped {n_partial} partial NPZs, "
                  f"will restart loop)")
            current_cell = new_cell


# ── Multi-GPU partitioning (CIF-index round-robin) ───────────────────────────


def _spawn_workers(cif_paths: list[Path]) -> int:
    """Master mode: spawn one worker subprocess per GPU in GPU_IDS, partitioning
    CIF indices round-robin.  Returns aggregate non-zero exit count.
    """
    import subprocess

    n_workers = len(GPU_IDS)
    log_dir = DATASET_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[int, subprocess.Popen, Path]] = []
    print(f"Spawning {n_workers} worker(s) across GPUs {GPU_IDS}")
    n_cifs = len(cif_paths)
    for partition_idx, gpu in enumerate(GPU_IDS):
        my_indices = [i for i in range(n_cifs)
                      if i % n_workers == partition_idx]
        if not my_indices:
            print(f"  GPU {gpu}: no CIFs in partition, skipping")
            continue
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PILOT_CIF_INDICES": ",".join(str(i) for i in my_indices),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
            # Force unbuffered stdout so per-trajectory `print(...)` lines
            # flush to the log file as they happen instead of waiting for
            # the ~4 KB block buffer to fill.  Equivalent to `python -u`.
            "PYTHONUNBUFFERED": "1",
        }
        log_path = log_dir / f"gpu{gpu}.log"
        log_fh = log_path.open("w")
        cmd = [sys.executable, "-u", str(Path(__file__).resolve())]
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh,
                                 stderr=subprocess.STDOUT)
        print(f"  GPU {gpu}  PID {proc.pid}  "
              f"n_cifs={len(my_indices)}  trajectories~={len(my_indices)*N_TRAJ_PER_CIF}  "
              f"log={log_path}")
        procs.append((gpu, proc, log_path))

    print(f"\nWaiting for {len(procs)} worker(s) to finish... "
          f"(tail logs in another shell to watch progress)")
    fail = 0
    for gpu, proc, log_path in procs:
        rc = proc.wait()
        status = "OK" if rc == 0 else f"FAIL (exit {rc})"
        print(f"  GPU {gpu}: {status}  log={log_path}")
        if rc != 0:
            fail += 1

    manifests = sorted(DATASET_ROOT.glob("manifest_all_p*.csv"))
    if manifests:
        print(f"\nPer-partition manifests:")
        for m in manifests:
            n_rows = max(0, sum(1 for _ in m.open()) - 1)
            print(f"  {m}  ({n_rows} trajectories)")

    return fail


def _parse_cif_indices_env(n_cifs: int) -> set[int]:
    """Read PILOT_CIF_INDICES env var.  Format: comma-separated indices,
    e.g. "0,4,8,12".  Returns the full set if unset."""
    raw = os.environ.get("PILOT_CIF_INDICES", "").strip()
    if not raw:
        return set(range(n_cifs))
    try:
        indices = {int(x.strip()) for x in raw.split(",") if x.strip()}
    except ValueError as exc:
        raise SystemExit(f"PILOT_CIF_INDICES parse error: {exc}")
    invalid = indices - set(range(n_cifs))
    if invalid:
        raise SystemExit(
            f"PILOT_CIF_INDICES contains invalid indices {sorted(invalid)} "
            f"(valid range: 0-{n_cifs-1})"
        )
    return indices


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    cif_paths = list_cifs()

    is_worker = bool(os.environ.get("PILOT_CIF_INDICES", "").strip())
    if len(GPU_IDS) >= 1 and not is_worker:
        fail = _spawn_workers(cif_paths)
        if fail:
            raise SystemExit(f"{fail} worker(s) exited with errors")
        return

    n_cifs = len(cif_paths)
    indices_to_run = _parse_cif_indices_env(n_cifs)

    # Partition-aware sidecar paths so concurrent workers don't clobber each
    # other.  The "_p" suffix encodes this worker's partition by GPU label
    # (read from CUDA_VISIBLE_DEVICES) when running under _spawn_workers.
    suffix = ""
    raw = os.environ.get("PILOT_CIF_INDICES", "").strip()
    if raw:
        gpu_label = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
        suffix = f"_p{gpu_label}"
    manifest_path = DATASET_ROOT / f"manifest_all{suffix}.csv"
    failure_log = DATASET_ROOT / f"failures{suffix}.csv"
    # Calibration cache is SHARED across workers — every worker reads the
    # same file at startup and appends as it calibrates new CIFs.  Since
    # appends are atomic at the OS level (one CSV row at a time) and
    # workers process disjoint CIF index sets, no synchronization is
    # needed.  Re-runs read all rows back so previously calibrated CIFs
    # start at their known-good cell size, skipping the probe.
    calibration_path = DATASET_ROOT / CALIBRATION_CACHE_FILENAME
    calibration_cache = _load_calibration_cache(calibration_path)

    print(f"MACE big-dataset generation  (source tag: {SOURCE_TAG})")
    if len(indices_to_run) < n_cifs:
        print(f"  [partition] running {len(indices_to_run)} of {n_cifs} CIFs")
    else:
        print(f"  CIFs: {n_cifs} (all)")
    print(f"  N_TRAJ_PER_CIF = {N_TRAJ_PER_CIF}  → "
          f"~{len(indices_to_run) * N_TRAJ_PER_CIF} trajectories in this partition")
    print(f"  Target cell: {CELL_SIZE} Å  (OOM-driven shrink: factor={SHRINK_FACTOR}, "
          f"MIN_CELL={MIN_CELL} Å)")
    print(f"  Regimes: {REGIME_STRATA}")
    print(f"  MACE: model={MACE_MODEL}  device={MACE_DEVICE}  "
          f"dtype={MACE_DEFAULT_DTYPE}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"  Relax: FIRE steps={N_STEPS}  maxstep={OPT_MAXSTEP}  "
          f"wall(k={WALL_K},exp={WALL_EXPONENT})")
    print(f"  Cleanup: {CLEANUP_METHOD}")
    print(f"  Output: {DATASET_ROOT}")
    print(f"  Failures log: {failure_log}")
    print(f"  Calibration cache: {calibration_path}  "
          f"({len(calibration_cache)} entries loaded)\n")

    print("Initializing MACE-MPA medium (cold start ~30s on first run)...")
    _mace_device = ("cpu" if (MACE_DEVICE in ("cuda", "auto")
                              and not torch.cuda.is_available())
                    else MACE_DEVICE)
    if _mace_device != MACE_DEVICE:
        print(f"  No CUDA device visible — falling back to CPU "
              f"(MACE_DEVICE={MACE_DEVICE!r}).", flush=True)
    calc = mace_mp(model=MACE_MODEL, device=_mace_device,
                    default_dtype=MACE_DEFAULT_DTYPE)
    print(f"MACE ready.  torch.cuda.is_available()={torch.cuda.is_available()}\n")

    t_start = time.perf_counter()
    all_rows: list[dict] = []
    n_done = n_skipped = n_failed = 0
    for cif_idx, cif_path in enumerate(cif_paths):
        if cif_idx not in indices_to_run:
            continue
        try:
            rows = run_cif(cif_path, cif_idx, n_cifs, calc,
                            failure_log=failure_log,
                            calibration_path=calibration_path,
                            calibration_cache=calibration_cache)
        except Exception as exc:
            # Catch-all — a bug in our own code shouldn't kill a multi-day
            # generation either.
            print(f"[{cif_idx+1}/{n_cifs}] FAIL run_cif: {cif_path.name} "
                  f"({type(exc).__name__}: {exc})")
            _append_failure(failure_log,
                             cif_spec=None, regime=None, seed=None,
                             stage="run_cif_uncaught",
                             exc_type=type(exc).__name__, exc_msg=str(exc),
                             tb=traceback.format_exc())
            n_failed += 1
            rows = []

        if rows:
            n_done += 1
            all_rows.extend(rows)
        elif cif_idx in indices_to_run:
            # No rows but didn't crash → either fully skipped or all skipped.
            n_skipped += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_manifest(all_rows, manifest_path)
    elapsed = time.perf_counter() - t_start
    print(f"\nDone.  cifs_run={n_done}  skipped={n_skipped}  "
          f"failed_uncaught={n_failed}  rows_written={len(all_rows)}")
    print(f"Manifest: {manifest_path}")
    if failure_log.is_file():
        n_failure_rows = max(0, sum(1 for _ in failure_log.open()) - 1)
        print(f"Failures log: {failure_log}  ({n_failure_rows} rows)")
    if calibration_path.is_file():
        n_cal_rows = max(0, sum(1 for _ in calibration_path.open()) - 1)
        print(f"Calibration cache: {calibration_path}  ({n_cal_rows} rows)")
    print(f"Wall time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
