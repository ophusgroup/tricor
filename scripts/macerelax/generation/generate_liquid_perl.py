"""Perlmutter (multi-node SLURM) variant of generate_mace_trajectories.py.

Functionally identical to the local-machine script except for one addition:
when launched under SLURM (SLURM_JOB_NUM_NODES > 1), each node takes a
round-robin slice of the CIF index range based on SLURM_NODEID before
spawning its per-GPU workers.  Local single-machine runs (no SLURM env)
behave exactly like the sibling script.

Recommended launch path on Perlmutter:
    sbatch scripts/macerelax/generation/perlmutter_generate.sbatch

The sbatch file launches one srun task per node; this script's main()
reads SLURM_NODEID / SLURM_JOB_NUM_NODES at startup, partitions the
CIF list across nodes, then _spawn_workers further partitions across
the 4 GPUs visible to this node.

Original docstring follows.

────────────────────────────────────────────────────────────────────────

CIF-list-driven MACE+wall trajectory generation for the big dataset.

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
import ctypes
import ctypes.util
import gc
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
CIF_DIR       = Path("/pscratch/sd/e/ehrdt/cifs_mp_exp_le100meV")
CIF_LIST_FILE = None   # None = use every *.cif in CIF_DIR

# --- Output ---
# LIQUID FORK: writes to a SEPARATE dataset root so big_exp stays immutable.
# The liquid trajectories are merged with big_exp at training-assembly time.
DATASET_ROOT  = Path("/pscratch/sd/e/ehrdt/tricor/big_exp_liquid")

# --- Per-CIF generation ---
# LIQUID FORK: one liquid trajectory per CIF, written at idx=6 (the existing
# big_exp run occupies idx 0..5).  IDX_OFFSET shifts the trajectory index +
# rng_seed so the (idx, seed) pair is globally unique across the union of
# big_exp + big_exp_liquid and stays reproducible (seed = per_cif_base + idx).
N_TRAJ_PER_CIF = 1
IDX_OFFSET     = 6              # liquid lives at idx00006 / seed = base + 6
BASE_SEED      = 2_000_000      # MUST match big_exp so per-CIF seed blocks line up
SEED_STEP      = 1000           # gap between consecutive CIFs' seed blocks;
                                  # must exceed N_TRAJ_PER_CIF + IDX_OFFSET so
                                  # liquid (idx=6) never overlaps the next CIF.
SOURCE_TAG     = "mace_liquid_v1"  # written into every NPZ's `source` field

# --- Calibration reuse ---
# LIQUID FORK: load the per-CIF calibrated cell sizes from the ORIGINAL big_exp
# run so each liquid trajectory uses the EXACT same cell as that CIF's other 6
# regimes (consistency) and skips the expensive OOM probe.  New calibrations
# (CIFs not present in big_exp) still append to DATASET_ROOT's own cache.
CALIBRATION_LOAD_PATH = Path(
    "/pscratch/sd/e/ehrdt/tricor/big_exp/calibrated_cells.csv")

# --- Optional: stop after this many CIFs (smoke testing) ---
MAX_CIFS = None                  # None = run them all

# --- Multi-GPU spawning ---
# Round-robin partition over CIF indices.  Set GPU_IDS = [<id>] for single
# GPU; GPU_IDS = [] disables pinning entirely (and the workers run in the
# parent process — handy for debugging).
GPU_IDS = [1,2]

# --- Host-memory hygiene + worker recycling (host-RAM OOM mitigation) ---
# A long-lived worker accrues host RSS across CIFs — glibc retains freed pages
# from numba-parallel measure_g3 + per-CIF numpy temporaries (arena
# fragmentation), eventually tripping the SLURM cgroup OOM-killer.  That kill
# is an uncatchable SIGKILL, so the CUDA-OOM cell-shrink loop never sees it.
# Two mitigations, both leak-source-agnostic:
#   (a) After each real CIF, gc.collect() + malloc_trim(0) returns freed pages
#       to the OS.  Pair with `export MALLOC_ARENA_MAX=2` in the sbatch.
#   (b) Recycle each GPU worker after WORKER_RECYCLE_EVERY real CIFs — a hard
#       RSS ceiling no matter what residual leak remains.  Safe because the
#       per-CIF manifest resume (_is_cif_already_done) lets a respawned worker
#       fast-forward over finished CIFs and continue where it left off.
HOST_MEM_TRIM_EVERY  = 1      # gc+malloc_trim every N processed CIFs (0=never)
HOST_MEM_LOG         = True   # print per-CIF RSS / peak RSS to the worker log
WORKER_RECYCLE_EVERY = 400    # respawn worker after N real CIFs (0 = disabled)
WORKER_RECYCLE_EXIT_CODE = 23 # worker→master "respawn me" signal (not an error)

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

# LIQUID FORK: only the liquid regime.  `liquid` was originally dropped from
# the big_exp sweep (its endpoints looked close to amorphous at step 60); this
# fork retroactively adds it now that tricor ships a working liquid preset
# (src/tricor/supercell.py PRESETS["liquid"]).
REGIME_STRATA = ["liquid"]
DENSITY_BY_REGIME = {
    # tricor's liquid PRESET sets no density override (it would inherit the
    # Supercell default 0.96), but 0.96 is a non-regime-specific default, not a
    # physical liquid density.  In big_exp's density ladder the disordered
    # regimes sit at 0.92 (amorphous/SRO/LRO/MRO-corrected), so liquid — the
    # most expanded phase — goes just below that floor at 0.90.  Still < the
    # 0.98 crystalline_30 max, so it fits big_exp's calibrated cells.
    "liquid":           0.90,
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
            continue   # synthesized below from tricor's PRESETS isn't applicable
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
MACE_DEVICE         = "cuda"
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
    # LIQUID FORK: index from IDX_OFFSET so trajectories land at idx00006...
    # (big_exp used idx 0..5) and rng_seed = base_seed + idx stays unique.
    for k in range(n_traj):
        idx = IDX_OFFSET + k
        regime = density_order[k % len(density_order)]
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


def _append_manifest(rows: list[dict], out_path: Path) -> None:
    """Append rows to the partition roll-up, creating it (with header) if
    absent.  Used instead of overwrite so the roll-up survives worker
    recycling — each lifetime appends only the CIFs it freshly processed.
    Per-CIF manifests stay authoritative; regenerate the full roll-up from
    them if you need rows from earlier jobs."""
    if not rows:
        return
    new_file = not out_path.is_file()
    fieldnames = list(rows[0].keys())
    with out_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


# ── Host-memory hygiene (host-RAM leak mitigation) ──────────────────────────

_LIBC: "ctypes.CDLL | bool | None" = None


def _libc() -> "ctypes.CDLL | None":
    """Cached handle to libc for malloc_trim; False once if unavailable."""
    global _LIBC
    if _LIBC is None:
        try:
            _LIBC = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        except OSError:
            _LIBC = False
    return _LIBC or None


def _host_rss_mb() -> tuple[float, float]:
    """(current RSS, peak RSS) in MB from /proc/self/status (VmRSS / VmHWM)."""
    rss = hwm = 0.0
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = float(line.split()[1]) / 1024.0
                elif line.startswith("VmHWM:"):
                    hwm = float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return rss, hwm


def _trim_host_memory() -> None:
    """Force a Python GC pass and hand freed glibc-arena pages back to the OS.
    malloc_trim is what actually drops RSS that gc alone leaves resident."""
    gc.collect()
    libc = _libc()
    if libc is not None:
        try:
            libc.malloc_trim(0)
        except (AttributeError, OSError):
            pass


def _cif_is_done(cif_path: Path) -> bool:
    """Cheap resume check used by the worker loop to fast-forward over CIFs
    already complete on disk (so they don't count toward the recycle budget
    or get re-appended to the roll-up)."""
    try:
        compound, mp_id = parse_cif_name(cif_path)
    except ValueError:
        return False
    return _is_cif_already_done(_per_cif_out_dir(compound, mp_id),
                                N_TRAJ_PER_CIF)


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


def _slurm_node_partition(n_cifs: int) -> list[int]:
    """Return the CIF indices this SLURM node should process.

    Reads SLURM_JOB_NUM_NODES and SLURM_NODEID from the environment.  When
    not under SLURM (single-machine launch) both default to 1/0 and this
    returns the full range, behaving identically to the local script.

    Partition is round-robin across nodes so each node sees a mix of
    big/small/dense CIFs (the CIF list is alphabetical, so contiguous
    slices would be unbalanced).
    """
    n_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    node_rank = int(os.environ.get("SLURM_NODEID", "0"))
    if not (0 <= node_rank < n_nodes):
        raise SystemExit(
            f"SLURM_NODEID={node_rank} out of range for "
            f"SLURM_JOB_NUM_NODES={n_nodes}"
        )
    indices = [i for i in range(n_cifs) if i % n_nodes == node_rank]
    if n_nodes > 1:
        print(f"[slurm] node {node_rank}/{n_nodes}: assigned "
              f"{len(indices)}/{n_cifs} CIFs")
    return indices


def _spawn_workers(cif_paths: list[Path], cif_indices: list[int]) -> int:
    """Master mode: spawn one worker subprocess per GPU in GPU_IDS, partitioning
    the provided CIF index slice round-robin across the node's GPUs.

    `cif_indices` is the node-level slice (from _slurm_node_partition); each
    GPU worker gets a further round-robin subset of these.  Returns aggregate
    non-zero exit count.
    """
    import subprocess

    n_workers = len(GPU_IDS)
    log_dir = DATASET_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Node label for log filenames (so multi-node runs don't clobber each
    # other's logs).  Empty string when single-node — preserves the simpler
    # gpuN.log naming on a workstation.
    node_label = ""
    if int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) > 1:
        node_label = f"node{os.environ.get('SLURM_NODEID', '0')}_"

    def _launch(gpu: int, my_indices: list[int], log_path: Path):
        """Spawn one worker subprocess for a GPU's CIF slice; return (proc, fh).
        Log is opened in append mode so recycle respawns share one continuous
        per-GPU log instead of clobbering earlier lifetimes."""
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PILOT_CIF_INDICES": ",".join(str(i) for i in my_indices),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
            # Force unbuffered stdout in the child so per-trajectory
            # `print(...)` lines flush to the log file as they happen
            # instead of waiting for the ~4 KB block buffer to fill.
            # Equivalent to running `python -u`.
            "PYTHONUNBUFFERED": "1",
        }
        fh = log_path.open("a")
        # Belt-and-suspenders: `-u` on the command line too, in case
        # PYTHONUNBUFFERED gets stripped by some intermediate launcher.
        cmd = [sys.executable, "-u", str(Path(__file__).resolve())]
        proc = subprocess.Popen(cmd, env=env, stdout=fh,
                                 stderr=subprocess.STDOUT)
        return proc, fh

    print(f"Spawning {n_workers} worker(s) across GPUs {GPU_IDS}  "
          f"(this node: {len(cif_indices)} CIFs)")
    # gpu -> {proc, fh, indices, log_path, lives}.  Recycled workers
    # (exit WORKER_RECYCLE_EXIT_CODE) are respawned in place with the same
    # CIF slice; per-CIF resume fast-forwards them past finished CIFs.
    active: dict[int, dict] = {}
    for partition_idx, gpu in enumerate(GPU_IDS):
        # Round-robin the node's slice across its GPUs.  Indices are CIF
        # indices in the global cif_paths list — preserved so per-CIF
        # seeds stay deterministic across the cluster.
        my_indices = [cif_indices[i] for i in range(len(cif_indices))
                      if i % n_workers == partition_idx]
        if not my_indices:
            print(f"  GPU {gpu}: no CIFs in partition, skipping")
            continue
        log_path = log_dir / f"{node_label}gpu{gpu}.log"
        proc, fh = _launch(gpu, my_indices, log_path)
        print(f"  GPU {gpu}  PID {proc.pid}  "
              f"n_cifs={len(my_indices)}  trajectories~={len(my_indices)*N_TRAJ_PER_CIF}  "
              f"log={log_path}")
        active[gpu] = {"proc": proc, "fh": fh, "indices": my_indices,
                       "log_path": log_path, "lives": 1}

    print(f"\nWaiting for {len(active)} worker(s) to finish... "
          f"(tail logs in another shell to watch progress)")
    if WORKER_RECYCLE_EVERY:
        print(f"  [recycle] workers respawn every {WORKER_RECYCLE_EVERY} CIFs "
              f"to cap host RAM (exit {WORKER_RECYCLE_EXIT_CODE} = respawn, "
              f"not a failure)")
    fail = 0
    while active:
        for gpu, st in list(active.items()):
            rc = st["proc"].poll()
            if rc is None:
                continue
            st["fh"].close()
            if rc == WORKER_RECYCLE_EXIT_CODE:
                st["lives"] += 1
                print(f"  GPU {gpu}: recycled after lifetime "
                      f"{st['lives'] - 1} — respawning to reclaim host RAM")
                proc, fh = _launch(gpu, st["indices"], st["log_path"])
                st["proc"], st["fh"] = proc, fh
            elif rc == 0:
                print(f"  GPU {gpu}: OK  log={st['log_path']}")
                del active[gpu]
            else:
                print(f"  GPU {gpu}: FAIL (exit {rc})  log={st['log_path']}")
                fail += 1
                del active[gpu]
        if active:
            time.sleep(2)

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
        # Multi-node partition first: each SLURM node takes a round-robin
        # slice of the full CIF list, then _spawn_workers further partitions
        # that slice across the node's 4 GPUs.  Off-SLURM single-machine
        # runs get the full range here (n_nodes=1, node_rank=0).
        this_node_cif_indices = _slurm_node_partition(len(cif_paths))
        fail = _spawn_workers(cif_paths, this_node_cif_indices)
        if fail:
            raise SystemExit(f"{fail} worker(s) exited with errors")
        return

    n_cifs = len(cif_paths)
    indices_to_run = _parse_cif_indices_env(n_cifs)

    # Partition-aware sidecar paths so concurrent workers don't clobber each
    # other.  Under SLURM the suffix encodes both node and GPU; on a single
    # box it encodes just the GPU.  Calibration cache stays unsuffixed so
    # all workers share it (atomic append + disjoint partitions = safe).
    suffix = ""
    raw = os.environ.get("PILOT_CIF_INDICES", "").strip()
    if raw:
        node_part = ""
        if int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) > 1:
            node_part = f"n{os.environ.get('SLURM_NODEID', '0')}_"
        gpu_label = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
        suffix = f"_p{node_part}g{gpu_label}"
    manifest_path = DATASET_ROOT / f"manifest_all{suffix}.csv"
    failure_log = DATASET_ROOT / f"failures{suffix}.csv"
    # Calibration cache is SHARED across workers — every worker reads the
    # same file at startup and appends as it calibrates new CIFs.  Since
    # appends are atomic at the OS level (one CSV row at a time) and
    # workers process disjoint CIF index sets, no synchronization is
    # needed.  Re-runs read all rows back so previously calibrated CIFs
    # start at their known-good cell size, skipping the probe.
    calibration_path = DATASET_ROOT / CALIBRATION_CACHE_FILENAME
    # LIQUID FORK: seed the in-memory cache from big_exp's calibrated cells so
    # liquid reuses each CIF's exact cell + skips the OOM probe.  New CIFs still
    # calibrate and append to DATASET_ROOT's own cache (calibration_path).
    _load_from = CALIBRATION_LOAD_PATH or calibration_path
    calibration_cache = _load_calibration_cache(_load_from)
    if _load_from != calibration_path:
        calibration_cache.update(_load_calibration_cache(calibration_path))

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
    calc = mace_mp(model=MACE_MODEL, device=MACE_DEVICE,
                    default_dtype=MACE_DEFAULT_DTYPE)
    print(f"MACE ready.  torch.cuda.is_available()={torch.cuda.is_available()}\n")

    t_start = time.perf_counter()
    all_rows: list[dict] = []
    n_done = n_skipped = n_failed = 0
    processed_this_life = 0
    recycle_pending = False
    my_indices = [i for i in range(n_cifs) if i in indices_to_run]
    for cif_idx in my_indices:
        cif_path = cif_paths[cif_idx]

        # Cheap resume fast-forward: a CIF already complete on disk costs only
        # a manifest read.  It must NOT count toward the recycle budget and
        # must NOT be re-appended to the partition roll-up (would duplicate).
        if _cif_is_done(cif_path):
            print(f"[{cif_idx+1}/{n_cifs}] SKIP done: {cif_path.stem}")
            n_skipped += 1
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
        else:
            # No rows but didn't crash → all trajectories for this CIF failed.
            n_skipped += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Host-memory hygiene (the real OOM mitigation) ──────────────────
        processed_this_life += 1
        if HOST_MEM_TRIM_EVERY and processed_this_life % HOST_MEM_TRIM_EVERY == 0:
            _trim_host_memory()
        if HOST_MEM_LOG:
            rss, hwm = _host_rss_mb()
            print(f"  [host-mem] after CIF {cif_idx+1}: "
                  f"RSS={rss:,.0f} MB  peak={hwm:,.0f} MB  "
                  f"(life={processed_this_life})", flush=True)

        # ── Worker recycling: hard RSS ceiling regardless of residual leak ──
        if WORKER_RECYCLE_EVERY and processed_this_life >= WORKER_RECYCLE_EVERY:
            recycle_pending = True
            break

    # Append (not overwrite) so the roll-up survives recycling.
    _append_manifest(all_rows, manifest_path)

    if recycle_pending:
        rss, hwm = _host_rss_mb()
        print(f"\n[recycle] {processed_this_life} CIFs processed this "
              f"lifetime (RSS={rss:,.0f} MB  peak={hwm:,.0f} MB).  Exiting "
              f"{WORKER_RECYCLE_EXIT_CODE} so the master respawns a fresh "
              f"worker and reclaims host RAM.", flush=True)
        sys.stdout.flush()
        raise SystemExit(WORKER_RECYCLE_EXIT_CODE)

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
