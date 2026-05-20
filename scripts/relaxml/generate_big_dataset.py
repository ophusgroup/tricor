"""Breadth-heavy trajectory generation for the foundation-style relaxml run.

Drives the v2 multicomp pipeline over a list of in-scope CIFs (typically
~500 from ``select_training_cifs.py``), generating N_TRAJ_PER_CIF
balanced-regime trajectories per CIF and writing one merged manifest.

Self-contained: does NOT import the per-compound v2 generator's main(), but
does import its sampling + trajectory-running building blocks.  The
per-CIF compound_name, mp_id, refine_orientations, and k_restraint
overrides are threaded through new per-CIF parameters rather than the
module globals that ``generate_surrogate_trajectories_multicomp_v2.py``
relies on for its single-compound flow.

Per CIF:
  1. Resolve compound + mp_id from the filename (mp-XXX_Formula.cif).
  2. Build reference Atoms and CoordinationShellTarget once.
  3. Allocate a deterministic seed range (BASE_SEED + cif_idx * SEED_STEP).
  4. Generate N_TRAJ_PER_CIF stratified samples (balanced regime sweep).
  5. Run them in a per-CIF worker pool.
  6. Append manifest rows to the global manifest.

Outputs land in ``DATASET_ROOT / <compound>_<mp_id>_trajectories/``
(per-CIF subdirs) plus a merged manifest at
``DATASET_ROOT / manifest.csv``.  The merged manifest schema matches
the per-compound v2 generator (so ``add_shell_tgt_to_npz.py`` + the
existing dataloader Just Work, with one new ``mp_id`` column for
disambiguation).

Edit the CONFIG section below, then run:
    python scripts/relaxml/generate_big_dataset.py
"""

from __future__ import annotations

# Limit BLAS threads per worker BEFORE numpy gets imported.
import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

# ══════════════════════════════════════════════════════════════════════════════
# Hard memory cap — self-wrap via systemd-run
# ══════════════════════════════════════════════════════════════════════════════
#
# When launched directly (``python generate_big_dataset.py``), the script
# re-execs itself under ``systemd-run --user --scope`` so a runaway
# allocation can only OOM-kill processes inside this transient cgroup —
# never other users' (sd-pam) anchors on the shared box.  See
# RELAXML_SESSION.txt §19.6 for the incident that motivated this.
#
# The wrap is idempotent (sentinel env var skips re-wrapping when we
# re-enter inside the scope) and is a no-op if systemd-run isn't on PATH
# (prints a warning).
#
# Override the cap without editing the file:
#     RELAXML_MEMCAP=100G python generate_big_dataset.py
#
# Inspect the running scope:
#     systemctl --user status relaxml-bigdata-rest.scope
#
import shutil
import sys

MEMORY_CAP = os.environ.get("RELAXML_MEMCAP", "100G")
SCOPE_UNIT_NAME = "relaxml-bigdata-rest"


def _ensure_memory_cap() -> None:
    """Re-exec self via ``systemd-run --user --scope`` if not already
    inside such a scope.  No-op when systemd-run is unavailable (with a
    warning) or when we're already wrapped (sentinel env var).
    """
    if os.environ.get("_RELAXML_BIGDATA_WRAPPED") == "1":
        return
    if shutil.which("systemd-run") is None:
        print(
            "[warn] systemd-run not on PATH; running WITHOUT memory cap. "
            "On a shared box an OOM here can take down other users' sessions.",
            flush=True,
        )
        return
    os.environ["_RELAXML_BIGDATA_WRAPPED"] = "1"
    cmd = [
        "systemd-run", "--user", "--scope",
        f"--unit={SCOPE_UNIT_NAME}",
        # Belt-and-suspenders: ensure the sentinel reaches the wrapped
        # process even if systemd-run's env-inheritance behavior changes.
        "--setenv=_RELAXML_BIGDATA_WRAPPED=1",
        "-p", f"MemoryMax={MEMORY_CAP}",
        "-p", "MemorySwapMax=0",
        sys.executable, *sys.argv,
    ]
    print(
        f"[mem-cap] re-execing under systemd-run --user --scope "
        f"MemoryMax={MEMORY_CAP} ({SCOPE_UNIT_NAME})",
        flush=True,
    )
    os.execvp("systemd-run", cmd)
    raise RuntimeError("os.execvp returned unexpectedly")


if __name__ == "__main__":
    _ensure_memory_cap()

# ──────────────────────────────────────────────────────────────────────────────
# Heavy imports happen AFTER the wrap so they're not paid twice per launch.

import csv
import multiprocessing as mp
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from tricor.shells import CoordinationShellTarget
from tricor.supercell import Supercell

# Reuse the static sampling helpers from the v2 generator (they don't
# touch the per-compound globals we care about).
from generate_surrogate_trajectories_multicomp_v2 import (
    REGIME_STRATA,
    SampleConfig,
    _jittered_weights,
    _num_grains_for,
    _stratum_counts,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

# Source of in-scope CIFs.  Either:
#   * A directory containing the CIFs to use (every *.cif in there),
#   * Plus optionally a selected_cifs.txt with one filename per line that
#     restricts the run to a subset (output of select_training_cifs.py).
# Switched to the full 2733-CIF e_above_hull≤100 meV set (was the 516-CIF
# training subset).  SKIP_IF_DONE handles the 516 already in DATASET_ROOT,
# so this run only generates trajectories for the ~2217 not-yet-done CIFs.
CIF_DIR = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos_le100meV")
CIF_LIST_FILE = None   # use every *.cif in CIF_DIR

# Output dataset root.  Per-CIF subdirs land here; merged manifest at the
# top.  Reusing the relaxml_big_v1 root so the final manifest covers all
# 2733 CIFs in one place.
DATASET_ROOT = Path("/wigeon/users/ehrdt/prod/relaxml_big_rest")

# Trajectories per CIF.  ~516 CIFs × 20 ≈ 10320 trajectories target.
N_TRAJ_PER_CIF = 20

# Supercell + tricor knobs.
CELL_SIZE = 50.0
REL_DENSITY = 0.96
N_STEPS_DEFAULT = 200
TRAJECTORY_STRIDE = 5
REFINE_ORIENTATIONS = True
K_RESTRAINT = 0.0

# Seed plan: each CIF gets a contiguous block of [BASE_SEED + cif_idx * SEED_STEP,
# BASE_SEED + cif_idx * SEED_STEP + N_TRAJ_PER_CIF) seeds.  SEED_STEP must
# exceed N_TRAJ_PER_CIF so blocks don't overlap; 1000 gives lots of margin.
BASE_SEED = 1_000_000
SEED_STEP = 1000

# Worker pool size per CIF.  The pool is rebuilt per CIF (reference Atoms
# changes), so each CIF pays ~1-2s pool-startup overhead.
NUM_WORKERS = 8

# Recycle each worker after this many trajectories.  Each recycle pays
# ~5-10s (spawn context re-imports everything + _init_worker rebuilds
# the reference Atoms + CoordinationShellTarget).  In return, any
# accumulated state inside the worker (numba JIT cache growth, tricor
# internal caches, potential shell_relax leaks) is wiped clean.  At 5
# trajectories/worker recycle with 20 trajectories per CIF and 4 workers
# = ~5 recycles per CIF → ~40s extra per CIF (≪ generation cost).  Set
# to None to disable recycling (workers persist for the full CIF).
MAXTASKS_PER_CHILD = None

# Optional: stop after this many CIFs (None = run them all).  Useful for
# smoke tests / partial generation.
MAX_CIFS = None

# Skip a CIF whose per-CIF output dir already has a complete manifest of
# the expected size.  Lets you resume after an interrupted run.
SKIP_IF_DONE = True

# Diagnostic: print per-trajectory RSS so memory growth can be tracked.
# Reads /proc/self/status (no extra dependency).  Cheap (~1 ms per
# trajectory).  Useful for finding per-trajectory memory leaks — combine
# with NUM_WORKERS=1 + MAX_CIFS=1 + MAXTASKS_PER_CHILD=None to see growth
# pattern inside a single worker over a full CIF.
LOG_MEMORY_PER_TRAJECTORY = False

# Additional dataset roots to check for already-done CIFs.  Useful when
# running a follow-up generation into a SEPARATE DATASET_ROOT while still
# wanting to skip CIFs already covered by a previous run (e.g. running
# the "rest" of the 2733-CIF set into relaxml_big_rest while skipping the
# 516 already done under relaxml_big_v1).  Completions found here count
# as "done" for skipping purposes but their existing rows are NOT
# included in the new merged manifest — the new manifest stays clean.
PREVIOUS_DATASET_ROOTS: list[Path] = [
    Path("/wigeon/users/ehrdt/prod/relaxml_big_v1"),
]

# ══════════════════════════════════════════════════════════════════════════════


_MP_ID_RE = re.compile(r"^(mp-\d+)_(.+)$")


def parse_cif_name(cif_path: Path) -> tuple[str, str]:
    """Return (compound_name, mp_id) parsed from ``mp-XXX_Formula.cif``."""
    m = _MP_ID_RE.match(cif_path.stem)
    if not m:
        raise ValueError(
            f"Could not parse mp-id + formula from filename: {cif_path.name}. "
            "Expected pattern 'mp-NNNN_Formula.cif'."
        )
    return m.group(2), m.group(1)


def build_stratified_samples_for_cif(
    rng: np.random.Generator,
    n_total: int,
    cell_size: float,
    base_seed: int,
) -> list[SampleConfig]:
    """Local copy of v2's build_stratified_samples that takes per-CIF n_total
    and base_seed as args instead of reading module globals.  Otherwise
    identical sampling logic — same REGIME_STRATA, same n_crystalline ≥ 1
    rule, same _jittered_weights call."""
    configs: list[SampleConfig] = []
    counts = _stratum_counts_for(n_total)
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
                rng_seed=base_seed + idx,
                num_steps=preset_num_steps,
            ))
            idx += 1
    return configs


def _stratum_counts_for(total: int) -> list[int]:
    """Local mirror of v2's _stratum_counts (which uses the module's
    REGIME_STRATA).  Re-imports REGIME_STRATA in case we ever override."""
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


# ── Per-worker state set by _init_worker ─────────────────────────────────────
_WORKER_REF = None
_WORKER_SHELL_TARGET = None
_WORKER_CELL_SIZE: float = 0.0
_WORKER_REL_DENSITY: float = 0.0
_WORKER_OUT_DIR: Path | None = None
_WORKER_COMPOUND_NAME: str = ""
_WORKER_MP_ID: str = ""


def _rss_gb() -> float:
    """Process RSS in GB, read from /proc/self/status.  No external dep."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 0.0


def _init_worker(
    cif_path_str: str,
    compound_name: str,
    mp_id: str,
    cell_size: float,
    rel_density: float,
    out_dir_str: str,
) -> None:
    global _WORKER_REF, _WORKER_SHELL_TARGET
    global _WORKER_CELL_SIZE, _WORKER_REL_DENSITY, _WORKER_OUT_DIR
    global _WORKER_COMPOUND_NAME, _WORKER_MP_ID
    _WORKER_REF = ase_read(cif_path_str, format="cif")
    _WORKER_SHELL_TARGET = CoordinationShellTarget.from_atoms(_WORKER_REF)
    _WORKER_CELL_SIZE = float(cell_size)
    _WORKER_REL_DENSITY = float(rel_density)
    _WORKER_OUT_DIR = Path(out_dir_str)
    _WORKER_COMPOUND_NAME = str(compound_name)
    _WORKER_MP_ID = str(mp_id)


def _run_one_worker(cfg: SampleConfig) -> dict | None:
    try:
        row = _run_one_trajectory(
            cfg,
            _WORKER_REF,
            _WORKER_SHELL_TARGET,
            _WORKER_CELL_SIZE,
            _WORKER_REL_DENSITY,
            _WORKER_OUT_DIR,
            _WORKER_COMPOUND_NAME,
            _WORKER_MP_ID,
        )
        if LOG_MEMORY_PER_TRAJECTORY:
            print(
                f"[pid={os.getpid()} {_WORKER_COMPOUND_NAME}-{_WORKER_MP_ID} "
                f"traj={cfg.idx:4d}] RSS={_rss_gb():.2f} GB",
                flush=True,
            )
        return row
    except Exception as e:
        print(
            f"[{_WORKER_COMPOUND_NAME}-{_WORKER_MP_ID}  "
            f"{cfg.idx+1:4d}] FAILED: {type(e).__name__}: {e}",
            flush=True,
        )
        return None


def _run_one_trajectory(
    cfg: SampleConfig,
    ref,
    shell_target,
    cell_size: float,
    rel_density: float,
    out_dir: Path,
    compound_name: str,
    mp_id: str,
) -> dict:
    """One Supercell.generate run; dumps a single .npz."""
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
        f"{compound_name}_{mp_id}_{cfg.anchor_regime}_cell{int(cell_size):03d}_"
        f"idx{cfg.idx:05d}_seed{cfg.rng_seed:09d}.npz"
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
        # Big-dataset additions — compound/mp_id stamped in the .npz so a
        # stray file can be traced back to its source CIF without the
        # manifest.
        compound=np.asarray(compound_name),
        mp_id=np.asarray(mp_id),
    )

    n_atoms = len(sc.atoms)
    size_mb = outfile.stat().st_size / (1024 * 1024)
    print(
        f"[{compound_name}-{mp_id}  {cfg.idx+1:4d}] "
        f"{cfg.anchor_regime:>16s} "
        f"gs={cfg.grain_size:5.1f} ng={cfg.num_grains:3d} nc={cfg.n_crystalline:3d} "
        f"cf={cfg.crystalline_fraction:.3f} atoms={n_atoms:5d} "
        f"loss {summary['initial_loss']:6.2f}→{summary['final_loss']:6.2f} "
        f"best={summary['best_loss']:6.2f}  {runtime:5.1f}s  {size_mb:4.1f}MB",
        flush=True,
    )

    return {
        "idx": int(cfg.idx),
        "compound": compound_name,
        "mp_id": mp_id,
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


def _per_cif_out_dir(compound_name: str, mp_id: str) -> Path:
    return DATASET_ROOT / f"{compound_name}_{mp_id}_trajectories"


def _is_already_done(out_dir: Path, expected: int) -> bool:
    manifest = out_dir / "manifest.csv"
    if not manifest.is_file():
        return False
    with manifest.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) < expected:
        return False
    for row in rows:
        if not (out_dir / row["filename"]).is_file():
            return False
    return True


def _write_per_cif_manifest(manifest: list[dict], out_dir: Path) -> None:
    if not manifest:
        return
    headers = list(manifest[0].keys())
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in manifest:
            writer.writerow(row)


def _read_existing_manifest(out_dir: Path) -> list[dict]:
    p = out_dir / "manifest.csv"
    if not p.is_file():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    if not CIF_DIR.is_dir():
        raise SystemExit(f"CIF_DIR not found: {CIF_DIR}")

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    if CIF_LIST_FILE is not None and Path(CIF_LIST_FILE).is_file():
        names = [
            line.strip() for line in Path(CIF_LIST_FILE).read_text().splitlines()
            if line.strip()
        ]
        cif_paths = [CIF_DIR / n for n in names]
    else:
        cif_paths = sorted(CIF_DIR.glob("*.cif"))

    cif_paths = [p for p in cif_paths if p.is_file()]
    if MAX_CIFS is not None:
        cif_paths = cif_paths[:MAX_CIFS]

    print(f"Running over {len(cif_paths)} CIFs from {CIF_DIR}.")
    print(f"Output: {DATASET_ROOT}/")
    print(f"Per-CIF target: {N_TRAJ_PER_CIF} trajectories, {NUM_WORKERS} workers.")
    print(f"REFINE_ORIENTATIONS={REFINE_ORIENTATIONS}  K_RESTRAINT={K_RESTRAINT}")
    print()

    merged_manifest: list[dict] = []
    t_start = time.perf_counter()
    n_done = n_skipped = n_failed_full = 0

    for cif_idx, cif_path in enumerate(cif_paths):
        try:
            compound_name, mp_id = parse_cif_name(cif_path)
        except ValueError as e:
            print(f"[{cif_idx+1}/{len(cif_paths)}] SKIP unparseable: {cif_path.name} ({e})")
            continue

        out_dir = _per_cif_out_dir(compound_name, mp_id)
        # Resume check in current root: completed work goes into the new
        # merged manifest so the new run's manifest is self-contained.
        if SKIP_IF_DONE and _is_already_done(out_dir, N_TRAJ_PER_CIF):
            print(
                f"[{cif_idx+1}/{len(cif_paths)}] SKIP done (current root): "
                f"{compound_name}/{mp_id} ({N_TRAJ_PER_CIF} trajectories)"
            )
            merged_manifest.extend(_read_existing_manifest(out_dir))
            n_skipped += 1
            continue
        # Cross-root check: skip CIFs already done under any previous
        # dataset root, but DON'T merge their rows into the new manifest
        # (those trajectories belong to the old dataset, not this run).
        if SKIP_IF_DONE and any(
            _is_already_done(
                prev / f"{compound_name}_{mp_id}_trajectories",
                N_TRAJ_PER_CIF,
            )
            for prev in PREVIOUS_DATASET_ROOTS
        ):
            print(
                f"[{cif_idx+1}/{len(cif_paths)}] SKIP done (previous root): "
                f"{compound_name}/{mp_id}"
            )
            n_skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        per_cif_seed = BASE_SEED + cif_idx * SEED_STEP

        rng = np.random.default_rng(per_cif_seed)
        configs = build_stratified_samples_for_cif(
            rng, N_TRAJ_PER_CIF, CELL_SIZE, per_cif_seed,
        )

        print(
            f"\n[{cif_idx+1}/{len(cif_paths)}] {compound_name}/{mp_id} → "
            f"{out_dir}  seed_base={per_cif_seed}"
        )

        per_cif_manifest: list[dict] = []
        try:
            if NUM_WORKERS <= 1:
                ref = ase_read(str(cif_path), format="cif")
                shell_target = CoordinationShellTarget.from_atoms(ref)
                for cfg in configs:
                    row = _run_one_trajectory(
                        cfg, ref, shell_target, CELL_SIZE, REL_DENSITY,
                        out_dir, compound_name, mp_id,
                    )
                    if LOG_MEMORY_PER_TRAJECTORY:
                        print(
                            f"[serial pid={os.getpid()} "
                            f"{compound_name}-{mp_id} "
                            f"traj={cfg.idx:4d}] RSS={_rss_gb():.2f} GB",
                            flush=True,
                        )
                    per_cif_manifest.append(row)
            else:
                ctx = mp.get_context("spawn")
                with ctx.Pool(
                    processes=NUM_WORKERS,
                    initializer=_init_worker,
                    initargs=(
                        str(cif_path), compound_name, mp_id,
                        CELL_SIZE, REL_DENSITY, str(out_dir),
                    ),
                    maxtasksperchild=MAXTASKS_PER_CHILD,
                ) as pool:
                    for row in pool.imap_unordered(
                        _run_one_worker, configs, chunksize=1,
                    ):
                        if row is not None:
                            per_cif_manifest.append(row)
        except Exception as e:
            print(
                f"[{cif_idx+1}/{len(cif_paths)}] FAIL {compound_name}/{mp_id}: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            n_failed_full += 1
            # Still write whatever partial manifest we collected for this
            # CIF — the rerun will see it via SKIP_IF_DONE only if complete.
            _write_per_cif_manifest(per_cif_manifest, out_dir)
            continue

        _write_per_cif_manifest(per_cif_manifest, out_dir)
        merged_manifest.extend(per_cif_manifest)
        n_done += 1
        elapsed = time.perf_counter() - t_start
        eta = (
            elapsed / max(cif_idx + 1, 1) * (len(cif_paths) - cif_idx - 1) / 60.0
        )
        print(
            f"[{cif_idx+1}/{len(cif_paths)}] {compound_name}/{mp_id} done: "
            f"{len(per_cif_manifest)}/{N_TRAJ_PER_CIF}.  "
            f"ETA {eta:.1f} min remaining.",
            flush=True,
        )

    # Top-level merged manifest.
    elapsed = time.perf_counter() - t_start
    merged_path = DATASET_ROOT / "manifest.csv"
    if merged_manifest:
        headers = list(merged_manifest[0].keys())
        with merged_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in merged_manifest:
                writer.writerow(row)

    print()
    print(
        f"Done.  ran={n_done}  skipped={n_skipped}  failed={n_failed_full}  "
        f"total_rows={len(merged_manifest)}"
    )
    print(f"Manifest: {merged_path}")
    print(f"Total wall time: {elapsed/60.0:.1f} min")


if __name__ == "__main__":
    main()
