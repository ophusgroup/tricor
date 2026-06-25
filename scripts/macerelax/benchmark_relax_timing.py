"""Apples-to-apples relaxation TIMING benchmark: student vs MACE vs pruned-MACE.

Answers one question: for the structures you actually relax, how long does an
80-step relaxation take with
  (A) your trained student model,
  (B) full MACE-MPA + wall (the teacher), and
  (C) the message-passing-PRUNED foundation model from
      "Scalable foundation interatomic potentials via message-passing pruning
       and graph partitioning" (npj Comput. Mater. 2026), via MatterTune,
on the *same* starting structure and the *same* GPU.

WHY A CACHED STRUCTURE
The three relaxers need three different software stacks and likely three
different conda envs:
  - student   : tricor + graphite + mace env
  - mace_full : mace env (mace.calculators.mace_mp)
  - mace_pruned: MatterTune env (forked MACE-MT backbone; no tricor)
Packing + bond_relax cleanup require tricor, which the MatterTune env won't
have.  So the FIRST relaxer you run BUILDS the cleaned structure (needs tricor)
and caches it to STRUCTURE_CACHE as a plain .npz.  Every later run — in any
env — LOADS that npz and relaxes the byte-identical atoms.  That is what makes
the comparison fair: same atoms, same cell, same step count.

TIMING METHOD
  - One throwaway WARMUP relaxation per structure (discarded) absorbs CUDA
    init, cuDNN autotune, JIT/compile, allocator warmup.
  - Then N_REPEAT timed relaxations with torch.cuda.synchronize() around the
    timed region.
  - Force a FIXED number of force evaluations (FIRE fmax=0, student tol=0) so
    per-step numbers are clean and directly comparable to the paper's
    µs/atom/step.
  - Results append to RESULTS_CSV so A/B/C runs (separate invocations, separate
    envs) accumulate into one table you can diff.

USAGE
  # 1. tricor/mace env — builds + caches the structure, times the student:
  /home/ehrdt/miniforge3/envs/mace/bin/python scripts/macerelax/benchmark_relax_timing.py
  #    (set RELAXER = "student")

  # 2. same env — times full MACE on the cached structure:
  #    (set RELAXER = "mace_full")

  # 3. MatterTune env — times the pruned model on the same cached structure:
  #    (set RELAXER = "mace_pruned", fill in the MATTERTUNE_* config)

Then read RESULTS_CSV: total_s and us_per_atom_per_eval side by side.

Per repo convention (memory: no CLI flags): configure via the CONFIG block
below, never argparse.
"""

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

# --- which relaxer to time this invocation ---
#   "student"     → your trained model (needs tricor env; builds the cache)
#   "mace_full"   → full MACE-MPA + wall (needs mace env)
#   "mace_pruned" → pruned foundation model via MatterTune (needs MatterTune env)
RELAXER = "student"

# --- resources ---
GPU_ID      = 0
NUM_THREADS = 4

# --- the structure(s) to benchmark ----------------------------------------
# Pick the SAME (cif, regime, cell, seed) you measured your "~7 s" on, so the
# comparison is honest.  One structure is enough for a timing answer; add more
# tuples to average over a few.
CIF_PATH = "/home/ehrdt/cifs_mp_exp_le100meV/mp-2657_TiO2.cif"   # EDIT ME
REGIME   = "amorphous"
SEED     = 2_000_000
# Supercell edge(s) in Å.  Cubic float or (a,b,c) tuple.  Match your 7-s run.
CELL_DIMS = (100.0, 100.0, 100.0)
REL_DENSITY_OVERRIDE = None    # None → use the regime's production density

# Cached cleaned structure.  Built by the first run, reused by all others.
# Keyed by CIF/regime/cell/seed so different configs don't collide.
STRUCTURE_CACHE_DIR = "/home/ehrdt/tricor/scratch/relax_timing"
RESULTS_CSV         = "/home/ehrdt/tricor/scratch/relax_timing/results.csv"

# --- step counts (the thing under test) -----------------------------------
N_FIRE_STEPS = 80     # force evaluations for mace_full / mace_pruned
STUDENT_MAX_ITER = 16  # student forward passes.  With training stride k=5 this
                       # is ~80 FIRE-steps-equivalent — the fair match to 80.

# --- timing controls ---
N_WARMUP = 1
N_REPEAT = 3          # median of N_REPEAT timed runs is reported

# --- wall (shared by mace_full / mace_pruned, matches production) ---
USE_WALL      = True
WALL_K        = 1000.0
WALL_EXPONENT = 4
WALL_MARGIN   = 0.0
OPT_MAXSTEP   = 0.3

# --- cleanup before relax (matches generate_* scripts) ---
BOND_RELAX_N_ITER   = 20
BOND_RELAX_MAX_STEP = 0.1

# --- (A) student model ---
# None → reuse generate_with_student.py's MODEL_* run/ckpt resolution.
STUDENT_CKPT_OVERRIDE = None
STUDENT_EDGE_CHUNK    = 2_000_000

# --- (B) full MACE ---
MACE_MODEL        = "medium-mpa-0"
MACE_DEFAULT_DTYPE = "float32"

# --- (C) pruned MACE via MatterTune --------------------------------------
# These mirror the MatterTune "prune & partition" docs + their Li3PO4 example.
# VERIFY against your installed MatterTune version — the API moved around.
# For a TIMING test the weights are irrelevant (forward cost = architecture),
# so any pretrained MACE-OMAT checkpoint at the right layer count works.
MATTERTUNE_CKPT        = "/path/to/mace-omat.ckpt"   # EDIT ME
MATTERTUNE_PRUNE_MP    = 1        # layers to retain = MP×1 (the paper's setting)
MATTERTUNE_DEVICE      = "cuda"
# Set True to also exercise their multi-GPU graph-partitioning calculator
# (MatterTunePartitionCalculator).  Leave False for the single-GPU number.
MATTERTUNE_PARTITION   = False
MATTERTUNE_N_GPUS      = 4

# ═══════════════════════════════════════════════════════════════════════════
# Environment (must precede torch import)
# ═══════════════════════════════════════════════════════════════════════════
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
# Set our GPU BEFORE importing generate_with_student (it uses setdefault, so
# ours wins) and before torch initializes CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(GPU_ID))
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(NUM_THREADS))

import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(NUM_THREADS)

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS / "generation"))   # wall_calculator lives here

# wall_calculator is ASE-only → importable in every env, including MatterTune.
from wall_calculator import MinDistanceWallCalculator, per_pair_min_from_atoms  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Structure cache (build once with tricor, reuse everywhere)
# ═══════════════════════════════════════════════════════════════════════════

def _cell_tag() -> str:
    c = CELL_DIMS if isinstance(CELL_DIMS, (tuple, list)) else (CELL_DIMS,) * 3
    return "x".join(f"{float(v):g}" for v in c)


def _cache_path() -> Path:
    stem = Path(CIF_PATH).stem
    name = f"{stem}__{REGIME}__cell{_cell_tag()}__seed{SEED}.npz"
    return Path(STRUCTURE_CACHE_DIR) / name


def _build_and_cache_structure(path: Path) -> dict:
    """Pack + bond_relax cleanup via the production tricor path, then cache.

    Imports generate_with_student lazily so envs WITHOUT tricor (MatterTune)
    can still run, as long as the cache already exists.
    """
    import generate_with_student as gws

    # Drive gws's single-source helpers with this benchmark's knobs.
    gws.CELL_DIMS = (CELL_DIMS if isinstance(CELL_DIMS, (tuple, list))
                     else (CELL_DIMS,) * 3)
    gws.BOND_RELAX_N_ITER = BOND_RELAX_N_ITER
    gws.BOND_RELAX_MAX_STEP = BOND_RELAX_MAX_STEP

    rho = (REL_DENSITY_OVERRIDE if REL_DENSITY_OVERRIDE is not None
           else gws.DENSITY_BY_REGIME[REGIME])

    print(f"[build] packing {Path(CIF_PATH).name}  regime={REGIME}  "
          f"cell={gws.CELL_DIMS}  rho={rho}  seed={SEED}")
    t0 = time.perf_counter()
    cell, shell, ref, summary = gws.build_supercell(
        Path(CIF_PATH), REGIME, rho, SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gws.cleanup(cell, shell, device=dev)
    print(f"[build] pack+cleanup done in {time.perf_counter()-t0:.1f}s  "
          f"n_atoms={len(cell.atoms)}")

    weight_vector = gws.build_weight_vector(cell, summary, REGIME, rho)
    shell_target  = gws.build_shell_target(ref)

    payload = {
        "positions": cell.atoms.positions.astype(np.float32),
        "cell":      np.asarray(cell.atoms.cell.array, dtype=np.float32),
        "numbers":   cell.atoms.numbers.astype(np.int64),
        "weight_vector": np.asarray(weight_vector, dtype=np.float32),
    }
    for k, v in shell_target.items():        # shell_pair_* / shell_triplet_*
        payload[f"st__{k}"] = np.asarray(v)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    print(f"[build] cached → {path}")
    return payload


def load_structure() -> dict:
    path = _cache_path()
    if path.is_file():
        print(f"[struct] loading cached structure {path}")
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    if RELAXER == "mace_pruned":
        raise SystemExit(
            f"[abort] no cached structure at {path}.\n"
            "        Run RELAXER='student' (or 'mace_full') in the tricor env "
            "first to build it, then re-run the pruned timing here.")
    return _build_and_cache_structure(path)


def _ase_atoms(struct: dict):
    from ase import Atoms
    return Atoms(numbers=struct["numbers"], positions=struct["positions"],
                 cell=struct["cell"], pbc=True)


def _shell_target(struct: dict) -> dict:
    return {k[len("st__"):]: struct[k] for k in struct if k.startswith("st__")}


# ═══════════════════════════════════════════════════════════════════════════
# Relaxers — each returns (n_force_evals, fn) where fn() runs ONE relaxation
# ═══════════════════════════════════════════════════════════════════════════

def _wrap_wall(atoms, base_calc):
    if not USE_WALL:
        atoms.calc = base_calc
        return atoms
    r_min = per_pair_min_from_atoms(atoms, margin=WALL_MARGIN)
    atoms.calc = MinDistanceWallCalculator(
        base_calc=base_calc, r_min_per_pair=r_min,
        k=WALL_K, exponent=WALL_EXPONENT)
    return atoms


def make_fire_relaxer(struct: dict, base_calc):
    """FIRE for a fixed N_FIRE_STEPS evals (fmax=0 → never early-stops)."""
    from ase.optimize import FIRE

    def run():
        atoms = _ase_atoms(struct)
        _wrap_wall(atoms, base_calc)
        opt = FIRE(atoms, maxstep=OPT_MAXSTEP, logfile=None)
        opt.run(fmax=0.0, steps=N_FIRE_STEPS)   # fmax=0 ⇒ exactly N steps
        return atoms.positions

    return N_FIRE_STEPS + 1, run   # +1 for the initial force eval


def build_mace_full_calc():
    from mace.calculators import mace_mp
    print(f"[mace_full] loading {MACE_MODEL} (cold ~30s)…")
    return mace_mp(model=MACE_MODEL, device="cuda",
                   default_dtype=MACE_DEFAULT_DTYPE)


def build_mace_pruned_calc():
    """Pruned foundation-model calculator via MatterTune.

    The exact API differs across MatterTune versions — this follows the
    docs' "prune & partition" page + the Li3PO4 example.  Adjust import
    paths / config field names to your installed version if it errors;
    the contract this function must satisfy is simply: return an ASE
    Calculator whose backbone retains MATTERTUNE_PRUNE_MP message-passing
    layers.  (Weights need not be fine-tuned — we only time the forward.)
    """
    import mattertune as mt   # noqa: F401

    # Single-GPU pruned calculator -----------------------------------------
    backbone = mt.backbones.MACEBackboneModule.load_from_checkpoint(
        MATTERTUNE_CKPT)
    # Retain K message-passing layers (the paper's pruning knob).
    backbone.pruning_message_passing = MATTERTUNE_PRUNE_MP
    if not MATTERTUNE_PARTITION:
        print(f"[mace_pruned] MP×{MATTERTUNE_PRUNE_MP} single-GPU calculator")
        return backbone.ase_calculator()

    # Multi-GPU graph-partitioning calculator (their headline scaling path).
    from mattertune.backbones.parallel import (  # type: ignore
        ParallizedInferenceDDP, MatterTunePartitionCalculator)
    print(f"[mace_pruned] MP×{MATTERTUNE_PRUNE_MP} "
          f"partitioned over {MATTERTUNE_N_GPUS} GPUs")
    ddp = ParallizedInferenceDDP(backbone, num_gpus=MATTERTUNE_N_GPUS)
    return MatterTunePartitionCalculator(ddp)


def make_student_relaxer(struct: dict):
    """Iterative student inference for a fixed STUDENT_MAX_ITER passes."""
    import generate_with_student as gws

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if STUDENT_CKPT_OVERRIDE is not None:
        ckpt_path = Path(STUDENT_CKPT_OVERRIDE)
    else:
        run_dir = gws._resolve_run_dir(
            gws.MODEL_LOG_DIR, gws.MODEL_RUN_NAME, gws.MODEL_RUN_TIMESTAMP)
        ckpt_path = gws._resolve_checkpoint(run_dir, gws.MODEL_EPOCH)

    model = gws._load_model(ckpt_path, dev)
    gws.patch_model_edge_chunking(model, STUDENT_EDGE_CHUNK)

    cell    = struct["cell"].astype(np.float32)
    species = struct["numbers"].astype(np.int64)
    w       = struct["weight_vector"].astype(np.float32)
    shell_target = {k: struct[k] for k in struct if k.startswith("st__")}
    shell_target = {k[len("st__"):]: v for k, v in shell_target.items()}

    def run():
        pos, n_iter, _ = gws.run_iterative_inference(
            model, struct["positions"].astype(np.float32),
            cell, species, w, shell_target, dev,
            cutoff=gws.CUTOFF, max_iter=STUDENT_MAX_ITER,
            tol=0.0,            # tol=0 ⇒ always run the full STUDENT_MAX_ITER
            collect_every=0,
        )
        return pos

    return STUDENT_MAX_ITER, run


# ═══════════════════════════════════════════════════════════════════════════
# Timing
# ═══════════════════════════════════════════════════════════════════════════

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_relaxer(n_evals: int, run, n_atoms: int) -> dict:
    print(f"[warmup] {N_WARMUP} run(s)…", flush=True)
    for _ in range(N_WARMUP):
        run()
        _sync()

    samples = []
    for i in range(N_REPEAT):
        _sync()
        t0 = time.perf_counter()
        run()
        _sync()
        dt = time.perf_counter() - t0
        samples.append(dt)
        print(f"[timed] run {i+1}/{N_REPEAT}: {dt:.3f} s", flush=True)

    total = statistics.median(samples)
    per_eval_ms = total / n_evals * 1e3
    per_atom_eval_us = total / n_evals / n_atoms * 1e6
    return {
        "relaxer": RELAXER,
        "cif": Path(CIF_PATH).name,
        "regime": REGIME,
        "n_atoms": n_atoms,
        "n_evals": n_evals,
        "total_s": round(total, 4),
        "ms_per_eval": round(per_eval_ms, 3),
        "us_per_atom_per_eval": round(per_atom_eval_us, 4),
        "n_repeat": N_REPEAT,
        "samples_s": ";".join(f"{s:.4f}" for s in samples),
    }


def append_results(row: dict) -> None:
    path = Path(RESULTS_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"=== relax timing benchmark · RELAXER={RELAXER} ===")
    print(f"    device CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}  "
          f"cuda={torch.cuda.is_available()}")

    struct = load_structure()
    n_atoms = int(len(struct["numbers"]))
    print(f"    structure: n_atoms={n_atoms}  cell={_cell_tag()} Å")

    if RELAXER == "student":
        n_evals, run = make_student_relaxer(struct)
    elif RELAXER == "mace_full":
        n_evals, run = make_fire_relaxer(struct, build_mace_full_calc())
    elif RELAXER == "mace_pruned":
        n_evals, run = make_fire_relaxer(struct, build_mace_pruned_calc())
    else:
        raise SystemExit(f"[abort] unknown RELAXER={RELAXER!r}")

    row = time_relaxer(n_evals, run, n_atoms)

    print("\n=== RESULT ===")
    for k, v in row.items():
        print(f"  {k:>22}: {v}")
    append_results(row)
    print(f"\nappended → {RESULTS_CSV}")
    print("compare relaxers by `total_s` and `us_per_atom_per_eval` "
          "(the paper reports MACE-M-Omat MP×1 ≈ 111 µs/atom/step single-GPU).")


if __name__ == "__main__":
    main()
