"""Test 1 — fmax_initial sensitivity of the student model + MACE comparison.

Question: can we use a placeholder fmax_initial at generation time, or
does the student depend on it strongly enough that wrong values produce
qualitatively worse structures?

Test set: diverse chemistry from the composition_test_big eval manifest
(B4C, GeO2, Fe2N, VO2, TiS2, YN).  These are deliberately a more
realistic stress test than the training-distribution chemistries — if the
model handles fmax robustly here, it'll handle the placeholder in
production generation.

Settings match production training (50 Å cells, 60 MACE steps, 15
student iters).  Total cost: ~1.5 GPU-hours for 6 systems.

Method
------
For each of TEST_CIFS at the standard production cell size:
  1. Build packed amorphous supercell + bond_relax cleanup.
  2. Measure the TRUE fmax_initial via one MACE-MPA forward pass on the
     cleaned structure.
  3. Run MACE+wall FIRE for N_STEPS_MACE=60 steps → "ground truth" final.
  4. Run the student model N times (MAX_ITER=15 each) with fmax_initial
     set to each of FMAX_SWEEP values.  For each fmax setting, record:
       - student final structure
       - per-iter |delta| stats
       - RMSE vs MACE ground truth (min-image)
       - PDF MSE vs MACE ground truth

Output (per CIF, written to OUTPUT_ROOT/<system>/):
  - mace_truth.xyz                final-frame MACE+wall structure
  - student_fmax_<v>.xyz          one per fmax_sweep value
  - sensitivity.csv               table of metrics per fmax
  - sensitivity.png               RMSE + PDF MSE vs fmax curves

A summary.csv at OUTPUT_ROOT aggregates across systems.

Verdict criterion
-----------------
"Median fmax suffices" if, for every system tested:
  - max(RMSE) − min(RMSE) across the sweep < TOLERANCE_RMSE_RANGE
  - AND RMSE at the median-fmax setting < MAX_ACCEPTABLE_RMSE
Otherwise the model is fmax-sensitive — the placeholder approach is
unreliable and the retrain-without-fmax_initial plan is needed before
production generation.

Run:
    /global/common/software/m5020/ehrdt/tricor/bin/python \\
        scripts/macerelax/pilot/test_fmax_vs_mace.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# --- resource caps ---
GPU_ID      = 0
NUM_THREADS = 4

# --- inputs ---
# Eval-set CIFs (drawn from composition_test_big_eval held-out systems).
# Picked to span diverse chemistry classes — carbides, nitrides, oxides,
# sulfides, with 3d TM, 4d TM, rare-earth, main-group anions.  These are
# the kind of stress test the model will face in real use.
CIF_DIR    = Path("/pscratch/sd/e/ehrdt/tricor/cifs_mp_cnos_le100meV_training")
TEST_CIFS = (
    "mp-530074_B4C.cif",    # carbide, light main-group
    "mp-223_GeO2.cif",      # network oxide (similar to training SiO2 chemistry)
    "mp-21476_Fe2N.cif",    # 3d TM nitride
    "mp-541404_VO2.cif",    # 3d TM oxide (Mott-Peierls), same as eval example
    "mp-2156_TiS2.cif",     # layered TM sulfide (novel chemistry)
    "mp-2114_YN.cif",       # rare-earth nitride (most OOD)
)

# --- output ---
OUTPUT_ROOT = Path("/pscratch/sd/e/ehrdt/macerelax/pilot_fmax_sensitivity")

# --- model checkpoint ---
MODEL_LOG_DIR        = "/pscratch/sd/e/ehrdt/macerelax/lightning_logs"
MODEL_RUN_NAME       = "ddp_v1"
MODEL_RUN_TIMESTAMP: str | None = None
MODEL_EPOCH          = "best"
USE_EMA_WEIGHTS      = True

# --- architecture (must match training) ---
MAX_Z                    = 120
NODE_DIM                 = 128
EDGE_DIM                 = 128
NUM_CONVS                = 4
WEIGHT_ENCODER_HIDDEN    = 64
SPECIES_PAIR_DIM         = 16
SHELL_TARGET_SPECIES_DIM = 8
SHELL_TARGET_HIDDEN      = 64
SHELL_TARGET_DROPOUT     = 0.0

# --- generation parameters (match production training settings) ---
REGIME            = "amorphous"
SEED              = 2_000_000
# 40 Å matches production's MIN_CELL — generate_mace_trajectories.py's OOM
# calibration loop typically shrunk dense-chemistry systems (Fe2N, TaS2, etc.)
# from 50 Å down to ~40-45 Å before MACE would fit.  We start at 40 here to
# avoid hitting OOM at all for high-density anions/cations.
CELL_DIMS         = (40.0, 40.0, 40.0)
RELATIVE_DENSITY  = 0.92

# --- MACE ground truth ---
# Production settings: 60 FIRE steps to match the training data the student
# was trained on.  ~10 sec per step on big cells → ~10 min/CIF for MACE alone.
MACE_MODEL          = "medium-mpa-0"
MACE_DEVICE         = "cuda:0"
MACE_DEFAULT_DTYPE  = "float32"
N_STEPS_MACE        = 60   # matches generate_mace_trajectories.py production
OPT_MAXSTEP         = 0.3
FMAX_TARGET         = 0.05
WALL_K              = 1000.0
WALL_EXPONENT       = 4
WALL_MARGIN         = 0.0

# --- fmax sweep ---
# Values are in eV/Å.  Picked to span the training-distribution range
# (typical median is ~5-7; we test below, at, and above).
FMAX_SWEEP = (0.0, 1.0, 3.0, 5.0, 7.0, 10.0, 20.0)
# Where in FMAX_SWEEP is the "median" placeholder we'd default to in production?
# Used purely for the verdict logic.
MEDIAN_FMAX_VALUE = 5.0

# --- inference ---
MAX_ITER             = 15
CONVERGENCE_TOL_ANG  = 0.001
CUTOFF               = 5.0
BOND_RELAX_N_ITER    = 80
BOND_RELAX_MAX_STEP  = 0.1

# --- verdict thresholds ---
TOLERANCE_RMSE_RANGE = 0.1     # Å — accept "median suffices" if range < this
MAX_ACCEPTABLE_RMSE  = 0.5     # Å — accept "median suffices" if RMSE@median < this

# --- PDF metric (cheap; for trend visualization) ---
PDF_R_MAX   = 8.0
PDF_R_STEP  = 0.05
COMPUTE_PDF = True

# ─────────────────────────────────────────────────────────────────────────────

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(GPU_ID))
# Match the production allocator config — expandable_segments handles
# variable-shape graphs (different cells across CIFs in this sweep) and
# garbage_collection_threshold preempts the close-to-OOM fragment buildup.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)
_n = str(NUM_THREADS)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _n)

import csv
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict

import numpy as np
import torch
torch.set_num_threads(NUM_THREADS)
from torch_geometric.data import Batch

from ase.io import read as ase_read, write as ase_write
from ase.optimize import FIRE

import tricor as tc
from tricor.macerelax.flow_utils import (
    periodic_radius_graph_cell_list,
    periodic_radius_graph_chunked,
)
from tricor.macerelax.model import RelaxMLModel
from tricor.macerelax.data import (
    ShellTargetData,
    _weight_vector_from_row,
    _min_image_displacement,
)
from tricor.shells import CoordinationShellTarget
from tricor.macerelax.shell_target import extract_shell_target_arrays
from mace.calculators import mace_mp

_GEN_DIR = Path(__file__).resolve().parent.parent / "generation"
if str(_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(_GEN_DIR))
from wall_calculator import MinDistanceWallCalculator, per_pair_min_from_atoms


# ─────────────────────────────────────────────────────────────────────────────
# Tricor preset overrides (same as generate_with_student.py)
# ─────────────────────────────────────────────────────────────────────────────


def _build_local_preset() -> dict:
    d = dict(tc.Supercell.PRESETS[REGIME])
    d["displacement_sigma"] = 0.0
    d["num_steps"] = 0
    return d


LOCAL_PRESET = _build_local_preset()

_MP_ID_RE = re.compile(r"^(mp-\d+)_(.+)$")


# ─────────────────────────────────────────────────────────────────────────────
# Model + supercell helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_run_dir(log_dir: str, run_name: str,
                     run_timestamp: str | None) -> Path:
    exp_dir = Path(log_dir) / run_name
    if not exp_dir.is_dir():
        raise SystemExit(f"[abort] no training logs at {exp_dir}")
    if run_timestamp is not None:
        name = run_timestamp if run_timestamp.startswith("run_") \
                else f"run_{run_timestamp}"
        run = exp_dir / name
        if not run.is_dir():
            raise SystemExit(f"[abort] {run} not found")
        return run
    candidates = sorted(
        (d for d in exp_dir.iterdir() if d.name.startswith("run_") and d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit(f"[abort] no run_* dirs under {exp_dir}")
    return candidates[-1]


def _resolve_checkpoint(run_dir: Path, epoch: str) -> Path:
    if epoch not in ("last", "best") and Path(epoch).is_file():
        return Path(epoch).resolve()
    ckpt_dir = run_dir / "checkpoints"
    target = ckpt_dir / f"{epoch}.pt"
    if not target.is_file():
        alt = "last" if epoch == "best" else "best"
        alt_path = ckpt_dir / f"{alt}.pt"
        if alt_path.is_file():
            print(f"[warn] {target.name} not found, using {alt}.pt")
            return alt_path.resolve()
        raise SystemExit(f"[abort] no {epoch}.pt under {ckpt_dir}")
    return target.resolve()


def _build_model() -> RelaxMLModel:
    return RelaxMLModel(
        max_z=MAX_Z, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        num_convs=NUM_CONVS,
        weight_encoder_hidden=WEIGHT_ENCODER_HIDDEN,
        species_pair_dim=SPECIES_PAIR_DIM,
        shell_target_species_dim=SHELL_TARGET_SPECIES_DIM,
        shell_target_hidden=SHELL_TARGET_HIDDEN,
        shell_target_dropout=SHELL_TARGET_DROPOUT,
    )


def _load_student(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = _build_model()
    for p in model.processor.edge_norms[-1].parameters():
        p.requires_grad_(False)
    src = payload["ema"] if USE_EMA_WEIGHTS else payload["model"]
    src = {k.removeprefix("_orig_mod."): v for k, v in src.items()}
    model.load_state_dict(src, strict=False)
    print(f"[student] loaded {ckpt_path.name} epoch={int(payload.get('epoch', -1))}")
    return model.eval().to(device)


def parse_cif_name(cif_path: Path) -> tuple[str, str]:
    m = _MP_ID_RE.match(cif_path.stem)
    return (m.group(1), m.group(2)) if m else ("", cif_path.stem)


def build_supercell(cif_path: Path, cell_dims: tuple[float, float, float] | None = None):
    """Pack a tricor supercell.  cell_dims overrides CELL_DIMS for OOM retries."""
    if cell_dims is None:
        cell_dims = tuple(CELL_DIMS)
    ref = ase_read(str(cif_path), format="cif")
    shell = tc.CoordinationShellTarget.from_atoms(ref, phi_num_bins=90)
    from tricor import G3Distribution
    dist = G3Distribution(ref, label=str(cif_path.stem))
    dist.measure_g3(r_max=10.0, r_step=0.1, phi_num_bins=90, show_progress=False)
    cell = tc.Supercell(
        dist, cell_dim_angstroms=tuple(cell_dims),
        relative_density=RELATIVE_DENSITY,
        rng_seed=SEED, label=f"{cif_path.stem}_fmax_test_{cell_dims[0]:.0f}",
    )
    summary = cell.generate(shell, **LOCAL_PRESET,
                             refine_orientations=False, show_progress=False)
    cell.bond_relax(shell, n_iter=BOND_RELAX_N_ITER,
                    max_step=BOND_RELAX_MAX_STEP)
    return cell, shell, ref, summary


# OOM retry — match production: shrink cell by 5% per attempt, min 30 Å.
OOM_SHRINK_FACTOR = 0.95
OOM_MIN_CELL      = 30.0


def build_supercell_with_oom_retry(cif_path: Path):
    """Try CELL_DIMS, shrink and retry on CUDA OOM during MACE warmup."""
    dims = list(CELL_DIMS)
    while dims[0] >= OOM_MIN_CELL:
        try:
            cell, shell, ref, summary = build_supercell(cif_path, tuple(dims))
            return cell, shell, ref, summary, tuple(dims)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            new_dims = [d * OOM_SHRINK_FACTOR for d in dims]
            print(f"  [oom] supercell construction OOM at {dims[0]:.1f} Å → "
                  f"shrinking to {new_dims[0]:.1f} Å")
            dims = new_dims
    raise RuntimeError(
        f"[abort] cell construction OOM below MIN_CELL={OOM_MIN_CELL} Å"
    )


def build_weight_vector(cell, summary: dict, fmax_initial: float) -> np.ndarray:
    grain_size = float(summary.get("grain_size") or 0.0)
    num_grains = int(summary.get("n_grains") or 0)
    crystalline_fraction = float(summary.get("crystalline_fraction") or 0.0)
    r_min_per_pair = per_pair_min_from_atoms(cell.atoms, margin=WALL_MARGIN)
    wall_global_min = (float(min(r_min_per_pair.values()))
                        if r_min_per_pair else 0.0)
    fake_row = {
        "grain_size":           str(grain_size),
        "num_grains":           str(num_grains),
        "crystalline_fraction": str(crystalline_fraction),
        "rel_density":          str(RELATIVE_DENSITY),
        "wall_global_min":      str(wall_global_min),
        "fmax_initial":         str(fmax_initial),
    }
    return _weight_vector_from_row(fake_row)


# ─────────────────────────────────────────────────────────────────────────────
# Student inference
# ─────────────────────────────────────────────────────────────────────────────

def _periodic_graph(pos, cutoff, cell):
    try:
        return periodic_radius_graph_cell_list(pos, cutoff, cell)
    except ValueError:
        return periodic_radius_graph_chunked(pos, cutoff, cell=cell)


def _build_data(positions, cell, z, weight_vector, shell, cutoff) -> Batch:
    edge_index, edge_vec = _periodic_graph(positions, cutoff, cell)
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    edge_attr = torch.hstack([edge_vec, edge_len])
    data = ShellTargetData(
        z=z, pos=positions, edge_index=edge_index, edge_attr=edge_attr,
        w=weight_vector.unsqueeze(0),
        shell_pair_species=shell["pair_species"],
        shell_pair_features=shell["pair_features"],
        shell_pair_batch=torch.zeros(shell["pair_species"].shape[0],
                                     dtype=torch.long, device=positions.device),
        shell_trip_species=shell["trip_species"],
        shell_trip_features=shell["trip_features"],
        shell_trip_batch=torch.zeros(shell["trip_species"].shape[0],
                                     dtype=torch.long, device=positions.device),
    )
    return Batch.from_data_list([data])


def _wrap_positions(pos, cell):
    inv_cell = torch.linalg.inv(cell)
    frac = pos @ inv_cell.T
    frac = frac - torch.floor(frac)
    return frac @ cell


@torch.no_grad()
def run_student(model, initial_positions, cell, species,
                weight_vector, shell_arrays, device,
                *, max_iter=MAX_ITER, tol=CONVERGENCE_TOL_ANG, cutoff=CUTOFF):
    pos = torch.tensor(initial_positions, dtype=torch.float32, device=device)
    cell_t = torch.tensor(cell, dtype=torch.float32, device=device)
    w_t = torch.tensor(weight_vector, dtype=torch.float32, device=device)
    z = torch.tensor(species, dtype=torch.long, device=device)
    shell = {
        "pair_species":  torch.tensor(shell_arrays["shell_pair_species"],
                                      dtype=torch.long, device=device),
        "pair_features": torch.tensor(shell_arrays["shell_pair_features"],
                                      dtype=torch.float32, device=device),
        "trip_species":  torch.tensor(shell_arrays["shell_triplet_species"],
                                      dtype=torch.long, device=device),
        "trip_features": torch.tensor(shell_arrays["shell_triplet_features"],
                                      dtype=torch.float32, device=device),
    }
    n_iter = 0
    total_disp = 0.0
    for it in range(max_iter):
        batch = _build_data(pos, cell_t, z, w_t, shell, cutoff)
        delta = model(
            batch.z, batch.edge_index, batch.edge_attr,
            batch.w, batch.batch,
            batch.shell_pair_species, batch.shell_pair_features,
            batch.shell_pair_batch,
            batch.shell_trip_species, batch.shell_trip_features,
            batch.shell_trip_batch,
        )
        pos_new = _wrap_positions(pos + delta, cell_t)
        max_step = (pos_new - pos).norm(dim=-1).max().item()
        total_disp += float(delta.norm(dim=-1).mean().item())
        pos = pos_new
        n_iter += 1
        if max_step < tol:
            break
    return pos.cpu().numpy(), n_iter, total_disp


# ─────────────────────────────────────────────────────────────────────────────
# MACE ground truth
# ─────────────────────────────────────────────────────────────────────────────

def measure_fmax_initial_mace(atoms, calc) -> float:
    """One MACE forward to measure the actual fmax_initial."""
    atoms.calc = calc
    forces = atoms.get_forces()
    return float(np.abs(forces).max())


def run_mace_fire(atoms, calc):
    """Same MACE+wall FIRE protocol as production generation, shorter."""
    r_min = per_pair_min_from_atoms(atoms, margin=WALL_MARGIN)
    atoms.calc = MinDistanceWallCalculator(
        base_calc=calc, r_min_per_pair=r_min,
        k=WALL_K, exponent=WALL_EXPONENT,
    )
    e0 = float(atoms.get_potential_energy())
    f0 = atoms.get_forces()
    fmax_init = float(np.abs(f0).max())
    best = {"E": e0, "pos": atoms.positions.copy()}

    opt = FIRE(atoms, maxstep=OPT_MAXSTEP, logfile=None)

    def cb():
        try:
            e = float(atoms.get_potential_energy())
        except Exception:
            return
        if e < best["E"]:
            best["E"] = e
            best["pos"] = atoms.positions.copy()

    opt.attach(cb, interval=1)
    opt.run(fmax=FMAX_TARGET, steps=N_STEPS_MACE)
    return best["pos"].astype(np.float32), fmax_init


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def rmse_to_truth(pred: np.ndarray, truth: np.ndarray, cell: np.ndarray) -> float:
    pt = torch.tensor(pred, dtype=torch.float64)
    tt = torch.tensor(truth, dtype=torch.float64)
    ct = torch.tensor(cell, dtype=torch.float64)
    disp = _min_image_displacement(pt, tt, ct)
    return float(disp.pow(2).sum(dim=-1).mean().sqrt().item())


def pdf_simple(positions, species, cell, r_max=PDF_R_MAX, r_step=PDF_R_STEP):
    """Cheap per-system g(r) summed over all pairs.  Just for comparing
    shape — not species-resolved."""
    pos = np.asarray(positions, dtype=np.float64)
    cell_arr = np.asarray(cell, dtype=np.float64)
    inv_cell = np.linalg.inv(cell_arr)
    n = len(pos)
    nbin = int(round(r_max / r_step))
    hist = np.zeros(nbin, dtype=np.float64)
    for i in range(n):
        delta = pos[i:i+1] - pos
        frac = delta @ inv_cell
        frac -= np.round(frac)
        delta = frac @ cell_arr
        d = np.linalg.norm(delta, axis=-1)
        d = d[(d > 0) & (d < r_max)]
        idx = (d / r_step).astype(int)
        np.add.at(hist, idx, 1.0)
    r = (np.arange(nbin) + 0.5) * r_step
    V = abs(np.linalg.det(cell_arr))
    shell = 4 * np.pi * r ** 2 * r_step
    rho_avg = n / V
    g = hist / (n * rho_avg * shell + 1e-30)
    return r, g


# ─────────────────────────────────────────────────────────────────────────────
# Per-system orchestration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FmaxResult:
    system:        str
    cif_filename:  str
    fmax_setting:  float
    fmax_measured: float    # true MACE-measured value (one per system)
    n_iter:        int
    cumulative_disp: float
    rmse_vs_mace:  float
    pdf_mse:       float


def run_one(student_model, mace_calc, cif_path: Path, device) -> list[FmaxResult]:
    mp_id, compound = parse_cif_name(cif_path)
    sys_label = f"{compound}_{mp_id}"
    out_dir = OUTPUT_ROOT / sys_label
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {sys_label} ===")

    # ── Build the structure with OOM retry (single seed, single regime) ──
    cell, shell, ref_atoms, summary, dims_used = build_supercell_with_oom_retry(
        cif_path,
    )
    atoms = cell.atoms.copy()
    initial_pos = atoms.positions.copy()
    species_numbers = atoms.numbers.copy()
    cell_arr = atoms.cell.array.copy()
    if dims_used != tuple(CELL_DIMS):
        print(f"  [oom-retry] settled at cell={dims_used[0]:.1f} Å "
              f"(was {CELL_DIMS[0]:.1f} Å)")

    # ── Measure true fmax_initial (one MACE forward) ─────────────────────
    try:
        fmax_meas = measure_fmax_initial_mace(atoms.copy(), mace_calc)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"MACE OOM even at cell={dims_used[0]:.1f} Å — system too dense, skip"
        )
    print(f"  [mace] measured fmax_initial = {fmax_meas:.3f} eV/Å")

    # ── Run MACE+wall FIRE for ground truth ──────────────────────────────
    print(f"  [mace] running FIRE for {N_STEPS_MACE} steps...")
    truth_pos, _ = run_mace_fire(atoms.copy(), mace_calc)
    ase_write(str(out_dir / "mace_truth.xyz"),
              _atoms(species_numbers, truth_pos, cell_arr,
                     frame_label="mace_truth"),
              format="extxyz")

    # ── Shell target arrays (shared across all student runs) ────────────
    shell_arrays = extract_shell_target_arrays(
        CoordinationShellTarget.from_atoms(ref_atoms)
    )

    # ── Reference PDF from MACE truth ────────────────────────────────────
    if COMPUTE_PDF:
        r_grid, g_truth = pdf_simple(truth_pos, species_numbers, cell_arr)
    else:
        r_grid = g_truth = None

    # ── Sweep fmax_initial through the student ──────────────────────────
    results: list[FmaxResult] = []
    for fmax_set in FMAX_SWEEP:
        wv = build_weight_vector(cell, summary, fmax_set)
        t0 = time.time()
        student_pos, n_iter, cum_disp = run_student(
            student_model, initial_pos, cell_arr, species_numbers,
            wv, shell_arrays, device,
        )
        dt = time.time() - t0
        rmse = rmse_to_truth(student_pos, truth_pos, cell_arr)
        if COMPUTE_PDF:
            _, g_student = pdf_simple(student_pos, species_numbers, cell_arr)
            pdf_mse = float(np.mean((g_student - g_truth) ** 2))
        else:
            pdf_mse = float("nan")

        ase_write(str(out_dir / f"student_fmax_{fmax_set:.1f}.xyz"),
                  _atoms(species_numbers, student_pos, cell_arr,
                         frame_label=f"student_fmax_{fmax_set:.1f}"),
                  format="extxyz")

        r = FmaxResult(
            system=sys_label, cif_filename=cif_path.name,
            fmax_setting=float(fmax_set), fmax_measured=fmax_meas,
            n_iter=n_iter, cumulative_disp=cum_disp,
            rmse_vs_mace=rmse, pdf_mse=pdf_mse,
        )
        results.append(r)
        print(f"    fmax={fmax_set:5.1f}  iters={n_iter:2d}  "
              f"RMSE={rmse:.4f} Å  PDF MSE={pdf_mse:.3e}  dt={dt:.1f}s")

    # ── Per-system CSV + plot ────────────────────────────────────────────
    with open(out_dir / "sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    _plot_sensitivity(results, out_dir / "sensitivity.png", sys_label, fmax_meas)
    return results


def _atoms(species, positions, cell, frame_label):
    from ase import Atoms
    a = Atoms(numbers=species, positions=np.asarray(positions, np.float64),
              cell=np.asarray(cell, np.float64), pbc=True)
    a.info["frame_label"] = frame_label
    return a


def _plot_sensitivity(results: list[FmaxResult], out_path: Path,
                       title: str, fmax_meas: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fmax = np.array([r.fmax_setting for r in results])
    rmse = np.array([r.rmse_vs_mace for r in results])
    pdf  = np.array([r.pdf_mse for r in results])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fmax, rmse, "o-", color="C0")
    axes[0].axvline(fmax_meas, color="C3", ls="--",
                    label=f"true MACE-measured = {fmax_meas:.2f}")
    axes[0].axvline(MEDIAN_FMAX_VALUE, color="0.4", ls=":",
                    label=f"placeholder = {MEDIAN_FMAX_VALUE:.1f}")
    axes[0].set_xlabel("fmax_initial setting (eV/Å)")
    axes[0].set_ylabel("RMSE vs MACE truth (Å)")
    axes[0].set_title("Student RMSE sensitivity")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(fmax, pdf, "o-", color="C0")
    axes[1].axvline(fmax_meas, color="C3", ls="--")
    axes[1].axvline(MEDIAN_FMAX_VALUE, color="0.4", ls=":")
    axes[1].set_xlabel("fmax_initial setting (eV/Å)")
    axes[1].set_ylabel("PDF MSE vs MACE truth")
    axes[1].set_title("Student PDF sensitivity")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3, which="both")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Resolve student checkpoint.
    run_dir = _resolve_run_dir(MODEL_LOG_DIR, MODEL_RUN_NAME, MODEL_RUN_TIMESTAMP)
    ckpt_path = _resolve_checkpoint(run_dir, MODEL_EPOCH)
    student = _load_student(ckpt_path, device)

    # Load MACE.
    print(f"[mace] loading {MACE_MODEL} (cold start ~30s)")
    mace_calc = mace_mp(model=MACE_MODEL, device=MACE_DEVICE,
                         default_dtype=MACE_DEFAULT_DTYPE)
    print("[mace] ready")

    all_results: list[FmaxResult] = []
    missing: list[str] = []
    for name in TEST_CIFS:
        cif_path = CIF_DIR / name
        if not cif_path.is_file():
            missing.append(name)
            print(f"[skip] {name} not found in {CIF_DIR}")
            continue
        try:
            rs = run_one(student, mace_calc, cif_path, device)
            all_results.extend(rs)
        except Exception as exc:
            traceback.print_exc(limit=3, file=sys.stdout)
            print(f"[fail] {name}: {type(exc).__name__}: {exc}")

    # ── Aggregate ────────────────────────────────────────────────────────
    if not all_results:
        print("\n[abort] no results produced")
        return
    summary_csv = OUTPUT_ROOT / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        w.writeheader()
        for r in all_results:
            w.writerow(asdict(r))

    # ── Verdict per system ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VERDICT per system:")
    print(f"  Tolerance: RMSE range < {TOLERANCE_RMSE_RANGE} Å, RMSE@median < {MAX_ACCEPTABLE_RMSE} Å\n")
    by_sys: dict[str, list[FmaxResult]] = {}
    for r in all_results:
        by_sys.setdefault(r.system, []).append(r)
    all_pass = True
    for sys_label, rs in by_sys.items():
        rmses = np.array([r.rmse_vs_mace for r in rs])
        rmse_range = float(rmses.max() - rmses.min())
        median_r = next((r for r in rs
                          if abs(r.fmax_setting - MEDIAN_FMAX_VALUE) < 1e-6), None)
        rmse_at_median = median_r.rmse_vs_mace if median_r else float("nan")
        verdict = (rmse_range < TOLERANCE_RMSE_RANGE
                    and rmse_at_median < MAX_ACCEPTABLE_RMSE)
        all_pass = all_pass and verdict
        flag = "✓ pass" if verdict else "✗ fail"
        print(f"  {sys_label:24s}  rmse_range={rmse_range:.3f} Å  "
              f"rmse@median={rmse_at_median:.3f} Å  {flag}")
    print()
    if all_pass:
        print("  → OVERALL: median fmax placeholder SUFFICES")
        print("    Safe to proceed with generate_with_student.py using DEFAULT_FMAX_INITIAL")
    else:
        print("  → OVERALL: median fmax placeholder is NOT enough on at least one system")
        print("    Recommend retraining without fmax_initial in conditioning")
    if missing:
        print(f"\n  Missing CIFs (not in {CIF_DIR}):")
        for n in missing:
            print(f"    {n}")
    print("=" * 60)


if __name__ == "__main__":
    main()
