"""Scale-invariance check using the CURRENT generation pipeline at both
small and large cell sizes.

The earlier version of this script loaded ``initial_positions`` /
``best_positions`` from legacy training NPZ — those were produced with
``shell_relax`` + thermal jitter (a strategy we no longer use), so they
disagreed wildly with our current ``bond_relax``-only initial structures
at large cell.  This version fixes that by generating BOTH small and
large initials with the same pipeline:

    build_supercell( regime preset, num_steps=0, displacement_sigma=0 )
        → bond_relax( n_iter=20, max_step=0.1 )         ← "initial"
        → MACE+wall+FIRE( 60 steps )                    ← "small MACE" (small only)
        → run_iterative_inference (15 iters)            ← "student" (small & large)

Same regime preset, same bond_relax, just different cell dimensions.  If
the student generalises cleanly to large cells, all five curves overlay
in g(r) and ADF — especially the two initials (gray dashed / dotted)
which should be statistically identical modulo finite-size noise.

Outputs:
    scratch/scale_invariance.png          — 6 × 6 panel grid
    scratch/scale_invariance_data.npz     — cached g(r) + ADF arrays

Run (requires MACE — use the ``mace`` conda env, not ``tricor``):
    /home/ehrdt/miniforge3/envs/mace/bin/python scratch/scale_invariance_check.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts" / "macerelax"))
sys.path.insert(0, str(_REPO / "scripts" / "macerelax" / "generation"))


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CIF_DIR = Path("/home/ehrdt/cifs_mp_exp_le100meV")
CIFS = [
    ("Si",   "mp-149_Si.cif"),
    ("TiO2", "mp-1439_TiO2.cif"),
    ("Fe2N", "mp-21476_Fe2N.cif"),
]

REGIMES = ["amorphous", "SRO", "MRO", "LRO", "nanocrystalline", "crystalline_30"]

DENSITY_BY_REGIME = {
    "amorphous":       0.92,
    "SRO":             0.92,
    "MRO":             0.88,
    "LRO":             0.92,
    "nanocrystalline": 0.96,
    "crystalline_30":  0.98,
}

SMALL_CELL_DIMS = (50.0, 50.0, 50.0)
LARGE_CELL_DIMS = (100.0, 100.0, 400.0)
# Fe2N at 50³ has ~12 k atoms × MACE-MPA — typically fits but tight on
# buffle.  Drop to 40³ as needed via OOM retry below.
SMALL_CELL_OOM_FLOOR = 35.0
RNG_SEED = 2_000_000

BOND_RELAX_N_ITER   = 20
BOND_RELAX_MAX_STEP = 0.1

DEFAULT_FMAX_INITIAL = 5.0
WALL_MARGIN          = 0.0

# --- MACE ground truth ---
MACE_MODEL          = "medium-mpa-0"
MACE_DEVICE         = "cuda:0"
MACE_DEFAULT_DTYPE  = "float32"
N_STEPS_MACE        = 60
OPT_MAXSTEP         = 0.3
FMAX_TARGET         = 0.05
WALL_K              = 1000.0
WALL_EXPONENT       = 4

# --- Analysis ---
PDF_R_MAX        = 8.0
PDF_NBINS        = 200
ADF_R_CUT        = 3.5
ADF_NBINS        = 180
ADF_MAX_CENTERS  = 5_000

CACHE_NPZ  = _REPO / "scratch" / "scale_invariance_data.npz"
OUTPUT_PNG = _REPO / "scratch" / "scale_invariance.png"
TIMINGS_CSV = _REPO / "scratch" / "scale_invariance_timings.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Analysis primitives
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(pos, box):
    return pos - np.floor(pos / box) * box


def compute_pdf(positions, box_diag, r_max=PDF_R_MAX, n_bins=PDF_NBINS):
    from scipy.spatial import cKDTree
    if len(positions) < 2:
        return np.linspace(0, r_max, n_bins), np.zeros(n_bins)
    pos = _wrap(np.asarray(positions, dtype=np.float64),
                np.asarray(box_diag, dtype=np.float64))
    tree = cKDTree(pos, boxsize=box_diag)
    pairs = tree.query_pairs(r_max, output_type="ndarray")
    if len(pairs) == 0:
        return np.linspace(0, r_max, n_bins), np.zeros(n_bins)
    delta = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    delta -= np.round(delta / box_diag) * box_diag
    dist = np.linalg.norm(delta, axis=1)
    edges = np.linspace(0.0, r_max, n_bins + 1)
    hist, _ = np.histogram(dist, bins=edges)
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    N = len(pos)
    V = float(np.prod(box_diag))
    rho = N / max(V, 1e-12)
    norm = 0.5 * N * 4.0 * np.pi * r ** 2 * dr * rho
    g = np.where(norm > 0, hist.astype(float) / norm, 0.0)
    return r, g


def compute_adf(positions, box_diag, r_cut=ADF_R_CUT, n_bins=ADF_NBINS,
                max_centers=ADF_MAX_CENTERS, rng_seed=0):
    from scipy.spatial import cKDTree
    pos = _wrap(np.asarray(positions, dtype=np.float64),
                np.asarray(box_diag, dtype=np.float64))
    N = len(pos)
    if N < 3:
        return np.linspace(0, 180, n_bins), np.zeros(n_bins)
    tree = cKDTree(pos, boxsize=box_diag)
    rng = np.random.default_rng(int(rng_seed))
    if max_centers is not None and max_centers < N:
        sample = rng.choice(N, size=max_centers, replace=False)
    else:
        sample = np.arange(N)
    bin_edges = np.linspace(0.0, 180.0, n_bins + 1)
    total_hist = np.zeros(n_bins, dtype=np.int64)
    for i in sample:
        nbr_idx = tree.query_ball_point(pos[i], r_cut)
        nbrs = np.array([j for j in nbr_idx if j != int(i)], dtype=np.intp)
        if len(nbrs) < 2:
            continue
        delta = pos[nbrs] - pos[i]
        delta -= np.round(delta / box_diag) * box_diag
        d = np.linalg.norm(delta, axis=1)
        d = np.where(d > 1e-12, d, 1e-12)
        unit = delta / d[:, None]
        cos_mat = unit @ unit.T
        triu = np.triu_indices(len(nbrs), k=1)
        cos_vals = np.clip(cos_mat[triu], -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_vals))
        h, _ = np.histogram(angles, bins=bin_edges)
        total_hist += h
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    total = total_hist.sum()
    return centres, (total_hist.astype(float) / total) if total > 0 else \
        np.zeros(n_bins)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline calls
# ─────────────────────────────────────────────────────────────────────────────

def _build_local_presets(tc):
    base = {}
    for name in REGIMES:
        if name == "crystalline_30":
            base[name] = dict(
                num_steps=0, grain_size=30.0, displacement_sigma=0.0,
                bond_weight=3.0, angle_weight=1.5,
            )
        else:
            d = dict(tc.Supercell.PRESETS[name])
            d["displacement_sigma"] = 0.0
            d["num_steps"]          = 0
            base[name] = d
    return base


def build_initial(tc, ase_read, G, cif_path, regime, rho, cell_dims, device):
    """Run the current production pipeline: build_supercell + bond_relax."""
    from tricor import CoordinationShellTarget, G3Distribution
    ref = ase_read(str(cif_path), format="cif")
    shell = CoordinationShellTarget.from_atoms(ref, phi_num_bins=90)
    dist = G3Distribution(ref, label=cif_path.stem)
    dist.measure_g3(r_max=10.0, r_step=0.1, phi_num_bins=90,
                    show_progress=False)
    cell = tc.Supercell(
        dist, cell_dim_angstroms=tuple(cell_dims),
        relative_density=rho, rng_seed=RNG_SEED,
        label=f"{cif_path.stem}_{regime}_{RNG_SEED}",
    )
    preset = dict(LOCAL_PRESETS[regime])
    cell.generate(shell, **preset, refine_orientations=False,
                  show_progress=False)
    cell.bond_relax(shell, n_iter=BOND_RELAX_N_ITER,
                    max_step=BOND_RELAX_MAX_STEP, device=device)
    return cell, shell, ref


def run_student(G, model, device, positions, cell_arr, species,
                regime, rho, ref, summary):
    shell_arrays = G.build_shell_target(ref)
    # build_weight_vector needs a cell-like with .atoms; replicate the
    # minimum surface generate_with_student uses.
    class _Cell:
        def __init__(self, atoms): self.atoms = atoms
    from ase.atoms import Atoms
    atoms = Atoms(numbers=species, positions=positions, cell=cell_arr, pbc=True)
    dummy = _Cell(atoms)
    wv = G.build_weight_vector(dummy, summary, regime, rho)
    final, _, _ = G.run_iterative_inference(
        model, positions.copy(), cell_arr, species,
        wv, shell_arrays, device,
    )
    return final


def run_mace(atoms, mace_calc):
    """Same MACE+wall+FIRE protocol as production (per pilot/test_fmax_vs_mace)."""
    from wall_calculator import MinDistanceWallCalculator, per_pair_min_from_atoms
    from ase.optimize import FIRE
    r_min = per_pair_min_from_atoms(atoms, margin=WALL_MARGIN)
    atoms.calc = MinDistanceWallCalculator(
        base_calc=mace_calc, r_min_per_pair=r_min,
        k=WALL_K, exponent=WALL_EXPONENT,
    )
    best = {
        "E": float(atoms.get_potential_energy()),
        "pos": atoms.positions.copy(),
    }

    def cb():
        try:
            e = float(atoms.get_potential_energy())
        except Exception:
            return
        if e < best["E"]:
            best["E"] = e
            best["pos"] = atoms.positions.copy()

    opt = FIRE(atoms, maxstep=OPT_MAXSTEP, logfile=None)
    opt.attach(cb, interval=1)
    opt.run(fmax=FMAX_TARGET, steps=N_STEPS_MACE)
    return best["pos"].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-(system, regime) driver
# ─────────────────────────────────────────────────────────────────────────────

def process_cell(tc, G, ase_read, model, device, mace_calc,
                 cif_path, regime, rho, cell_dims, want_mace, timings):
    """Return ({curve: (r, g, theta, adf)}, timing_rows) for one cell.

    ``timings`` is a mutable list accumulating per-step rows
    ``{"system","regime","cell","n_atoms","step","seconds"}`` so the
    end-of-run summary can compare MACE vs student wall-clock at the
    same scale.
    """
    cell_label = f"{int(cell_dims[0])}x{int(cell_dims[1])}x{int(cell_dims[2])}"
    t0 = time.perf_counter()
    cell, shell, ref = build_initial(tc, ase_read, G,
                                     cif_path, regime, rho, cell_dims, device)
    n_atoms = len(cell.atoms)
    box_diag = np.diag(cell.atoms.cell.array)
    init_pos = cell.atoms.positions.copy()
    species = cell.atoms.numbers.copy()
    cell_arr = cell.atoms.cell.array.copy()
    summary = {
        "grain_size":            float(cell.atoms.info.get("grain_size", 0.0)),
        "n_grains":              int(cell.atoms.info.get("n_grains", 0)),
        "crystalline_fraction":  float(
            cell.atoms.info.get("crystalline_fraction", 0.0)),
    }
    t_init = time.perf_counter() - t0
    timings.append({
        "system": cif_path.stem, "regime": regime, "cell": cell_label,
        "n_atoms": n_atoms, "step": "init", "seconds": t_init,
    })
    print(f"      [init] {cell_dims} n={n_atoms} ({t_init:.1f}s)",
          flush=True)

    # MACE on small cell only.
    mace_pos = None
    if want_mace and mace_calc is not None:
        from ase.atoms import Atoms
        small_atoms = Atoms(numbers=species, positions=init_pos.copy(),
                            cell=cell_arr, pbc=True)
        t_m = time.perf_counter()
        mace_pos = run_mace(small_atoms, mace_calc)
        t_mace = time.perf_counter() - t_m
        timings.append({
            "system": cif_path.stem, "regime": regime, "cell": cell_label,
            "n_atoms": n_atoms, "step": "mace_fire", "seconds": t_mace,
        })
        print(f"      [mace] FIRE {N_STEPS_MACE} steps ({t_mace:.1f}s)",
              flush=True)

    # Student.
    t_s = time.perf_counter()
    student_pos = run_student(G, model, device, init_pos.copy(),
                               cell_arr, species, regime, rho, ref, summary)
    t_student = time.perf_counter() - t_s
    timings.append({
        "system": cif_path.stem, "regime": regime, "cell": cell_label,
        "n_atoms": n_atoms, "step": "student", "seconds": t_student,
    })
    print(f"      [student] {t_student:.1f}s", flush=True)

    curves = {}
    r, g = compute_pdf(init_pos, box_diag)
    th, ad = compute_adf(init_pos, box_diag)
    curves["init"] = dict(r=r, g=g, th=th, ad=ad)
    if mace_pos is not None:
        r, g = compute_pdf(mace_pos, box_diag)
        th, ad = compute_adf(mace_pos, box_diag)
        curves["mace"] = dict(r=r, g=g, th=th, ad=ad)
    r, g = compute_pdf(student_pos, box_diag)
    th, ad = compute_adf(student_pos, box_diag)
    curves["student"] = dict(r=r, g=g, th=th, ad=ad)
    return curves


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

CURVE_STYLE = {
    "small_init":    dict(color="#888888", lw=1.0, ls="--", label="small init"),
    "small_mace":    dict(color="#000000", lw=1.6, ls="-",  label="small MACE"),
    "small_student": dict(color="#1f77b4", lw=1.2, ls="-",  label="small student"),
    "large_init":    dict(color="#aaaaaa", lw=1.0, ls=":",  label="large init"),
    "large_student": dict(color="#d62728", lw=1.2, ls="-",  label="large student"),
}


def make_grid(all_curves, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_rows = len(REGIMES)
    n_systems = len(CIFS)
    n_cols = n_systems * 2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 2.0 * n_rows),
        squeeze=False, sharex="col",
    )
    for r_idx, regime in enumerate(REGIMES):
        for s_idx, (label, _) in enumerate(CIFS):
            ax_g = axes[r_idx, 2 * s_idx]
            ax_a = axes[r_idx, 2 * s_idx + 1]
            for name, style in CURVE_STYLE.items():
                d = all_curves.get(label, {}).get(regime, {}).get(name)
                if d is None:
                    continue
                ax_g.plot(d["r"], d["g"], **style)
                ax_a.plot(d["th"], d["ad"], **style)
            ax_g.set_xlim(0, PDF_R_MAX)
            ax_a.set_xlim(0, 180)
            if r_idx == 0:
                ax_g.set_title(f"{label} — g(r)")
                ax_a.set_title(f"{label} — ADF")
            if s_idx == 0:
                ax_g.set_ylabel(regime, rotation=0, ha="right", va="center",
                                fontsize=10)
            if r_idx == n_rows - 1:
                ax_g.set_xlabel("r (Å)")
                ax_a.set_xlabel("θ (deg)")
            ax_g.grid(alpha=0.3, lw=0.4)
            ax_a.grid(alpha=0.3, lw=0.4)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=s["color"], lw=s["lw"], ls=s["ls"])
               for s in CURVE_STYLE.values()]
    labels = [s["label"] for s in CURVE_STYLE.values()]
    fig.legend(handles, labels, loc="lower center",
               ncol=len(CURVE_STYLE), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(str(out_png), dpi=140, bbox_inches="tight")
    print(f"[plot] wrote {out_png}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────

def save_cache(all_curves, path):
    flat = {}
    for sys_label, regimes in all_curves.items():
        for regime, curves in regimes.items():
            for name, d in curves.items():
                stem = f"{sys_label}|{regime}|{name}"
                for k, v in d.items():
                    flat[f"{stem}|{k}"] = v
    np.savez(str(path), **flat)
    print(f"[cache] wrote {path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_PRESETS = {}


def main() -> None:
    global LOCAL_PRESETS
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,garbage_collection_threshold:0.8",
    )
    import torch
    import tricor as tc
    from ase.io import read as ase_read
    from mace.calculators import mace_mp
    import generate_with_student as G

    LOCAL_PRESETS = _build_local_presets(tc)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[scale] device={device}", flush=True)

    # Load student.
    model = G._load_model(
        G._resolve_checkpoint(
            G._resolve_run_dir(G.MODEL_LOG_DIR, G.MODEL_RUN_NAME,
                               G.MODEL_RUN_TIMESTAMP),
            G.MODEL_EPOCH,
        ),
        device,
    )
    G.patch_model_edge_chunking(model, G.EDGE_CHUNK_SIZE)

    # Load MACE.
    print(f"[scale] loading MACE: {MACE_MODEL}", flush=True)
    mace_calc = mace_mp(
        model=MACE_MODEL, device=MACE_DEVICE,
        default_dtype=MACE_DEFAULT_DTYPE,
    )

    all_curves = {}
    timings: list[dict] = []
    for sys_label, cif_name in CIFS:
        all_curves[sys_label] = {}
        cif_path = CIF_DIR / cif_name
        if not cif_path.is_file():
            print(f"  ! missing CIF: {cif_path}")
            continue
        for regime in REGIMES:
            print(f"  [{sys_label}/{regime}]", flush=True)
            rho = DENSITY_BY_REGIME[regime]

            # Small cell with MACE.
            try:
                small_curves = process_cell(
                    tc, G, ase_read, model, device, mace_calc,
                    cif_path, regime, rho, SMALL_CELL_DIMS, want_mace=True,
                    timings=timings,
                )
                regime_data = {
                    "small_init":    small_curves["init"],
                    "small_mace":    small_curves["mace"],
                    "small_student": small_curves["student"],
                }
            except torch.cuda.OutOfMemoryError as exc:
                print(f"    small OOM: {exc}", flush=True)
                torch.cuda.empty_cache()
                regime_data = {}
            except Exception as exc:
                print(f"    small failed: {type(exc).__name__}: {exc}",
                      flush=True)
                regime_data = {}

            # Large cell, no MACE.
            try:
                large_curves = process_cell(
                    tc, G, ase_read, model, device, mace_calc,
                    cif_path, regime, rho, LARGE_CELL_DIMS, want_mace=False,
                    timings=timings,
                )
                regime_data.update({
                    "large_init":    large_curves["init"],
                    "large_student": large_curves["student"],
                })
            except torch.cuda.OutOfMemoryError as exc:
                print(f"    large OOM: {exc}", flush=True)
                torch.cuda.empty_cache()
            except Exception as exc:
                print(f"    large failed: {type(exc).__name__}: {exc}",
                      flush=True)

            all_curves[sys_label][regime] = regime_data

            # Periodic cache so a mid-run crash still leaves usable data.
            save_cache(all_curves, CACHE_NPZ)
            save_timings(timings, TIMINGS_CSV)

    save_cache(all_curves, CACHE_NPZ)
    save_timings(timings, TIMINGS_CSV)
    print_timing_summary(timings)
    make_grid(all_curves, OUTPUT_PNG)


def save_timings(rows: list[dict], path) -> None:
    """Dump the per-step timing rows to CSV.  Re-written from scratch
    after every (system, regime) so a crash mid-run leaves the partial
    record on disk."""
    if not rows:
        return
    import csv
    fields = ["system", "regime", "cell", "n_atoms", "step", "seconds"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def print_timing_summary(rows: list[dict]) -> None:
    """End-of-run side-by-side table comparing MACE vs student per
    (system, cell-size).  Sums regime-level wall-clock so the totals
    are directly comparable.
    """
    if not rows:
        return
    # Index by (system, cell, step) → list of seconds across regimes.
    bucket: dict = {}
    for r in rows:
        key = (r["system"], r["cell"], r["step"])
        bucket.setdefault(key, []).append(float(r["seconds"]))
    # Pull out unique systems + cells.
    systems = sorted({r["system"] for r in rows})
    cells = sorted({r["cell"] for r in rows})
    print("\n────────  per-step wall-clock summary (sum over regimes)  "
          "────────")
    print(f"  {'system':<24}  {'cell':<16}  {'init':>9}  {'mace':>9}  "
          f"{'student':>9}  {'MACE/stud':>10}")
    print("  " + "-" * 86)
    for system in systems:
        for cell in cells:
            t_init = sum(bucket.get((system, cell, "init"), []))
            t_mace = sum(bucket.get((system, cell, "mace_fire"), []))
            t_stud = sum(bucket.get((system, cell, "student"), []))
            ratio = (t_mace / t_stud) if t_stud > 0 else float("nan")
            if t_init + t_mace + t_stud > 0:
                mace_s = f"{t_mace:>8.1f}s" if t_mace > 0 else "        —"
                ratio_s = (f"{ratio:>9.1f}×" if t_mace > 0
                           and t_stud > 0 else "         —")
                print(f"  {system:<24}  {cell:<16}  "
                      f"{t_init:>8.1f}s  {mace_s}  "
                      f"{t_stud:>8.1f}s  {ratio_s}")
    print()


if __name__ == "__main__":
    main()
