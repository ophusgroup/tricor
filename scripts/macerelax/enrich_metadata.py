"""Per-trajectory enrichment: cheap structural scalars + g(r) / ADF / g3 PNGs.

Walks ``OUTPUT_ROOT`` looking for trajectory files (XYZ or NPZ), computes a
small set of cheap per-structure scalars, and writes static PNGs for the
three correlation-function plots.  Results are appended to one CSV per rank
which ``build_dataset_table.py`` joins back into ``dataset_table.parquet``.

Scalars (per trajectory):

  * ``n_atoms``, ``cell_volume_A3``, ``density_at_per_A3``
  * ``min_pair_dist_A``
  * ``composition_json``       — {element_symbol: fraction}
  * ``mean_coord_by_species_json`` — {element_symbol: mean coord at COORD_CUTOFF_A}

PNGs (per trajectory — each gated by its ``RENDER_*`` CONFIG flag):

  * ``OUTPUT_ROOT/plots/{compound}_{mp_id}/{regime}/seed{seed}/gr.png``    (RENDER_GR)
  * ``OUTPUT_ROOT/plots/{compound}_{mp_id}/{regime}/seed{seed}/adf.png``   (RENDER_ADF, off by default)
  * ``OUTPUT_ROOT/plots/{compound}_{mp_id}/{regime}/seed{seed}/g3.png``    (RENDER_G3)

g(r) and g3 share one ``G3Distribution`` numba pass per structure.  g3 PNG
uses projection (ii) — per-triplet shell-integrated ``(r₂, φ)`` heatmaps
with r₁ fixed in the first-neighbor shell.  ADF (when enabled) is a 1D
marginalization of g3 over (r₁, r₂) within the same shell.

This script handles ONLY the cheap (no-MACE) side.  The MACE multi-crop
energy pass lives in a separate script (Task #15).

Idempotent — resume-safe via ``source_file`` set lookup against the existing
enrichment CSV.  Cheap to re-run.

Multi-rank: each rank writes ``enrichment.rank{N}.csv``.  Single-rank writes
``enrichment.csv``.  ``build_dataset_table.py`` globs ``enrichment*.csv``.

Usage:
    python scripts/macerelax/enrich_metadata.py
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# Where the trajectories live and where plots / CSV land.
OUTPUT_ROOT       = Path("/pscratch/sd/e/ehrdt/macerelax/generated_cnos_v1")
PLOTS_ROOT        = OUTPUT_ROOT / "plots"
ENRICHMENT_CSV    = OUTPUT_ROOT / "enrichment.csv"

# Which file types to enrich.  Production runs at the moment write XYZ only.
SCAN_XYZ          = True
SCAN_NPZ          = True

# Coordination-number cutoff.  Mean coord is averaged per species.
COORD_CUTOFF_A    = 3.5

# g(r) / g3 measurement parameters.
G3_R_MAX          = 10.0
G3_R_STEP         = 0.1
G3_PHI_NBINS      = 90
# Fraction of origin atoms sampled — controls per-structure cost.  At a
# 391 k-atom Fe2N slab, ~0.01 (~4 k origins) gives well-converged g(r) and
# g3 stats in ~5-10 s per structure with the numba backend.  Crank down
# if you have time to spare; crank up if cost is too high.
G3_SAMPLE_FRACTION = 0.01
G3_SAMPLE_RNG_SEED = 7
G3_BACKEND        = "auto"   # "auto" | "numba" | "python"

# First-neighbor shell used by the g3 projection (fix r₁ within shell, plot
# (r₂, φ) heatmap).  Also used by the ADF when RENDER_ADF=True.
SHELL_R_MIN_A     = 1.4
SHELL_R_MAX_A     = 3.2

# Which plots to render.  g(r) and g3 are the headline plots; ADF is a
# marginalization of g3 that turned out to be redundant in practice — the
# g3 (r₂, φ) heatmap already shows the angle structure.  Disabled by default.
RENDER_GR         = True
RENDER_ADF        = False
RENDER_G3         = True

# Plot quality.
PNG_DPI           = 96

# Skip a trajectory if its source_file is already in the existing CSV.
SKIP_IF_ALREADY_ENRICHED = True

# Resource caps for numpy / numba threading.
NUM_THREADS       = 4

# ─────────────────────────────────────────────────────────────────────────────
# Env caps must be set BEFORE numpy / numba / matplotlib import.
# ─────────────────────────────────────────────────────────────────────────────

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, str(NUM_THREADS))

# Force Agg backend for matplotlib so the script works headless (Perlmutter
# login + compute nodes have no display).
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import csv
import json
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, asdict, fields as dc_fields
from datetime import datetime, timezone

import numpy as np
from ase.atoms import Atoms
from ase.io import read as ase_read
from ase.data import chemical_symbols
from scipy.spatial import cKDTree

from tricor import G3Distribution


# ─────────────────────────────────────────────────────────────────────────────
# Multi-rank detection — same rules as generate_with_student.py
# ─────────────────────────────────────────────────────────────────────────────

def _detect_rank_from_env() -> tuple[int, int, int]:
    if "LOCAL_RANK" in os.environ:
        return (int(os.environ["LOCAL_RANK"]),
                int(os.environ.get("RANK", os.environ["LOCAL_RANK"])),
                int(os.environ.get("WORLD_SIZE", 1)))
    if "SLURM_LOCALID" in os.environ:
        return (int(os.environ["SLURM_LOCALID"]),
                int(os.environ.get("SLURM_PROCID", os.environ["SLURM_LOCALID"])),
                int(os.environ.get("SLURM_NTASKS", 1)))
    return (0, 0, 1)


_LOCAL_RANK, _GLOBAL_RANK, _WORLD_SIZE = _detect_rank_from_env()
_IS_MULTI_RANK = _WORLD_SIZE > 1


# ─────────────────────────────────────────────────────────────────────────────
# EnrichmentRow — the per-traj output schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnrichmentRow:
    # joining keys ─ match generate_with_student's GenResult on these
    run_id:                str = ""
    cif_filename:          str = ""
    compound:              str = ""
    mp_id:                 str = ""
    regime:                str = ""
    rng_seed:              int = -1
    # provenance
    source_file:           str = ""    # path relative to OUTPUT_ROOT
    schema_version:        int = 1
    # cheap scalars
    n_atoms:               int   = -1
    cell_volume_A3:        float = -1.0
    density_at_per_A3:     float = -1.0
    min_pair_dist_A:       float = -1.0
    composition_json:      str   = ""
    mean_coord_by_species_json: str = ""
    # plot paths (relative to OUTPUT_ROOT — empty string if plot not produced)
    gr_png:                str = ""
    adf_png:               str = ""
    g3_png:                str = ""
    # metadata
    g3_origin_sample_size:    int   = -1
    g3_origin_sample_fraction: float = -1.0
    enriched_at_utc:       str = ""
    error:                 str = ""


def _enrichment_fieldnames() -> list[str]:
    return [f.name for f in dc_fields(EnrichmentRow)]


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory file discovery + loading
# ─────────────────────────────────────────────────────────────────────────────

# Regimes are referenced when parsing filenames; the list mirrors
# generate_with_student.py's CONFIG.  Order matters for parsing because
# "crystalline_30" contains an underscore — longest match wins.
_REGIMES_KNOWN = (
    "amorphous", "SRO", "MRO", "LRO",
    "nanocrystalline", "crystalline_30",
)


def discover_traj_files(root: Path) -> list[Path]:
    """Find all final-frame trajectory files under ``root``."""
    files: list[Path] = []
    if SCAN_NPZ:
        files.extend(sorted(root.glob("*_generated/*.npz")))
    if SCAN_XYZ:
        files.extend(sorted(root.glob("*_generated/*.xyz")))
    return files


def parse_filename_components(path: Path) -> tuple[str, str, str, int]:
    """``{compound}_{mp_id}_{regime}_seed{seed}.{ext}`` → tuple.

    Returns ``("", "", "", -1)`` if the filename doesn't match the pattern.
    Longest-regime-match-first because "crystalline_30" contains an
    underscore that the regime separator would otherwise eat.
    """
    stem = path.stem
    if "_seed" not in stem:
        return ("", "", "", -1)
    base, seed_str = stem.rsplit("_seed", 1)
    try:
        seed = int(seed_str)
    except ValueError:
        return ("", "", "", -1)
    for reg in sorted(_REGIMES_KNOWN, key=len, reverse=True):
        marker = f"_{reg}"
        if base.endswith(marker):
            cmp_mp = base[: -len(marker)]
            if "_mp-" in cmp_mp:
                compound, mp_part = cmp_mp.split("_mp-", 1)
                return compound, "mp-" + mp_part, reg, seed
            return cmp_mp, "", reg, seed
    return base, "", "", seed


def _scalar_from_npz_field(nz, key: str, default=None):
    if key not in nz.files:
        return default
    arr = nz[key]
    try:
        return arr.item()
    except Exception:
        return arr.tolist() if hasattr(arr, "tolist") else default


def load_traj(path: Path) -> tuple[Atoms | None, dict]:
    """Return ``(final_atoms, info_dict)``.

    ``info_dict`` carries embedded provenance: ``run_id``, ``regime``,
    ``rng_seed``, ``cif_idx``.  We don't load the initial frame — the
    cheap-scalars pass doesn't compute anything that compares against it.
    """
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as nz:
            final = np.asarray(nz["final_positions"], dtype=np.float64)
            cell = np.asarray(nz["cell"], dtype=np.float64)
            species = np.asarray(nz["species_numbers"], dtype=int)
            atoms = Atoms(numbers=species, positions=final, cell=cell, pbc=True)
            info = {
                "run_id":  _scalar_from_npz_field(nz, "run_id", "") or "",
                "regime":  _scalar_from_npz_field(nz, "regime", "") or "",
                "rng_seed": int(_scalar_from_npz_field(nz, "rng_seed", -1) or -1),
                "compound": _scalar_from_npz_field(nz, "compound", "") or "",
                "mp_id":   _scalar_from_npz_field(nz, "mp_id", "") or "",
                "cif_idx": int(_scalar_from_npz_field(nz, "cif_idx", -1) or -1),
            }
            return atoms, info

    if suffix == ".xyz":
        # Read only the LAST frame — production XYZ has only that anyway,
        # and for multi-frame XYZ we don't need the intermediates here.
        final_atoms = ase_read(str(path), format="extxyz", index=-1)
        if final_atoms is None:
            return None, {}
        info_src = final_atoms.info
        info = {
            "run_id":  str(info_src.get("run_id", "")),
            "regime":  str(info_src.get("regime", "")),
            "rng_seed": int(info_src.get("rng_seed", -1) or -1),
            "compound": str(info_src.get("compound", "")),
            "mp_id":   str(info_src.get("mp_id", "")),
            "cif_idx": int(info_src.get("cif_idx", -1) or -1),
        }
        return final_atoms, info

    return None, {}


# ─────────────────────────────────────────────────────────────────────────────
# Cheap structural scalars
# ─────────────────────────────────────────────────────────────────────────────

def _assert_orthorhombic(cell: np.ndarray) -> np.ndarray:
    off_diag = np.abs(cell - np.diag(np.diag(cell))).max()
    if off_diag > 1e-6:
        raise ValueError(
            "cKDTree-based scalars require an orthorhombic cell; "
            f"max off-diagonal = {off_diag:.3e}"
        )
    return np.diag(cell)


def compute_cheap_scalars(atoms: Atoms) -> dict:
    """Compute per-structure scalars that don't require MACE.

    All distance-based work uses scipy.spatial.cKDTree with PBC via
    ``boxsize`` (orthorhombic only — the user's CELL_DIMS is diagonal).
    """
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    box = _assert_orthorhombic(cell)
    n = len(atoms)
    volume = float(np.prod(box))
    density = (n / volume) if volume > 0 else 0.0

    pos = np.asarray(atoms.positions, dtype=np.float64)
    pos_wrapped = pos - np.floor(pos / box) * box
    tree = cKDTree(pos_wrapped, boxsize=box)

    # ── Nearest-neighbor distance ─
    dd, _ = tree.query(pos_wrapped, k=2)
    min_pair_dist = float(dd[:, 1].min())

    # ── Composition (atom fraction by element) ─
    cnt = Counter(int(z) for z in atoms.numbers)
    total = sum(cnt.values())
    composition = {
        chemical_symbols[z]: round(c / total, 6) for z, c in cnt.items()
    }

    # ── Mean coordination by species ─
    # query_ball_tree against self gives every pair within COORD_CUTOFF_A;
    # subtract 1 for the self-match.  N² in pairs found but cKDTree keeps
    # the constant low.
    indices_within = tree.query_ball_tree(tree, r=COORD_CUTOFF_A)
    coord_counts = np.fromiter(
        (max(len(lst) - 1, 0) for lst in indices_within),
        dtype=np.int64, count=len(indices_within),
    )
    mean_coord: dict = {}
    for z in np.unique(atoms.numbers):
        mask = atoms.numbers == z
        if mask.any():
            mean_coord[chemical_symbols[int(z)]] = round(
                float(coord_counts[mask].mean()), 4
            )

    return {
        "n_atoms":                   n,
        "cell_volume_A3":            volume,
        "density_at_per_A3":         density,
        "min_pair_dist_A":           min_pair_dist,
        "composition_json":          json.dumps(composition),
        "mean_coord_by_species_json": json.dumps(mean_coord),
    }


# ─────────────────────────────────────────────────────────────────────────────
# G3Distribution measurement + plot rendering
# ─────────────────────────────────────────────────────────────────────────────

def measure_distribution(atoms: Atoms, label: str) -> G3Distribution:
    """One numba-accelerated pass that gives us g(r), ADF (from g3), and g3."""
    dist = G3Distribution(atoms, label=label)
    dist.measure_g3(
        r_max=G3_R_MAX, r_step=G3_R_STEP, phi_num_bins=G3_PHI_NBINS,
        backend=G3_BACKEND,
        sample_fraction=G3_SAMPLE_FRACTION,
        sample_rng_seed=G3_SAMPLE_RNG_SEED,
    )
    return dist


def _species_counts(dist: G3Distribution) -> np.ndarray:
    """Atom counts per dist.species channel."""
    numbers = dist.atoms.numbers
    return np.array(
        [int(np.sum(numbers == z)) for z in dist.species],
        dtype=np.float64,
    )


def plot_gr(dist: G3Distribution, save_path: Path) -> None:
    """Per-species-pair g(r) normalized to ideal density.

    Counts in ``dist.g2count[a, b, k]`` come from a sampled origin set
    (size ``_origin_sample_size``).  Normalization is

        g_ab(r_k) = count / (N_orig_a * (N_b - δ_ab) / V * 4πr_k²dr)

    where N_orig_a is the number of α-origins SAMPLED (not total α atoms).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    r = dist.bin_centers
    g2 = dist.g2count
    counts = _species_counts(dist)
    syms = [chemical_symbols[int(z)] for z in dist.species]

    volume = float(abs(np.linalg.det(np.asarray(dist.atoms.cell.array))))
    sample_frac = float(getattr(dist, "_origin_sample_fraction", 1.0) or 1.0)
    shell_vol = 4.0 * np.pi * r * r * dist.r_step

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=PNG_DPI)
    for i, sym_i in enumerate(syms):
        for j, sym_j in enumerate(syms):
            if j < i:
                continue  # symmetric — skip the dual entry
            n_origin_a = counts[i] * sample_frac
            n_b_per_origin = counts[j] - (1.0 if i == j else 0.0)
            if n_origin_a <= 0 or n_b_per_origin <= 0 or volume <= 0:
                continue
            denom = n_origin_a * (n_b_per_origin / volume) * shell_vol
            with np.errstate(divide="ignore", invalid="ignore"):
                gr = np.where(denom > 0, g2[i, j] / denom, 0.0)
            ax.plot(r, gr, lw=1.2, label=f"{sym_i}–{sym_j}")
    ax.axhline(1.0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_xlim(0.0, dist.r_max)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(dist.label, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_adf(dist: G3Distribution, save_path: Path) -> None:
    """Bond-angle distribution within the first-neighbor shell, per triplet.

    Marginalize g3count over (r₁, r₂) within ``[SHELL_R_MIN_A, SHELL_R_MAX_A]``
    → 1D distribution in φ.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    r = dist.bin_centers
    phi_deg = dist.phi_deg
    g3 = dist.g3count

    r_mask = (r >= SHELL_R_MIN_A) & (r <= SHELL_R_MAX_A)
    if not r_mask.any():
        return

    # Slice over r₁ and r₂ within the shell, sum.  Two-step slicing
    # avoids materializing the full broadcasted mask tensor.
    g3_r1 = g3[:, r_mask, :, :]       # (T, n_r_in, num_r, num_phi)
    g3_r1r2 = g3_r1[:, :, r_mask, :]  # (T, n_r_in, n_r_in, num_phi)
    adf = g3_r1r2.sum(axis=(1, 2))     # (T, num_phi)

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=PNG_DPI)
    for t in range(adf.shape[0]):
        total = float(adf[t].sum())
        if total <= 0:
            continue
        label = dist.pair_labels[t] if t < len(dist.pair_labels) else str(t)
        ax.plot(phi_deg, adf[t] / total, lw=1.1, label=label)
    ax.set_xlabel("angle φ (deg)")
    ax.set_ylabel("normalized density")
    ax.set_xlim(0.0, 180.0)
    ax.legend(fontsize=7, loc="best", ncol=1)
    ax.set_title(
        f"{dist.label}  (r₁,r₂ ∈ [{SHELL_R_MIN_A}, {SHELL_R_MAX_A}] Å)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_g3(dist: G3Distribution, save_path: Path) -> None:
    """Projection (ii): per-triplet shell-integrated (r₂, φ) heatmaps.

    Sum g3count over r₁ within the first-neighbor shell, leaving a 2D
    (r₂, φ) field per triplet channel.  One subplot per triplet; layout
    picks the squarest grid that fits ``num_triplets`` channels.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    r = dist.bin_centers
    g3 = dist.g3count
    n_triplets = g3.shape[0]

    r1_mask = (r >= SHELL_R_MIN_A) & (r <= SHELL_R_MAX_A)
    if not r1_mask.any():
        return
    g3_shell = g3[:, r1_mask, :, :].sum(axis=1)  # (T, num_r2, num_phi)

    ncols = max(1, min(4, int(np.ceil(np.sqrt(n_triplets)))))
    nrows = int(np.ceil(n_triplets / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.6 * ncols, 2.2 * nrows),
        dpi=PNG_DPI, squeeze=False,
    )

    extent = (0.0, dist.r_max, 0.0, 180.0)
    for t in range(n_triplets):
        ax = axes[t // ncols, t % ncols]
        # imshow expects (rows, cols) — we want phi on y and r₂ on x.
        ax.imshow(
            g3_shell[t].T, origin="lower", aspect="auto",
            cmap="viridis", extent=extent,
        )
        label = dist.pair_labels[t] if t < len(dist.pair_labels) else str(t)
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("r₂ (Å)", fontsize=8)
        ax.set_ylabel("φ (deg)", fontsize=8)
        ax.tick_params(labelsize=7)

    for t in range(n_triplets, nrows * ncols):
        axes[t // ncols, t % ncols].axis("off")

    fig.suptitle(
        f"{dist.label} — g3  (r₁ ∈ [{SHELL_R_MIN_A}, {SHELL_R_MAX_A}] Å)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory enrichment
# ─────────────────────────────────────────────────────────────────────────────

def _plot_paths_for(compound: str, mp_id: str, regime: str,
                    rng_seed: int) -> tuple[Path, Path, Path]:
    sys_label = f"{compound}_{mp_id}" if mp_id else (compound or "unknown")
    plot_dir = PLOTS_ROOT / sys_label / regime / f"seed{rng_seed}"
    return plot_dir / "gr.png", plot_dir / "adf.png", plot_dir / "g3.png"


def enrich_one(traj_path: Path, root: Path) -> EnrichmentRow:
    """Compute scalars + plots for one trajectory; return an EnrichmentRow."""
    row = EnrichmentRow(source_file=str(traj_path.relative_to(root)))
    try:
        cmp, mp_id, regime, seed = parse_filename_components(traj_path)
        row.compound = cmp
        row.mp_id    = mp_id
        row.regime   = regime
        row.rng_seed = seed
        row.cif_filename = f"{mp_id}_{cmp}.cif" if (cmp and mp_id) else ""

        atoms, info = load_traj(traj_path)
        if atoms is None:
            row.error = "could not load trajectory"
            return row

        # Embedded info takes precedence over filename-derived values.
        if info.get("run_id"):   row.run_id = info["run_id"]
        if info.get("regime"):   row.regime = info["regime"]
        if info.get("rng_seed", -1) != -1: row.rng_seed = int(info["rng_seed"])
        if info.get("compound"): row.compound = info["compound"]
        if info.get("mp_id"):    row.mp_id    = info["mp_id"]
        # Keep cif_filename consistent with the final compound/mp_id values.
        if row.compound and row.mp_id:
            row.cif_filename = f"{row.mp_id}_{row.compound}.cif"

        # ── Cheap scalars ─
        for k, v in compute_cheap_scalars(atoms).items():
            setattr(row, k, v)

        # ── PNGs ─
        gr_p, adf_p, g3_p = _plot_paths_for(
            row.compound, row.mp_id, row.regime, row.rng_seed,
        )
        # A plot is "done" if either it's already on disk or it's been
        # disabled by the CONFIG flag.  This lets a re-run with RENDER_ADF=False
        # short-circuit cleanly when the gr/g3 PNGs already exist.
        plot_targets = [
            (RENDER_GR,  gr_p),
            (RENDER_ADF, adf_p),
            (RENDER_G3,  g3_p),
        ]
        needs_render = any(enabled and not p.is_file()
                           for enabled, p in plot_targets)
        if needs_render:
            label = f"{row.compound}_{row.mp_id}_{row.regime}_seed{row.rng_seed}"
            dist = measure_distribution(atoms, label)
            if RENDER_GR  and not gr_p.is_file():  plot_gr(dist, gr_p)
            if RENDER_ADF and not adf_p.is_file(): plot_adf(dist, adf_p)
            if RENDER_G3  and not g3_p.is_file():  plot_g3(dist, g3_p)
            row.g3_origin_sample_size     = int(
                getattr(dist, "_origin_sample_size", -1) or -1
            )
            row.g3_origin_sample_fraction = float(
                getattr(dist, "_origin_sample_fraction", -1.0) or -1.0
            )

        # Only record the path if the plot was actually produced (or already
        # existed).  Disabled plots get an empty string in the CSV.
        row.gr_png  = str(gr_p.relative_to(root))  if gr_p.is_file()  else ""
        row.adf_png = str(adf_p.relative_to(root)) if adf_p.is_file() else ""
        row.g3_png  = str(g3_p.relative_to(root))  if g3_p.is_file()  else ""

        row.enriched_at_utc = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        row.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(limit=3, file=sys.stdout)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# CSV I/O — resume + append
# ─────────────────────────────────────────────────────────────────────────────

def _enrichment_csv_path() -> Path:
    """Per-rank CSV under multi-rank, shared one otherwise."""
    if _IS_MULTI_RANK:
        return ENRICHMENT_CSV.with_name(
            f"{ENRICHMENT_CSV.stem}.rank{_GLOBAL_RANK}{ENRICHMENT_CSV.suffix}"
        )
    return ENRICHMENT_CSV


def load_completed_source_files(csv_path: Path) -> set[str]:
    """Read the existing CSV and return the set of completed ``source_file``s."""
    done: set[str] = set()
    if not csv_path.is_file():
        return done
    try:
        with csv_path.open() as fh:
            for row in csv.DictReader(fh):
                sf = (row.get("source_file") or "").strip()
                err = (row.get("error") or "").strip()
                if sf and not err:
                    done.add(sf)
    except Exception as exc:
        print(f"[warn] could not read {csv_path}: {exc}")
    return done


def append_row(csv_path: Path, row: EnrichmentRow) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existed = csv_path.is_file()
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_enrichment_fieldnames())
        if not existed:
            w.writeheader()
        w.writerow(asdict(row))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not OUTPUT_ROOT.is_dir():
        sys.exit(f"[abort] OUTPUT_ROOT not found: {OUTPUT_ROOT}")
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

    files_all = discover_traj_files(OUTPUT_ROOT)
    if not files_all:
        sys.exit(f"[abort] no .xyz/.npz trajectories under {OUTPUT_ROOT}")
    if _GLOBAL_RANK == 0:
        print(f"[scan] {len(files_all)} trajectory files total in {OUTPUT_ROOT}")

    # Round-robin partition.  Each rank gets a disjoint slice of files.
    files = files_all[_GLOBAL_RANK::_WORLD_SIZE]
    csv_path = _enrichment_csv_path()
    print(
        f"[rank {_GLOBAL_RANK}/{_WORLD_SIZE}] my slice: {len(files)} files  "
        f"→ {csv_path.name}", flush=True,
    )

    completed = (
        load_completed_source_files(csv_path)
        if SKIP_IF_ALREADY_ENRICHED else set()
    )
    if completed:
        print(f"[resume] {len(completed)} files already enriched; will skip")

    t_start = time.time()
    n_ok = n_skip = n_fail = 0

    for i, traj in enumerate(files, 1):
        rel = str(traj.relative_to(OUTPUT_ROOT))
        if rel in completed:
            n_skip += 1
            continue

        t0 = time.time()
        row = enrich_one(traj, OUTPUT_ROOT)
        dt = time.time() - t0
        elapsed = time.time() - t_start
        rate = i / max(elapsed, 1e-9) * 60.0
        n_remaining = len(files) - i
        eta_min = (n_remaining * (elapsed / max(i, 1))) / 60.0

        if row.error:
            n_fail += 1
            print(
                f"  [{i:>5d}/{len(files)}] ✗ {traj.name}  err={row.error}",
                flush=True,
            )
        else:
            n_ok += 1
            print(
                f"  [{i:>5d}/{len(files)}] ✓ {traj.name}  "
                f"{dt:5.1f}s  rate={rate:5.1f}/min  ETA={eta_min:5.1f}m",
                flush=True,
            )
        append_row(csv_path, row)

    print()
    print("=" * 60)
    print(f"  enriched : {n_ok}")
    print(f"  skipped  : {n_skip}")
    print(f"  failed   : {n_fail}")
    print(f"  wall     : {(time.time() - t_start) / 60:.1f} min")
    print(f"  output   : {csv_path}")
    print(f"  plots    : {PLOTS_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
