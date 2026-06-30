"""MACE multi-crop per-structure energy enrichment.

Walks ``OUTPUT_ROOT/*_generated/*.{xyz,npz}`` and computes a MACE-MPA-0
single-point on N_CROPS padded crops of each trajectory.  Per-crop:

  * extract a padded cube (PADDED_HALF_A on each side) around a random center
  * run MACE single-point on the cluster (no PBC — vacuum boundaries)
  * score only the interior cube (INTERIOR_HALF_A on each side) so the
    edge atoms' missing-neighbor effects don't pollute the per-atom average
  * record per-atom energy + max/mean force over the interior

The buffer (PADDED_HALF_A - INTERIOR_HALF_A) must be ≥ MACE's cutoff (≈5 Å)
so interior atoms see their full neighbor environment and look bulk-like.

Aggregates across crops:

  * mace_energy_per_atom_eV_mean / std / min / max
  * mace_fmax_eV_per_A_mean / max
  * mace_fmean_eV_per_A_mean
  * mace_crops_json — raw per-crop list for drill-down

Crops are stratified along the long axis (default Z) — N_CROPS bands, one
random center per band.  For uniform amorphous structures this is equivalent
to fully-random; for graded-disorder structures it guarantees the gradient
gets sampled.

This is the EXPENSIVE pass — budget ~7 s/crop × 4 crops × 14 k trajectories
≈ 100 GPU-hours.  Runs only on a GPU node.

Idempotent — resume-safe via source_file lookup in the existing CSV.
Multi-rank: each rank loads MACE on its own GPU, writes
``mace_enrichment.rank{N}.csv``.  Single-rank writes ``mace_enrichment.csv``.
``build_dataset_table.py`` globs ``mace_enrichment*.csv`` automatically.

Usage:
    /global/common/software/m5020/ehrdt/tricor/bin/python \\
        scripts/macerelax/enrich_mace_energy.py
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

OUTPUT_ROOT          = Path("/pscratch/sd/e/ehrdt/macerelax/generated_cnos_v1")
MACE_ENRICHMENT_CSV  = OUTPUT_ROOT / "mace_enrichment.csv"

# Which file types to enrich (NPZ preferred when both exist).
SCAN_XYZ             = True
SCAN_NPZ             = True

# ── MACE model ─────────────────────────────────────────────────────────────
# "medium-mpa-0" matches the teacher used by the pilot's MACE-as-teacher
# trajectories.  Other options: "small-mpa-0", "large-mpa-0", or an
# absolute path to a downloaded .pt checkpoint.
MACE_MODEL           = "medium-mpa-0"
MACE_DEFAULT_DTYPE   = "float32"
GPU_ID               = 0   # only used as fallback for single-process runs

# ── Crop geometry ──────────────────────────────────────────────────────────
# Padded cube side = 2*PADDED_HALF_A; interior cube side = 2*INTERIOR_HALF_A.
# Buffer = PADDED_HALF_A - INTERIOR_HALF_A.  Must be ≥ MACE cutoff (5 Å) for
# interior atoms to be bulk-correct.
PADDED_HALF_A        = 15.0   # 30³ Å padded cube
INTERIOR_HALF_A      = 10.0   # 20³ Å interior scoring region
N_CROPS              = 4

# Axis to stratify along (0=x, 1=y, 2=z).  The longest cell dimension is
# the natural choice (graded systems live along this axis).  For the
# production CELL_DIMS=(100, 100, 400), axis=2 stratifies along z.
STRATIFY_AXIS        = 2

# Crop-position PRNG seed.  Combined with the trajectory's rng_seed so that
# (rng_seed, crop_idx) → deterministic crop center.  Re-running enrichment
# always produces the same crops.
CROP_RNG_BASE        = 12345

# ── Resume + resource ───────────────────────────────────────────────────
SKIP_IF_ALREADY_ENRICHED = True
NUM_THREADS          = 4

# ─────────────────────────────────────────────────────────────────────────────
# Env caps must be set BEFORE numpy / torch import.
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Multi-rank detection — same scheme as the other scripts ────────────────
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

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
if not _IS_MULTI_RANK:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(GPU_ID))

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, str(NUM_THREADS))

import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass, asdict, fields as dc_fields
from datetime import datetime, timezone

import numpy as np
import torch
torch.set_num_threads(NUM_THREADS)

from ase.atoms import Atoms
from ase.io import read as ase_read

from mace.calculators import mace_mp


# ─────────────────────────────────────────────────────────────────────────────
# EnrichmentMaceRow — per-traj output schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnrichmentMaceRow:
    # ── joining keys ──────────────────────────────────────────────────────
    run_id:                str = ""
    cif_filename:          str = ""
    compound:              str = ""
    mp_id:                 str = ""
    regime:                str = ""
    rng_seed:              int = -1
    source_file:           str = ""
    schema_version:        int = 1
    # ── aggregates across crops ───────────────────────────────────────────
    mace_n_crops:                       int   = 0
    mace_energy_per_atom_eV_mean:       float = float("nan")
    mace_energy_per_atom_eV_std:        float = float("nan")
    mace_energy_per_atom_eV_min:        float = float("nan")
    mace_energy_per_atom_eV_max:        float = float("nan")
    mace_fmax_eV_per_A_mean:            float = float("nan")
    mace_fmax_eV_per_A_max:             float = float("nan")
    mace_fmean_eV_per_A_mean:           float = float("nan")
    mace_score_n_atoms_total:           int   = 0
    # ── per-crop raw values (JSON list of dicts) ──────────────────────────
    mace_crops_json:                    str   = ""
    # ── crop / model settings (so the row is self-describing) ─────────────
    mace_padded_half_A:                 float = PADDED_HALF_A
    mace_interior_half_A:               float = INTERIOR_HALF_A
    mace_stratify_axis:                 int   = STRATIFY_AXIS
    mace_model:                         str   = MACE_MODEL
    mace_eval_dtype:                    str   = MACE_DEFAULT_DTYPE
    # ── timing / status ───────────────────────────────────────────────────
    eval_wall_s:                        float = -1.0
    enriched_at_utc:                    str   = ""
    error:                              str   = ""


def _row_fieldnames() -> list[str]:
    return [f.name for f in dc_fields(EnrichmentMaceRow)]


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory file discovery + loading — mirror enrich_metadata.py's helpers
# ─────────────────────────────────────────────────────────────────────────────

_REGIMES_KNOWN = (
    "amorphous", "SRO", "MRO", "LRO",
    "nanocrystalline", "crystalline_30",
)


def discover_traj_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if SCAN_NPZ:
        files.extend(sorted(root.glob("*_generated/*.npz")))
    if SCAN_XYZ:
        files.extend(sorted(root.glob("*_generated/*.xyz")))
    return files


def parse_filename_components(path: Path) -> tuple[str, str, str, int]:
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


def _npz_scalar(nz, key: str, default):
    if key not in nz.files:
        return default
    try:
        return nz[key].item()
    except Exception:
        return default


def load_traj(path: Path) -> tuple[Atoms | None, dict]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as nz:
            final = np.asarray(nz["final_positions"], dtype=np.float64)
            cell = np.asarray(nz["cell"], dtype=np.float64)
            species = np.asarray(nz["species_numbers"], dtype=int)
            atoms = Atoms(numbers=species, positions=final, cell=cell, pbc=True)
            info = {
                "run_id":  _npz_scalar(nz, "run_id", "") or "",
                "regime":  _npz_scalar(nz, "regime", "") or "",
                "rng_seed": int(_npz_scalar(nz, "rng_seed", -1) or -1),
                "compound": _npz_scalar(nz, "compound", "") or "",
                "mp_id":   _npz_scalar(nz, "mp_id", "") or "",
            }
            return atoms, info
    if suffix == ".xyz":
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
        }
        return final_atoms, info
    return None, {}


# ─────────────────────────────────────────────────────────────────────────────
# Padded crop construction + scoring
# ─────────────────────────────────────────────────────────────────────────────

def _assert_orthorhombic(cell: np.ndarray) -> np.ndarray:
    off_diag = np.abs(cell - np.diag(np.diag(cell))).max()
    if off_diag > 1e-6:
        raise ValueError(
            f"Crop logic requires orthorhombic cell; max off-diagonal "
            f"= {off_diag:.3e}"
        )
    return np.diag(cell)


def stratified_crop_centers(cell_diag: np.ndarray,
                            rng: np.random.Generator) -> list[np.ndarray]:
    """Generate N_CROPS centers, stratified along STRATIFY_AXIS.

    Bands along the stratification axis partition
    ``[PADDED_HALF_A, cell - PADDED_HALF_A]`` into N_CROPS equal slices;
    each band contributes one random center.  Other axes are uniformly
    random within the same buffer constraint.
    """
    centers: list[np.ndarray] = []
    band_min = PADDED_HALF_A
    band_max = float(cell_diag[STRATIFY_AXIS]) - PADDED_HALF_A
    if band_max <= band_min:
        # Cell too small along the stratification axis to fit even one crop.
        return centers
    band_width = (band_max - band_min) / N_CROPS

    for crop_idx in range(N_CROPS):
        center = np.empty(3, dtype=np.float64)
        for ax in range(3):
            if ax == STRATIFY_AXIS:
                lo = band_min + crop_idx * band_width
                hi = lo + band_width
                center[ax] = float(rng.uniform(lo, hi))
            else:
                lo = PADDED_HALF_A
                hi = float(cell_diag[ax]) - PADDED_HALF_A
                if hi <= lo:
                    center[ax] = float(cell_diag[ax]) * 0.5
                else:
                    center[ax] = float(rng.uniform(lo, hi))
        centers.append(center)
    return centers


def extract_padded_crop(positions: np.ndarray,
                        cell_diag: np.ndarray,
                        species: np.ndarray,
                        center: np.ndarray,
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut a padded cube around ``center`` with PBC unwrapping.

    Returns:
        crop_positions    — positions of atoms inside the padded cube, with
                            origin shifted to the cube's lower corner so all
                            coords lie in ``[0, 2*PADDED_HALF_A]``.
        crop_species      — atomic numbers for those atoms.
        interior_mask     — bool array over crop atoms; True where the atom
                            lies inside the interior cube (scoring region).
    """
    # Compute minimum-image displacement of every atom from the center.
    rel = positions - center
    rel -= np.round(rel / cell_diag) * cell_diag

    inside_padded = np.all(np.abs(rel) <= PADDED_HALF_A, axis=1)
    rel_in = rel[inside_padded]
    crop_positions = rel_in + PADDED_HALF_A   # shift to [0, 2*PADDED_HALF_A]
    crop_species = species[inside_padded]
    interior_mask = np.all(np.abs(rel_in) <= INTERIOR_HALF_A, axis=1)
    return crop_positions, crop_species, interior_mask


def evaluate_crop(calc, crop_positions: np.ndarray, crop_species: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Run one MACE single-point on the cluster.  No PBC (vacuum buffer).

    Returns ``(per_atom_energies_eV, forces_eV_per_A)``.  Per-atom energies
    are obtained from ``calc.results["node_energies"]`` (MACE's default) or
    ``["energies"]`` as a fallback.  Forces are the standard ``forces`` key.
    """
    cluster_side = 2.0 * PADDED_HALF_A
    atoms = Atoms(
        numbers=crop_species.astype(int),
        positions=np.asarray(crop_positions, dtype=np.float64),
        cell=np.eye(3) * cluster_side,
        pbc=False,
    )
    atoms.calc = calc
    # Force a fresh forward pass even if calc had previous results.
    calc.results = {}
    _ = atoms.get_potential_energy()

    # MACE typically exposes per-atom energies under "node_energies"; some
    # builds use "energies".  Whichever is present, use it.
    res = calc.results
    per_atom = res.get("node_energies", None)
    if per_atom is None:
        per_atom = res.get("energies", None)
    if per_atom is None:
        raise RuntimeError(
            "MACE calculator did not expose per-atom energies "
            "(neither 'node_energies' nor 'energies' in results); "
            "cannot score the interior atoms."
        )
    forces = res.get("forces")
    if forces is None:
        raise RuntimeError("MACE calculator did not return forces.")
    return np.asarray(per_atom, dtype=np.float64), np.asarray(forces, dtype=np.float64)


def score_interior(per_atom_energies: np.ndarray, forces: np.ndarray,
                   interior_mask: np.ndarray) -> dict:
    """Reduce per-atom MACE outputs over the interior atoms only."""
    n_interior = int(interior_mask.sum())
    if n_interior == 0:
        return {
            "n_atoms_interior":      0,
            "energy_per_atom_eV":    float("nan"),
            "fmax_eV_per_A":         float("nan"),
            "fmean_eV_per_A":        float("nan"),
        }
    e_in = per_atom_energies[interior_mask]
    f_in = forces[interior_mask]
    fnorm = np.linalg.norm(f_in, axis=1)
    return {
        "n_atoms_interior":      n_interior,
        "energy_per_atom_eV":    float(e_in.mean()),
        "fmax_eV_per_A":         float(fnorm.max()),
        "fmean_eV_per_A":        float(fnorm.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory enrichment
# ─────────────────────────────────────────────────────────────────────────────

def enrich_one(traj_path: Path, root: Path, calc) -> EnrichmentMaceRow:
    row = EnrichmentMaceRow(source_file=str(traj_path.relative_to(root)))
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

        # Embedded info overrides filename-derived values.
        if info.get("run_id"):   row.run_id   = info["run_id"]
        if info.get("regime"):   row.regime   = info["regime"]
        if info.get("rng_seed", -1) != -1:
            row.rng_seed = int(info["rng_seed"])
        if info.get("compound"): row.compound = info["compound"]
        if info.get("mp_id"):    row.mp_id    = info["mp_id"]
        if row.compound and row.mp_id:
            row.cif_filename = f"{row.mp_id}_{row.compound}.cif"

        cell_arr = np.asarray(atoms.cell.array, dtype=np.float64)
        cell_diag = _assert_orthorhombic(cell_arr)
        positions = np.asarray(atoms.positions, dtype=np.float64)
        species   = np.asarray(atoms.numbers, dtype=int)

        # Deterministic crop generator seeded by (CROP_RNG_BASE, rng_seed).
        rng = np.random.default_rng((CROP_RNG_BASE, max(row.rng_seed, 0)))
        centers = stratified_crop_centers(cell_diag, rng)
        if not centers:
            row.error = (
                f"cell too small for crops "
                f"(cell_diag={cell_diag.tolist()}, padded_half={PADDED_HALF_A})"
            )
            return row

        crops_records: list[dict] = []
        e_per_atom_values: list[float] = []
        fmax_values: list[float] = []
        fmean_values: list[float] = []
        n_interior_total = 0

        t0 = time.time()
        for crop_idx, center in enumerate(centers):
            crop_pos, crop_sp, interior_mask = extract_padded_crop(
                positions, cell_diag, species, center,
            )
            n_cluster = int(crop_pos.shape[0])
            if n_cluster == 0:
                crops_records.append({
                    "crop_idx": crop_idx,
                    "center_xyz": center.tolist(),
                    "n_atoms_cluster": 0,
                    "n_atoms_interior": 0,
                    "energy_per_atom_eV": None,
                    "fmax_eV_per_A": None,
                    "fmean_eV_per_A": None,
                    "skipped": "empty_cluster",
                })
                continue

            per_atom_e, forces = evaluate_crop(calc, crop_pos, crop_sp)
            score = score_interior(per_atom_e, forces, interior_mask)

            crops_records.append({
                "crop_idx":            crop_idx,
                "center_xyz":          [round(float(v), 3) for v in center],
                "n_atoms_cluster":     n_cluster,
                "n_atoms_interior":    score["n_atoms_interior"],
                "energy_per_atom_eV":  score["energy_per_atom_eV"],
                "fmax_eV_per_A":       score["fmax_eV_per_A"],
                "fmean_eV_per_A":      score["fmean_eV_per_A"],
            })
            if score["n_atoms_interior"] > 0:
                e_per_atom_values.append(score["energy_per_atom_eV"])
                fmax_values.append(score["fmax_eV_per_A"])
                fmean_values.append(score["fmean_eV_per_A"])
                n_interior_total += score["n_atoms_interior"]

        row.eval_wall_s = float(time.time() - t0)
        row.mace_n_crops = len(centers)
        row.mace_score_n_atoms_total = n_interior_total
        row.mace_crops_json = json.dumps(crops_records)

        if e_per_atom_values:
            e_arr = np.asarray(e_per_atom_values, dtype=np.float64)
            f_arr = np.asarray(fmax_values, dtype=np.float64)
            fm_arr = np.asarray(fmean_values, dtype=np.float64)
            row.mace_energy_per_atom_eV_mean = float(e_arr.mean())
            row.mace_energy_per_atom_eV_std  = float(e_arr.std(ddof=0))
            row.mace_energy_per_atom_eV_min  = float(e_arr.min())
            row.mace_energy_per_atom_eV_max  = float(e_arr.max())
            row.mace_fmax_eV_per_A_mean      = float(f_arr.mean())
            row.mace_fmax_eV_per_A_max       = float(f_arr.max())
            row.mace_fmean_eV_per_A_mean     = float(fm_arr.mean())
        row.enriched_at_utc = datetime.now(timezone.utc).isoformat()

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        row.error = "OutOfMemoryError"
    except Exception as exc:
        row.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(limit=3, file=sys.stdout)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# CSV I/O — resume + append (same layout as enrich_metadata.py)
# ─────────────────────────────────────────────────────────────────────────────

def _enrichment_csv_path() -> Path:
    if _IS_MULTI_RANK:
        return MACE_ENRICHMENT_CSV.with_name(
            f"{MACE_ENRICHMENT_CSV.stem}.rank{_GLOBAL_RANK}"
            f"{MACE_ENRICHMENT_CSV.suffix}"
        )
    return MACE_ENRICHMENT_CSV


def load_completed_source_files(csv_path: Path) -> set[str]:
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


def append_row(csv_path: Path, row: EnrichmentMaceRow) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existed = csv_path.is_file()
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_row_fieldnames())
        if not existed:
            w.writeheader()
        w.writerow(asdict(row))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not OUTPUT_ROOT.is_dir():
        sys.exit(f"[abort] OUTPUT_ROOT not found: {OUTPUT_ROOT}")

    # Pin per-rank GPU.
    if _IS_MULTI_RANK and torch.cuda.is_available():
        torch.cuda.set_device(_LOCAL_RANK)
        device = torch.device("cuda", _LOCAL_RANK)
        device_str = f"cuda:{_LOCAL_RANK}"
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_str = "cuda"
    else:
        device = torch.device("cpu")
        device_str = "cpu"

    files_all = discover_traj_files(OUTPUT_ROOT)
    if not files_all:
        sys.exit(f"[abort] no .xyz/.npz trajectories under {OUTPUT_ROOT}")
    if _GLOBAL_RANK == 0:
        print(f"[scan] {len(files_all)} trajectory files total")
        print(f"[mace] model={MACE_MODEL}  dtype={MACE_DEFAULT_DTYPE}  "
              f"device={device_str}")
        print(f"[crop] padded_half={PADDED_HALF_A} Å  "
              f"interior_half={INTERIOR_HALF_A} Å  "
              f"n_crops={N_CROPS}  stratify_axis={STRATIFY_AXIS}")

    files = files_all[_GLOBAL_RANK::_WORLD_SIZE]
    csv_path = _enrichment_csv_path()
    print(f"[rank {_GLOBAL_RANK}/{_WORLD_SIZE}] my slice: {len(files)} files  "
          f"→ {csv_path.name}", flush=True)

    completed = (load_completed_source_files(csv_path)
                 if SKIP_IF_ALREADY_ENRICHED else set())
    if completed:
        print(f"[resume] {len(completed)} files already enriched; will skip")

    # Load MACE once and reuse.  Cold start ~30 s; reused across thousands
    # of structures.
    print(f"[mace] loading model (cold start ~30 s)...", flush=True)
    t_load = time.time()
    calc = mace_mp(
        model=MACE_MODEL, device=device_str, default_dtype=MACE_DEFAULT_DTYPE,
    )
    print(f"[mace] loaded in {time.time() - t_load:.1f} s", flush=True)

    t_start = time.time()
    n_ok = n_skip = n_fail = 0
    for i, traj in enumerate(files, 1):
        rel = str(traj.relative_to(OUTPUT_ROOT))
        if rel in completed:
            n_skip += 1
            continue

        t0 = time.time()
        row = enrich_one(traj, OUTPUT_ROOT, calc)
        dt = time.time() - t0
        elapsed = time.time() - t_start
        rate = i / max(elapsed, 1e-9) * 60.0
        eta_min = ((len(files) - i) * (elapsed / max(i, 1))) / 60.0

        if row.error:
            n_fail += 1
            print(f"  [{i:>5d}/{len(files)}] ✗ {traj.name}  err={row.error}",
                  flush=True)
        else:
            n_ok += 1
            print(
                f"  [{i:>5d}/{len(files)}] ✓ {traj.name}  "
                f"{dt:5.1f}s  e/atom={row.mace_energy_per_atom_eV_mean:+.3f}  "
                f"std={row.mace_energy_per_atom_eV_std:.3f}  "
                f"rate={rate:.1f}/min  ETA={eta_min:.1f}m",
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
    print("=" * 60)


if __name__ == "__main__":
    main()
