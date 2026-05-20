"""Post-hoc QC filter for phase-contrast SiO₂ trajectories.

Reads the manifest written by ``generate_phase_contrast_sio2.py``,
computes for each trajectory whether the relaxation actually reached
the polymorph it was supposed to reach, and writes:

  * ``qc_report.csv``     — every trajectory with measured metrics.
  * ``manifest_passed.csv`` — only the rows that passed all checks.

QC criteria (per trajectory's ``best_positions``):

  1. **Mean Si coordination** — count O atoms within ``COORD_CUTOFF`` Å
     of each Si and average across Si atoms.  4-coord polymorphs
     should give ~4; stishovite should give ~6.
  2. **First Si–O peak position** — peak of the Si–O radial
     distribution.  4-coord polymorphs target ~1.61 Å; stishovite
     targets ~1.78 Å.

A row passes only if both checks are within tolerance.  The
``manifest_passed.csv`` is what the dataloader should consume during
training so the model never sees a stishovite-labeled trajectory
that's actually stuck in 4-coord.

Run after ``generate_phase_contrast_sio2.py``:

    python filter_phase_contrast_qc.py

(no GPU needed; pure CPU + numpy)
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# Same OUT_DIR as the generator.  Manifest + .npz files live here.
DATA_DIR = Path(__file__).parent / "data" / "phase_contrast_v1" / "SiO2"
INPUT_MANIFEST = DATA_DIR / "manifest.csv"
QC_REPORT = DATA_DIR / "qc_report.csv"
PASSED_MANIFEST = DATA_DIR / "manifest_passed.csv"

# Per-polymorph QC targets.  ``coord`` is the expected mean Si
# coordination number; ``peak`` is the first Si–O peak position (Å).
# ``coord_tol`` and ``peak_tol`` are the half-widths of the
# pass-window.  Tolerances are generous — liquid-relaxed structures
# are imperfect even when they reach the right polymorph.
POLYMORPH_QC: dict[str, dict[str, float]] = {
    "alpha_quartz":       {"coord": 4.0, "peak": 1.61, "coord_tol": 0.7, "peak_tol": 0.10},
    "alpha_cristobalite": {"coord": 4.0, "peak": 1.61, "coord_tol": 0.7, "peak_tol": 0.10},
    "beta_cristobalite":  {"coord": 4.0, "peak": 1.61, "coord_tol": 0.7, "peak_tol": 0.10},
    "coesite":            {"coord": 4.0, "peak": 1.61, "coord_tol": 0.7, "peak_tol": 0.10},
    "stishovite":         {"coord": 6.0, "peak": 1.78, "coord_tol": 0.8, "peak_tol": 0.12},
}

COORD_CUTOFF = 2.2     # Å — anything closer than this is a "bonded" Si-O
PDF_BIN_WIDTH = 0.02   # Å — histogram bin width for peak detection
PDF_R_MIN = 1.3        # Å — search window for first Si-O peak
PDF_R_MAX = 2.2        # Å

# ─────────────────────────────────────────────────────────────────────────────

import csv
import sys

import numpy as np


def _min_image_distance_matrix(
    pos_a: np.ndarray, pos_b: np.ndarray, cell: np.ndarray,
) -> np.ndarray:
    """Pairwise min-image distances between two atom sets under PBC.

    Returns array of shape (len(pos_a), len(pos_b)).  Uses fractional
    coordinates round-trip — works for any (orthogonal or oblique)
    cell.  O(Na * Nb) memory; for the Si-only by O-only matrix in a
    14k-atom cell this is ~5k × ~10k = 50M floats = 200 MB.  Doable
    but borderline; we chunk over the larger axis if needed.
    """
    inv_cell = np.linalg.inv(cell)
    # Process in chunks over pos_a to keep peak memory bounded.
    CHUNK = 512
    n_a = pos_a.shape[0]
    out = np.empty((n_a, pos_b.shape[0]), dtype=np.float32)
    for s in range(0, n_a, CHUNK):
        e = min(s + CHUNK, n_a)
        delta = pos_b[None, :, :] - pos_a[s:e, None, :]
        delta_frac = delta @ inv_cell.T
        delta_frac -= np.round(delta_frac)
        delta = delta_frac @ cell
        out[s:e] = np.sqrt(np.einsum("...d,...d->...", delta, delta)).astype(np.float32)
    return out


def _measure_quality(
    positions: np.ndarray,
    species: np.ndarray,
    cell: np.ndarray,
) -> tuple[float, float, int]:
    """Compute (mean_si_coord, first_sio_peak_A, n_si).

    Returns ``(0.0, 0.0, 0)`` if the structure has no Si atoms (shouldn't
    happen for SiO2 but we guard anyway).
    """
    si_mask = species == 14
    o_mask = species == 8
    pos_si = positions[si_mask]
    pos_o = positions[o_mask]
    n_si = int(pos_si.shape[0])
    if n_si == 0 or pos_o.shape[0] == 0:
        return 0.0, 0.0, 0

    dists = _min_image_distance_matrix(pos_si, pos_o, cell)  # (n_si, n_o)

    # Mean Si coordination = mean # of O within COORD_CUTOFF per Si.
    coord_counts = (dists < COORD_CUTOFF).sum(axis=1)
    mean_si_coord = float(coord_counts.mean())

    # First Si-O peak: histogram in the [PDF_R_MIN, PDF_R_MAX] window
    # and pick the mode bin.  Window is chosen to bracket both 4-coord
    # (~1.61 Å) and 6-coord (~1.78 Å) Si-O bonds without leaking into
    # second-shell distances (~2.6 Å).
    flat = dists[(dists >= PDF_R_MIN) & (dists <= PDF_R_MAX)]
    if flat.size == 0:
        return mean_si_coord, 0.0, n_si
    n_bins = int(round((PDF_R_MAX - PDF_R_MIN) / PDF_BIN_WIDTH))
    hist, edges = np.histogram(flat, bins=n_bins, range=(PDF_R_MIN, PDF_R_MAX))
    peak_bin = int(np.argmax(hist))
    peak_r = float(edges[peak_bin] + 0.5 * PDF_BIN_WIDTH)
    return mean_si_coord, peak_r, n_si


def main() -> None:
    if not INPUT_MANIFEST.is_file():
        sys.exit(f"manifest not found: {INPUT_MANIFEST}")

    with open(INPUT_MANIFEST, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        manifest_columns = reader.fieldnames or []

    print(f"Reading {len(rows)} trajectories from {INPUT_MANIFEST}")
    print()

    report_columns = list(manifest_columns) + [
        "mean_si_coord", "first_sio_peak", "n_si",
        "coord_target", "peak_target", "passed", "fail_reason",
    ]

    qc_writer_h = open(QC_REPORT, "w", newline="")
    qc_writer = csv.DictWriter(qc_writer_h, fieldnames=report_columns)
    qc_writer.writeheader()

    passed_writer_h = open(PASSED_MANIFEST, "w", newline="")
    passed_writer = csv.DictWriter(passed_writer_h, fieldnames=manifest_columns)
    passed_writer.writeheader()

    n_pass = 0
    n_fail_coord = 0
    n_fail_peak = 0
    n_fail_unknown_poly = 0
    per_poly_pass: dict[str, int] = {}
    per_poly_total: dict[str, int] = {}

    for row in rows:
        polymorph = row["polymorph"]
        per_poly_total[polymorph] = per_poly_total.get(polymorph, 0) + 1
        qc_spec = POLYMORPH_QC.get(polymorph)
        if qc_spec is None:
            row_out = dict(row)
            row_out.update({
                "mean_si_coord": "", "first_sio_peak": "", "n_si": "",
                "coord_target": "", "peak_target": "",
                "passed": "False",
                "fail_reason": f"unknown polymorph: {polymorph}",
            })
            qc_writer.writerow(row_out)
            n_fail_unknown_poly += 1
            continue

        npz_path = DATA_DIR / row["filename"]
        with np.load(npz_path) as npz:
            positions = np.asarray(npz["best_positions"], dtype=np.float32)
            species = np.asarray(npz["species_numbers"], dtype=np.int64)
            cell = np.asarray(npz["cell"], dtype=np.float64)

        mean_coord, peak_r, n_si = _measure_quality(positions, species, cell)

        coord_ok = abs(mean_coord - qc_spec["coord"]) <= qc_spec["coord_tol"]
        peak_ok = abs(peak_r - qc_spec["peak"]) <= qc_spec["peak_tol"]
        passed = coord_ok and peak_ok
        fail_reasons = []
        if not coord_ok:
            n_fail_coord += 1
            fail_reasons.append(
                f"coord={mean_coord:.2f} vs {qc_spec['coord']}±{qc_spec['coord_tol']}"
            )
        if not peak_ok:
            n_fail_peak += 1
            fail_reasons.append(
                f"peak={peak_r:.3f} vs {qc_spec['peak']}±{qc_spec['peak_tol']}"
            )

        row_out = dict(row)
        row_out.update({
            "mean_si_coord": f"{mean_coord:.3f}",
            "first_sio_peak": f"{peak_r:.3f}",
            "n_si": str(n_si),
            "coord_target": str(qc_spec["coord"]),
            "peak_target": str(qc_spec["peak"]),
            "passed": str(passed),
            "fail_reason": "; ".join(fail_reasons) if fail_reasons else "",
        })
        qc_writer.writerow(row_out)

        if passed:
            passed_writer.writerow({k: row[k] for k in manifest_columns})
            n_pass += 1
            per_poly_pass[polymorph] = per_poly_pass.get(polymorph, 0) + 1

    qc_writer_h.close()
    passed_writer_h.close()

    n_total = len(rows)
    print(f"Pass: {n_pass} / {n_total}  ({100 * n_pass / max(n_total, 1):.1f}%)")
    print()
    print("Per-polymorph pass rates:")
    for poly in sorted(per_poly_total):
        p = per_poly_pass.get(poly, 0)
        t = per_poly_total[poly]
        print(f"  {poly:>20s}: {p:4d} / {t:4d}  ({100 * p / max(t, 1):5.1f}%)")
    print()
    print(f"Failures: {n_fail_coord} coord, {n_fail_peak} peak, "
          f"{n_fail_unknown_poly} unknown polymorph")
    print()
    print(f"qc_report:        {QC_REPORT}")
    print(f"passed_manifest:  {PASSED_MANIFEST}")
    print()
    print("Train with PASSED_MANIFEST to ensure the model never sees "
          "stishovite-labeled trajectories stuck in 4-coord (or any "
          "other phase-target mismatch).")


if __name__ == "__main__":
    main()
