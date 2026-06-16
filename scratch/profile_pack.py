"""Profile the `pack` stage of generate_with_student.py at 100×100×400.

Times the outer steps of build_supercell + bond_relax, and (via lightweight
monkey-patches) the internal sub-steps of Supercell.generate:

    ase_read
    CoordinationShellTarget.from_atoms
    G3Distribution + measure_g3
    Supercell.__init__           (incl. _build_random_atoms)
    Supercell.generate
        ├── _build_grain_atoms
        ├── _rebuild_spatial_index
        ├── _push_close_pairs_apart        (amorphous, when num_steps > 0)
        └── shell_relax                    (skipped here: num_steps=0)
    Supercell.bond_relax

Production runs use num_steps=0 (handled by bond_relax afterwards), so
shell_relax and _push_close_pairs_apart will be ~0 s.  The two prime
suspects for the 270-650 s wall-clock are _build_grain_atoms and
_rebuild_spatial_index.

CPU-only — no GPU, no student model.  Use this on buffle before any
porting work, to know which sub-step to attack first.

Run:
    python scratch/profile_pack.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

# Make tricor importable when running from repo root.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import tricor as tc                                             # noqa: E402
from tricor import CoordinationShellTarget, G3Distribution      # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CIF_DIR = Path("/home/ehrdt/cifs_mp_exp_le100meV")

# Light → medium → dense.  Si: sp3 covalent low-density.  TiO2 (rutile):
# moderate-density oxide.  Fe2N: dense metal nitride (MACE-OOM at 50³).
CIF_FILES = [
    "mp-149_Si.cif",        # light, single-species sp3 (reference)
    "mp-1439_TiO2.cif",     # medium, two-species oxide
    "mp-21476_Fe2N.cif",    # dense, metal nitride (MACE OOM at 50³)
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

CELL_DIMS = (100.0, 100.0, 400.0)
RNG_SEED  = 2_000_000

BOND_RELAX_N_ITER   = 20
BOND_RELAX_MAX_STEP = 0.1

# Saving phi_num_bins same as production (90).
PHI_NUM_BINS = 90


# ─────────────────────────────────────────────────────────────────────────────
# Preset assembly (mirrors generate_with_student.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_local_presets() -> dict:
    base: dict = {}
    for name in REGIMES:
        if name == "crystalline_30":
            base[name] = dict(
                num_steps=0, grain_size=30.0, displacement_sigma=0.0,
                bond_weight=3.0, angle_weight=1.5,
            )
        else:
            d = dict(tc.Supercell.PRESETS[name])
            d["displacement_sigma"] = 0.0
            base[name] = d
    return base


LOCAL_PRESETS = _build_local_presets()


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch instrumentation
# ─────────────────────────────────────────────────────────────────────────────
#
# We accumulate timings per call.  The profile loop resets the accumulator
# before each regime run, then reads it out.  Patching is done once at
# import time; the original methods stay callable via the saved references
# so unpatched code (e.g. tricor's internal cross-calls) sees normal
# semantics — we only add a timing wrapper.

_TIMERS: dict[str, float] = {}


def _reset_timers() -> None:
    _TIMERS.clear()


def _accum(name: str, dt: float) -> None:
    _TIMERS[name] = _TIMERS.get(name, 0.0) + dt


def _patch(cls, method_name: str, label: str) -> None:
    orig = getattr(cls, method_name)

    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig(self, *args, **kwargs)
        finally:
            _accum(label, time.perf_counter() - t0)

    wrapper.__name__ = orig.__name__
    wrapper.__qualname__ = getattr(orig, "__qualname__", orig.__name__)
    setattr(cls, method_name, wrapper)


# Methods inside Supercell.generate that we want to attribute.
_patch(tc.Supercell, "_build_grain_atoms",    "  _build_grain_atoms")
_patch(tc.Supercell, "_rebuild_spatial_index", "  _rebuild_spatial_index")
_patch(tc.Supercell, "shell_relax",            "  shell_relax")
_patch(tc.Supercell, "_build_random_atoms",    "  _build_random_atoms")

# Module-level helpers inside tricor._grain.  Patch at the source so all
# call-sites (including Supercell._build_grain_atoms) see the timed
# versions.
from tricor import _grain as _tc_grain                          # noqa: E402


def _wrap_module_fn(module, fn_name: str, label: str) -> None:
    orig = getattr(module, fn_name)

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            _accum(label, time.perf_counter() - t0)

    wrapper.__name__ = orig.__name__
    wrapper.__qualname__ = getattr(orig, "__qualname__", orig.__name__)
    setattr(module, fn_name, wrapper)


# Drill-down inside _build_grain_atoms: these are the four pre-loop
# sub-calls (Voronoi + radius + master block + rotations) plus the
# overlap-removal _push_close_pairs_apart.  Whatever wall-clock isn't
# attributed to one of these must be in the per-grain Python loop
# at line 514 of _grain.py.
_wrap_module_fn(_tc_grain, "_periodic_voronoi_3d",
                "    _periodic_voronoi_3d")
_wrap_module_fn(_tc_grain, "_grain_radius_3d",
                "    _grain_radius_3d")
_wrap_module_fn(_tc_grain, "_build_master_atom_block_3d",
                "    _build_master_atom_block_3d")
_wrap_module_fn(_tc_grain, "_random_rotation_matrices",
                "    _random_rotation_matrices")
_wrap_module_fn(_tc_grain, "_push_close_pairs_apart",
                "  _push_close_pairs_apart")
# Fast-path helper added in the same refactor that skips Voronoi.
if hasattr(_tc_grain, "_grain_assign_fast"):
    _wrap_module_fn(_tc_grain, "_grain_assign_fast",
                    "    _grain_assign_fast")


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory profile
# ─────────────────────────────────────────────────────────────────────────────

def profile_one(cif_path: Path, regime: str) -> dict:
    """Return {stage_name: seconds} for a single (CIF, regime) run."""
    _reset_timers()
    preset = dict(LOCAL_PRESETS[regime])
    preset["num_steps"] = 0      # matches production
    rho = DENSITY_BY_REGIME[regime]

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    ref = ase_read(str(cif_path), format="cif")
    timings["ase_read"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    shell = CoordinationShellTarget.from_atoms(ref, phi_num_bins=PHI_NUM_BINS)
    timings["CoordinationShellTarget.from_atoms"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dist = G3Distribution(ref, label=cif_path.stem)
    dist.measure_g3(r_max=10.0, r_step=0.1, phi_num_bins=PHI_NUM_BINS,
                    show_progress=False)
    timings["G3Distribution.measure_g3"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cell = tc.Supercell(
        dist,
        cell_dim_angstroms=tuple(CELL_DIMS),
        relative_density=rho,
        rng_seed=RNG_SEED,
        label=f"{cif_path.stem}_{regime}_{RNG_SEED}",
    )
    timings["Supercell.__init__"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    summary = cell.generate(
        shell, **preset, refine_orientations=False, show_progress=False,
    )
    timings["Supercell.generate (total)"] = time.perf_counter() - t0
    # Add per-sub-step timings collected by the monkey-patches.
    for k, v in _TIMERS.items():
        timings[k] = v

    t0 = time.perf_counter()
    cell.bond_relax(shell, n_iter=BOND_RELAX_N_ITER,
                    max_step=BOND_RELAX_MAX_STEP)
    timings["Supercell.bond_relax"] = time.perf_counter() - t0

    timings["__n_atoms__"] = float(len(cell.atoms))
    return timings


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

# Stage labels in the order we want them displayed.  Sub-stages indented to
# show parent/child relationship.
_DISPLAY_ORDER = [
    "ase_read",
    "CoordinationShellTarget.from_atoms",
    "G3Distribution.measure_g3",
    "Supercell.__init__",
    "  _build_random_atoms",
    "Supercell.generate (total)",
    "  _build_grain_atoms",
    "    _periodic_voronoi_3d",
    "    _grain_radius_3d",
    "    _build_master_atom_block_3d",
    "    _random_rotation_matrices",
    "    _grain_assign_fast",
    "  _rebuild_spatial_index",
    "  _push_close_pairs_apart",
    "  shell_relax",
    "Supercell.bond_relax",
]


def fmt_row(label: str, vals: list[float | None], n_atoms: list[float | None]
            ) -> str:
    cells = []
    for v in vals:
        if v is None:
            cells.append(f"{'—':>8}")
        elif v < 0.05:
            cells.append(f"{v*1000:>7.0f}m")        # ms
        else:
            cells.append(f"{v:>7.1f}s")
    return f"  {label:<38}" + "".join(cells)


def print_table(cif_label: str, per_regime: dict[str, dict]) -> None:
    print()
    print(f"━━━ {cif_label}  cell={CELL_DIMS}  seed={RNG_SEED}  "
          f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    header = "  " + " " * 38 + "".join(f"{r:>8}" for r in REGIMES)
    print(header)
    n_atoms_row = ["—" if r not in per_regime
                   else f"{int(per_regime[r]['__n_atoms__']):>7}"
                   for r in REGIMES]
    print(f"  {'n_atoms':<38}" + "".join(f"{c:>8}" for c in n_atoms_row))
    print("  " + "-" * (38 + 8 * len(REGIMES)))

    for label in _DISPLAY_ORDER:
        vals = []
        for regime in REGIMES:
            t = per_regime.get(regime, {})
            vals.append(t.get(label))
        if all(v is None for v in vals):
            continue
        print(fmt_row(label, vals, []))

    # Totals = sum of the three outer stages (matches generate_with_student.py's
    # pack/cleanup labels).
    pack_total = []
    for regime in REGIMES:
        t = per_regime.get(regime, {})
        pack = sum(t.get(k, 0.0) for k in [
            "ase_read",
            "CoordinationShellTarget.from_atoms",
            "G3Distribution.measure_g3",
            "Supercell.__init__",
            "Supercell.generate (total)",
        ])
        pack_total.append(pack if pack > 0 else None)
    print("  " + "-" * (38 + 8 * len(REGIMES)))
    print(fmt_row("PACK total (generate_with_student t_pack)",
                  pack_total, []))
    bond_relax_row = [per_regime.get(r, {}).get("Supercell.bond_relax")
                      for r in REGIMES]
    print(fmt_row("CLEANUP (Supercell.bond_relax)",
                  bond_relax_row, []))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[profile_pack] CELL={CELL_DIMS}  n_iter={BOND_RELAX_N_ITER}  "
          f"regimes={len(REGIMES)}  cifs={len(CIF_FILES)}")
    print(f"[profile_pack] tricor from: {tc.__file__}")

    for cif_name in CIF_FILES:
        cif_path = CIF_DIR / cif_name
        if not cif_path.is_file():
            print(f"  ✗ missing: {cif_path}")
            continue

        per_regime: dict[str, dict] = {}
        for regime in REGIMES:
            sys.stdout.write(f"  [{cif_name}/{regime}] ... ")
            sys.stdout.flush()
            t0 = time.perf_counter()
            try:
                timings = profile_one(cif_path, regime)
            except Exception as exc:
                dt = time.perf_counter() - t0
                print(f"FAILED ({dt:.1f}s) — {type(exc).__name__}: {exc}")
                continue
            dt = time.perf_counter() - t0
            n = int(timings["__n_atoms__"])
            print(f"{dt:6.1f}s   n_atoms={n}")
            per_regime[regime] = timings

        print_table(cif_name, per_regime)

    print()
    print("[profile_pack] done.")


if __name__ == "__main__":
    main()
