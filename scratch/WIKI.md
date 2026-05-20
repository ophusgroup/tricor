# tricor — Working Knowledge

A compressed handover document. Captures the conceptual model, the
gotchas, the fixes shipped, and the numbers worth knowing. Read this
before touching anything in `src/tricor/` or `tricor-docs/scripts/`.

## What tricor does

Builds **periodic atomic supercells** with controllable disorder,
from a single reference crystal (CIF / ASE Atoms) up to a target box
size. Output is an ASE `Atoms` object plus the measured rooted
three-body distribution `g3(r₀₁, r₀₂, φ)`. Designed as training-data
generation for ML interatomic potentials — the regimes (liquid →
amorphous → SRO → MRO → LRO → NC) span the realistic diversity an
MD trajectory would visit.

Six axes the user can dial:

| axis | knob | typical range |
|---|---|---|
| disorder | `grain_size` | `None` (liquid) → 35 Å (NC) |
| bond stiffness | `bond_weight` | 0.04 (liquid) → 5.0 (Cu NC) |
| angle stiffness | `angle_weight` | 0.0 (Cu, multi-modal) → 1.85 (SiO₂ NC) |
| repulsion stiffness | `repulsion_weight` | 0.45 (Cu liquid) → 2.3 (Si NC) |
| FIRE time budget | `num_steps` | 40 (Cu liquid) → 500 (SrTiO₃ NC) |
| init noise | `displacement_sigma` | 0.001 → 0.12 |

Two-step pipeline:

1. **Voronoi grain construction** (`_grain.py`). Seeds drawn,
   reference crystal tiled into a sphere covering each cell, rotated
   per grain, filtered by convex-hull membership, exact atom counts
   pinned to per-species stoichiometry.
2. **Spring-network relaxation** (`_shell_relax.py`). FIRE-style
   gradient descent on bond + angle + repulsion terms against the
   per-species shell target (`shells.py`). Optional preceding
   **orientation refinement** (`_resample.py`) does SO(3) coordinate
   descent on grain rotations.

## Pipeline anatomy (the order things happen)

```
CIF / ASE Atoms (reference)
    │
    ▼
CoordinationShellTarget.from_atoms(atoms)
    ├── per-species pair_peak / pair_inner / pair_outer / pair_hard_min
    ├── per-triplet angle_target + angle_mode_deg
    ├── coordination_target (auto-filtered for lattice artefacts)
    └── angle_enabled_mask (all True by default)
    │
    ▼
Supercell.from_atoms(atoms, cell_dim_angstroms, ..., rng_seed)
    │   sets up cell, RNG, target_distribution
    ▼
Supercell.generate(shell_target, **gen_kwargs)
    ├── _grain.py  → Voronoi cells, tile reference, place atoms
    │   ├── ``_grain_masters[i]["species_offset"]`` ← virtual species tag
    │   └── exact-count enforcement (rounds species counts to integers)
    ├── (optional) refine_initial_orientations  → SO(3) descent per grain
    └── shell_relax  → FIRE quench
        ├── rebuild_topology: K-nearest bond list with angle + species checks
        ├── force = bond + angle + repulsion + (optional restraint)
        ├── FIRE-inspired: zero velocity on direction reversal
        └── max_force_clip = 2.0, step decay 0.995/iter
    │
    ▼
Supercell.measure_g3(backend="auto")
    └── numba-parallel kernel (~25× faster than numpy reference)
```

## Subtleties that bit us this session

### 1. Auto-filter for lattice-artefact bond pairs

`CoordinationShellTarget.from_atoms` used to install bond springs at
*every* species-pair peak the reference crystal had. For SiO₂ that
meant Si-Si at 3.06 Å and O-O at 2.64 Å (both lattice separations
through a bridging atom) got bond springs alongside the real Si-O
1.61 Å bond. **FIRE could not converge** — each Si had geometrically
incompatible springs pulling it to 4 O at 1.61 Å AND 4 Si at 3.06 Å.

**Fix:** `auto_filter_lattice_artifacts: bool = True` default in
`from_atoms`. Heuristic: pair `(i, j)` is "real" iff `pair_peak[i, j]`
is the smallest enabled peak in *either* row `i` or column `j`.
Lattice artefacts are larger than both sides' minimum and get
`coordination_target = 0`. Catches Si-Si / O-O in SiO₂, Sr-Sr /
Ti-Ti / O-O / Sr-Ti in SrTiO₃. Single-species (Si, Cu) untouched
because `num_species < 2`.

Test: `tests/test_auto_filter.py`.

### 2. Grain interior freezing was the wrong default

`shell_relax` used to freeze grain-interior atoms during FIRE
(`is_boundary is not None` → interior force zeroed). This worked
fine for single-species cells but **broke multi-species** — interior
atoms couldn't accommodate cross-species spring strain that
propagated in from boundaries, and the system plateaued in a
high-energy basin.

**Fix:** `freeze_grain_interiors: bool = False` default. The
pre-2026-05 behaviour is now opt-in.

Two exceptions where we re-enable it via `regen_static_full.py`:

- **Cu** (any regime with `grain_size`) — `angle_weight=0` because
  FCC angles are multi-modal (60/90/120/180°). Without angle springs
  AND without freeze, FCC interior atoms drift during FIRE and the
  cuboctahedral order collapses. Re-enabling freeze fixes it.
- (Si/SiO₂/SrTiO₃/Carbon all run unfrozen — they have angle springs
  active.)

Test: `tests/test_supercell_smoke.py::test_freeze_grain_interiors_default_is_off`.

### 3. Virtual-species clobbering in orientation refinement

`_resample.py::refine_initial_orientations` re-tiles each grain into
a rotated master block during the SO(3) search. The old code
recomputed `species_idx` from atomic numbers via
`searchsorted(self._species, new_nums_world)` — fine for distinct
elements, **catastrophic for composite shell targets** where sp²-C
and sp³-C share atomic number 6. Every retiled atom got tagged as
the first virtual species (sp²_C, index 0), regardless of which
grain it actually belonged to. **Result:** refined sp³_nc cells
were 100% sp²_C, all bonds at 1.42 Å instead of 1.54 Å, zero sp³
tetrahedra detected.

**Fix:** `_grain.py` tags each master with `species_offset` from
its source; `_resample.py` reads `master["species_offset"]` directly
instead of guessing from atomic numbers. The numpy fallback is kept
for grains with no offset stored.

Test: `tests/test_virtual_species.py`.

### 4. Cu g(r) sub-NN spike

Cu uses `angle_weight=0` (cuboctahedral multi-modal) so the only
force keeping atoms apart is the hard-core repulsion wall. Pre-fix
`hard_core_scale = 0.78–0.94` put the wall at 1.95–2.35 Å, well
below the 2.56 Å NN. Atoms compressed onto the wall and the g(r)
peak appeared 0.2–0.6 Å below where it should.

**Fix:** ramp `hard_core_scale` from 0.85 (liquid) → 1.00 (NC) so
the wall sits at the inner edge of the NN shell. **Combined with**
strong bond springs (bw 2.5–5.0) so atoms are actively pulled to the
NN distance rather than parked at the wall. Bumped
`nonbond_push_scale` to 1.0 so non-bonded pairs are pushed past 3.84 Å
into the 1NN/2NN gap.

### 5. SrTiO₃ angle whitelist (manually applied in regen scripts)

After auto-filter, SrTiO₃'s real bonds are Ti-O (1.96 Å) and Sr-O
(2.77 Å). But the Sr-centred O-Sr-O angle distribution is
cuboctahedral (60°/90°/120°/180° simultaneously) — a single-target
angle spring fights all four modes and FIRE thrashes. Workaround:
`shell.with_angle_triplets([('Ti','O','O'), ('O','Ti','Ti')])`
keeps only the genuinely single-peaked Ti-centred 90° and the linear
Ti-O-Ti 180° backbone.

This is **not yet auto-detected** — `regen_static_full.py` and
`regen_refined_full.py` apply the whitelist explicitly in
`build_disorder` when `material == "strontium_titanate"`. The
"auto-detect multi-modal angles" idea is a future improvement.

### 6. Sphinx static-copy issue (build hygiene)

`docs/_static/` files don't always get copied into `docs/_build/html/_static/`
during an incremental Sphinx build. After regenerating any artefact,
either:

```bash
rsync -a docs/_static/ docs/_build/html/_static/
# or
rm -rf docs/_build && sphinx-build -b html docs docs/_build/html
```

`scripts/README.md` documents this gotcha.

## File map

```
src/tricor/
├── __init__.py         exports
├── shells.py           CoordinationShellTarget (auto-filter lives here)
├── supercell.py        Supercell class, generate() orchestrator
├── _grain.py           Voronoi grain construction (species_offset tag)
├── _resample.py        orientation refinement, refine_grains*
├── _shell_relax.py     FIRE quench, thermal_relax wrapper
├── _thermal_mc.py      numba MC kernels (thermal_relax_impl)
├── _monte_carlo.py     Supercell MC mixin + measure_g3 wrapper
├── g3.py               G3Distribution, target_g3, measure_g3 dispatch
├── _g3_numba.py        parallel g3 accumulation kernel
├── _plotting.py        export_*_html, polyhedra detectors
├── g3_widget.py        anywidget g3 viewer (Jupyter)
├── g3_compare_widget.py
├── structure_widget.py
└── static/             *.js, *.css, *.html for widgets

tests/
├── conftest.py             fixtures (atoms_si, atoms_cu, ...)
├── test_auto_filter.py     7 tests
├── test_fire_convergence.py 4 tests
├── test_g3_numba.py        9 tests
├── test_supercell_smoke.py 4 tests
└── test_virtual_species.py 4 tests
        (28 tests total, ~15 s)
```

```
tricor-docs/
├── docs/                 Sphinx source (algorithms/, examples/, etc.)
├── docs/_static/         pre-built HTML artefacts (iframes load these)
└── scripts/              regen scripts (see scripts/README.md)
    ├── regen_static_full.py        ← single source of truth for per-(material, regime) params
    ├── regen_static_overview.py    ← imports from regen_static_full
    ├── regen_refined_full.py       ← imports from regen_static_full
    └── regen_refined_overview.py   ← imports from regen_static_full
```

## Material-specific recipes (for the 5 demo materials)

| material | grains (Å) | weights | hcs / nps | quirks |
|---|---|---|---|---|
| Cu (FCC, 12-coord) | NC=18 | bw 2.5–5.0, **angle=0** | 0.85→1.00 / 1.0 | needs `freeze_grain_interiors=True`; sweet spot in `hard_core_scale` near 1.0 |
| Si (diamond, 4-coord) | NC=12 (preset) | bw 0.6–2.4 / a 0.2–1.0 | 0.86–0.93 / 0.45–0.85 | clean angle springs work |
| SiO₂ (α-quartz) | SRO 15 / MRO 20 / LRO 26 / NC 35 | bw=1.65 a=1.35 across | 0.82 / 0.72 | "FIRE sweet spot": same weights at every grain, only `grain_size` + `num_steps` + `σ` change |
| SrTiO₃ (perovskite) | SRO 14 / MRO 22 / LRO 28 / NC 35 | bw=1.0 a=0.7 | **1.10** / 0.75 | needs `with_angle_triplets([Ti-O-O, O-Ti-Ti])`; `hard_core_scale=1.10` is essential |
| Carbon (sp²/sp³) | 18 Å for all regimes | bw 2.5 a 1.2 | 0.92 / 0.85 | composite shell target via `from_targets({"sp2", "sp3"})`; the SP regime variant changes grain *chemistry* not size |

Polyhedra detector tolerances (`_plotting.py`):

| polyhedron | bond_length_tol | angle_tol_deg | notes |
|---|---|---|---|
| tetrahedra (Si/C/SiO₂) | 0.10 | 18° | Si static uses 18°, carbon uses 15° (stricter; sp²/sp³ would otherwise overlap) |
| octahedra (Ti) | 0.18 | 18° | |
| cuboctahedra (Cu/Sr) | 0.12 | 22° | needs the looser angle tol because 12-coord pairwise angles span 60–180° |
| triangles (sp²-C) | 0.10 | 15° | |

POLY_CAPS (subsample for HTML rendering): tetrahedra 5000, triangles 3500, cuboctahedra 1500, octahedra 280.

## Numba kernel (item #5 just shipped)

`_g3_numba.py` JITs the g3-accumulation inner loop with
`prange` over origin atoms. Per-thread accumulators reduced at the
end (no atomic adds, no GIL). Both backends produce **bit-identical**
`g3count` and `g2count`.

Benchmark (40 Å production cells):

| material | atoms | python | numba | speedup |
|---|---|---|---|---|
| Si MRO | 3068 | 5.5 s | 0.21 s | 26× |
| Cu MRO | 5202 | 17.1 s | 0.92 s | 19× |
| SiO₂ MRO | 4866 | 19.2 s | 0.98 s | 20× |
| SrTiO₃ MRO | 5130 | 28.8 s | 1.16 s | 25× |

`measure_g3(backend="auto" | "numba" | "python")`. JIT compile is
~1–2 s on first call; cached for subsequent calls in the same process.

As of `tricor 0.1.1`, **numba is a hard dependency** (no more `[fast]`
extra). `pip install tricor` is the only command needed.

## Performance numbers (where the wallclock goes)

For the **5 min** regeneration of a 40 Å SrTiO₃ NC cell on 2026
hardware (M-series Mac):

| step | time | what dominates |
|---|---|---|
| `_grain.py` construction | ~2 s | numpy-bound Voronoi convex hull tests |
| `refine_initial_orientations` | 60–120 s | SO(3) trial loop (50 × 4 × 2 = 400 trials × N grains) |
| `shell_relax` FIRE | 150–250 s | `neighbor_list` rebuild every 10 steps + force assembly |
| `measure_g3` | 1.2 s | (numba-parallel, was 28 s pre-acceleration) |
| HTML export | 1–3 s | mostly trajectory.json size for the movie |

So the bottleneck for 40 Å cells is now squarely **FIRE relaxation**
— the orientation refinement is parallelizable per-grain and `measure_g3`
already got 25×. Going to 100×100×500 Å means ~40× more atoms (~400 k
for SiO₂/SrTiO₃) and the existing FIRE pipeline is far too slow.

## The 100×100×500 Å problem (what's next)

Volume 5×10⁶ Å³ vs the current 6.4×10⁴ Å³ (40 Å cubes) — **80× more
atoms**. Naive scaling of the FIRE quench would be 80× × 5 min ≈
6.5 hours per cell, vs the user's <1 min target. **Need ~400×
wallclock speedup** beyond current.

Options live in `scratch/PLAN.md` (next).

## Useful one-liners

```bash
# Full test suite
.venv/bin/python -m pytest tests/

# Regen one (material, regime) pair
python scripts/regen_static_full.py --material silicon_dioxide --regime medium_range_order

# Verify numba bit-identical to python for any cell
python -c "
import tricor as tc
from ase.build import bulk
shell = tc.CoordinationShellTarget.from_atoms(bulk('Si','diamond',a=5.431))
cell = tc.Supercell.from_atoms(bulk('Si','diamond',a=5.431),cell_dim_angstroms=(20,20,20),r_max=6,r_step=0.1,phi_num_bins=24,rng_seed=42)
cell.generate(shell,num_steps=10,bond_weight=2,angle_weight=0.8,repulsion_weight=2.2,hard_core_scale=0.93,nonbond_push_scale=0.75,displacement_sigma=0.04,capture_trajectory=False,show_progress=False)
cell.measure_g3(backend='python',force=True,show_progress=False); a=cell.current_distribution.g3count.copy()
cell.measure_g3(backend='numba',force=True,show_progress=False); b=cell.current_distribution.g3count.copy()
import numpy as np; print('match:', np.array_equal(a, b))
"

# Force-copy regen artefacts into Sphinx build dir
rsync -a tricor-docs/docs/_static/ tricor-docs/docs/_build/html/_static/
```

## Open improvements that didn't ship this session

| # | what | scope |
|---|---|---|
| 1 | Per-pair `bond_weight` / `hard_core_scale` (currently global scalars) | small API change |
| 2 | Auto-detect multi-modal angles (subsume SrTiO₃ whitelist) | heuristic in `shells.py` |
| 3 | Per-pair `nonbond_push_scale` | matches #1 |
| 4 | `CHANGELOG.md` for `tricor` repo | trivial, important for users upgrading |
| 5 | Trusted publishing via GitHub Actions (no API token) | one yaml file |
| 6 | Reference experimental g(r) overlay for liquid/amorphous | docs only |
| 7 | ML acceleration for 100×100×500 Å cells | see ML section below |

---

# ML acceleration (`src/tricor/ml/`) — shipped 2026-05

EGNN-based ML backend for `Supercell.generate`, plus four end-to-end
demo notebooks at `/Users/cophus/Library/CloudStorage/Dropbox/python/tricor/demos/`.
Goal: 200³ Å cells (≈ 600 k atoms) in ~1 min with no unphysical
overlaps.  **Goal met.**

## Module layout

```
src/tricor/ml/
├── __init__.py        public API (EGNN, load_model, predict_*)
├── egnn.py            EGNN layer + model (~250 lines, equivariant)
├── dataset.py         TricorMLDataset (one-shot) +
│                      TricorMLStepDataset (step-pair) + cell-list PBC graph
├── inference.py       load_model, predict_positions, predict_iteratively,
│                      _enforce_hard_core, predict_and_optionally_relax
└── train.py           train loop, validate, MSE-on-displacement loss

tests/
├── test_ml_smoke.py   7 tests: graph, EGNN forward, equivariance, …
└── test_ml_sio2.py   14 tests: regime-by-regime acceptance gates
                      (skip when no /tmp/sio2_ml_demo/checkpoint.pt)
```

## What each notebook does (`demos/`)

| nb | scale | recipe | wallclock | quality |
|---|---|---|---|---|
| 01_static_sio2_40A | 40³ Å | Voronoi → FIRE | ≈ 4 min | gold standard |
| 02_refined_sio2_40A | 40³ Å | + SO(3) refine | ≈ 10 min | best at 40³ |
| 03_ml_sio2_200A | **200³ Å** | iter ML + repulsion proj | **≈ 1 min/regime** | **0 sub-NN bonds** ✓ |

NB 03 trains the step-pair EGNN (production) and (optionally, in a
bonus benchmark cell at the end) a one-shot EGNN to demonstrate the
failure mode that motivated the projection trick.  Both cache to
`demos/scratch/` (relative path).  Repo `.gitignore` blocks `*.pt`,
`*.h5`, `*.history.json`, `scratch/`, `src/tricor/ml/data/`.

## Key algorithmic finding: iterative ML + repulsion projection

**One-shot ML at 200³ fails** (10⁴ overlaps).  **Iter-K ML at 200³ also fails**
without intervention — model drifts into clustered configurations
after K=5 iterations because the training distribution (20³ Å,
600 atoms) doesn't cover the graph statistics at 200³ Å (600 k atoms),
and small per-step errors compound.

Fix that works: **interleave a cheap repulsion projection after each
EGNN step**.  Find atom pairs below `shell.pair_hard_min` via
`scipy.spatial.cKDTree`, push them apart by half the deficit, iterate
2–3× per ML step.  This keeps every intermediate configuration *in*
the model's training distribution → model output stays sensible →
no collapse.

Production config at 200³ Å:

```python
cell.generate(shell, backend='ml',
              ml_model=checkpoint_step_pt,
              ml_iterative_steps=10,
              ml_repulsion_iters_per_step=1,
              ml_final_repulsion_iters=5,
              ...)
```

Measured: amorphous 65 s · MRO 57 s · NC 55 s.  All three regimes:
`sub_NN = 0`, `min_pair_distance > 1.25 Å` (well above SiO₂'s
1.6 Å Si-O target).

## Step-pair training (`TricorMLStepDataset`)

Trained on (x_t, x_{t+1}) pairs sampled at stride=1 from sub-sampled
FIRE trajectories (32 frames/cell × 36 cells = 1116 samples).  Same
EGNN architecture as Demo 3 (32×3, 23 k params), 25 epochs MPS,
~7 min training.  Val loss 0.0056 (vs 0.434 for one-shot — 77×
lower because each prediction is a *small* displacement).

## Inference cost breakdown at 200³ Å (600 k atoms)

- PBC graph (`scipy.cKDTree`, cell-list): **1.7 s/build** ← bottleneck per iter
- EGNN forward (MPS): ~6 s for early iterations, grows if atoms cluster
- Repulsion sweep (cKDTree pairs + vectorised push): ~3 s/iter
- Per ML+rep step: ~10 s.  10 steps + final cleanup: ~70 s.

## Failure modes catalogued

| symptom | cause | mitigation |
|---|---|---|
| `min_d = 0` after many iter steps | atom clustering compounds, model produces nonsense | repulsion projection per step |
| `min_d ≈ box_size/2` outliers in (input, target) | PBC wrap during FIRE | min-image correction in `displacement_loss` |
| h5py BlockingIOError | NB 03 + NB 04 kernels share file | sentinel-based readiness check (no h5 read) |
| `@interact` clips g3 widget | Output wrapper height | `regime_picker` helper using Dropdown + Output |
| 200³ `min_pair_distance` OOM | `(N, N, 3)` allocation in benchmark stats | cKDTree-based fast_pair_stats |

## CPU/CUDA port (next, when you're ready)

The repulsion sweep is the only non-torch part of the inference
loop.  Port that to torch (cell-list neighbour finder on GPU, then
vectorised pair-push) and the whole iter+rep path becomes
end-to-end GPU-resident → 2-5× faster.

The bigger lever: port `_shell_relax.py`'s force computation
(bond + angle + repulsion) to torch on MPS/CUDA.  Same FIRE
integrator, ~5-10× faster per step.  Lets the user blend
`ml_iterative_steps` and `ml_fire_cleanup_steps` for any
quality/speed point.  See `scratch/PLAN.md`.
