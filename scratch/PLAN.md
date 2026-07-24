# PLAN: ML-accelerated SiO₂ supercell generation — status

**Branch:** `ml-test`
**Original goal:** generate SiO₂ supercells up to 200×200×200 Å in ~1 min, no unphysical atomic overlaps, no full FIRE relaxation at scale.
**Status: shipped.**  See `WIKI.md` for the algorithmic findings.

## What landed

### Code (`src/atomode/ml/`)

- `egnn.py` — equivariant graph neural net (Satorras 2021), ~250 LOC, 23 k params at the default 32-hidden × 3-layer config used by all demos.
- `dataset.py`
  - `AtomodeMLDataset` — one-shot (voronoi, fire_pos) pairs.
  - `AtomodeMLStepDataset` — (x_t, x_{t+stride}) trajectory pairs.
  - `build_pbc_graph_chunked` — cell-list PBC neighbour finder via `scipy.spatial.cKDTree`.  Falls back to chunked O(N²) when SciPy unavailable.
- `inference.py`
  - `load_model` — auto-picks MPS/CUDA/CPU device.
  - `predict_positions` — one-shot.
  - `predict_iteratively` — K-step iterative + optional repulsion projection per step + optional final repulsion sweep.
  - `_enforce_hard_core` — vectorised cKDTree-based pair push.
  - `predict_and_optionally_relax` — wraps the above into `Supercell.generate`'s ML backend.
- `train.py` — train + validate loops, MSE-on-displacement loss with min-image PBC correction (this fix was critical; see WIKI).

### Supercell integration (`src/atomode/supercell.py`)

`generate(backend='ml' | 'ml+fire', ml_model=…, …)` with new kwargs:
- `ml_iterative_steps` — K, default 0 (one-shot mode)
- `ml_iterative_momentum`, `ml_iterative_step_clip` — integrator knobs
- `ml_repulsion_iters_per_step` — projection sweeps after each EGNN step
- `ml_final_repulsion_iters` — one-shot final cleanup
- `ml_fire_cleanup_steps` — K extra FIRE steps after ML
- `ml_chunk_size` — PBC graph builder chunk size

### Tests

- `tests/test_ml_smoke.py` — 7 tests (graph, EGNN forward, equivariance, collate, end-to-end backend='ml').  Always run.
- `tests/test_ml_sio2.py` — 14 acceptance gate tests.  Skip when no `src/atomode/ml/data/sio2/checkpoint.pt` (gitignored — the file used to live there, now lives in `demos/scratch/checkpoint.pt`).  When you want to run the gates, copy the demo checkpoint into the legacy path or rewrite the test to point at the new location.

All 21 ML tests pass on the current branch.

### Demo notebooks (`/Users/cophus/Library/CloudStorage/Dropbox/python/atomode/demos/`)

| nb | what | wallclock |
|---|---|---|
| `01_static_sio2_40A.ipynb` | static + FIRE | ≈ 4 min |
| `02_refined_sio2_40A.ipynb` | refinement + FIRE | ≈ 10 min |
| `03_ml_sio2_200A.ipynb` | iter ML + repulsion proj at 200³ + bonus one-shot comparison | ≈ 1 min/regime production path, +bonus failure-mode demo |

All three use `./scratch/` (relative) for training data + checkpoints.  Repo `.gitignore` blocks `*.pt`, `*.h5`, `*.history.json`, `scratch/`, `src/atomode/ml/data/`.

## Numbers worth knowing (SiO₂ amorphous at 200³ Å)

| backend | time | min_d (Å) | sub-NN count |
|---|---:|---:|---:|
| FIRE truth (60³ Å reference) | 162 s | 1.31 | 0 |
| one-shot ML | 34 s | 0.09 | 19 332 |
| ML+FIRE-3 (Demo 3's legacy fallback) | 163 s | 0.62 | 538 |
| **iter K=10 + rep/step=1** | **65 s** | **1.28** | **0** ← production |
| iter K=10 + rep/step=2 + final 10 | 83 s | 1.45 | 0 |

## Open / future work

In rough priority:

### 1. CPU/CUDA port of `_shell_relax.py` forces

User's planned medium-term project.  Once landed, the `ml+fire` backend gets ~5–10× faster per FIRE step → "FIRE-quality g(r) at 200³ Å" within 1-min budget.  The current iter+rep approach is the right answer for "no overlaps", but for fully-relaxed g(r) the bottleneck is still the python-side FIRE loop.

### 2. Port the repulsion sweep to torch

Currently the only non-torch step in the iter+rep inference loop.  ~3 s/iter at 200³ on CPU; would be sub-second on MPS/CUDA.  Tied with #1 — both want a torch port of the per-pair force math.

### 3. Multi-scale training data

The current model trained on 36 × 20³ Å cells succeeded only with the projection-step trick.  Training on a mix of 20³ + 40³ + 60³ would let the model itself stay in-distribution at scale, possibly removing the need for the projection step.  ~3 hr data gen + retrain.

### 4. Other materials

Pipeline is generic — extend to Si, C (sp²/sp³), Cu, SrTiO₃.  Need:
- Per-material training data generator (regimes + grain_size schedule).
- Tune `pair_hard_min` thresholds (already in `shell.pair_hard_min` for the projection step).
- Multi-material model OR per-material models — equivariance is material-agnostic but conditioning needs work.

### 5. Acceptance-gate tests need a permanent home for the checkpoint

`tests/test_ml_sio2.py` currently expects `src/atomode/ml/data/sio2/checkpoint.pt` (now gitignored).  Either:
- have the test build a tiny checkpoint on demand via a fixture (~30 s in conftest), or
- bake a pre-trained checkpoint into the test data dir via `git lfs` or external download.

### 6. Smarter conditioning

The `grain_size` scalar conditioning is barely used by the current model (the Voronoi-tile pattern already encodes the regime).  For a generation-from-noise variant (no Voronoi seed) this is the main signal — would need to expand `cond_input_dim` to a richer per-cell vector (full target g(r) summary?).

### 7. Benchmark ORB-v3 as a MACE-MP0 alternative

Relevant to the speed story behind this whole project (ML vs FIRE vs MACE).  ORB-v3 (Orbital Materials) is ~accuracy-equivalent to MACE-MPA-0 but markedly faster on GPU, and drops into atomode's ASE-calculator interface with minimal change.  Investigated 2026-06-19, **deferred**.  Full feasibility notes — construction sites to change, ORB API, GPU-vs-CPU and conservative-vs-direct caveats, recommended first benchmark — in `scratch/ORB_V3_INTEGRATION.md`.
