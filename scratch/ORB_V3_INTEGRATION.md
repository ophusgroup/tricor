# Adding ORB-v3 as an alternative to MACE-MP0 — feasibility notes

**Date:** 2026-06-19. **Status: investigated, deferred (not started).**
Goal when revived: let tricor relax / calibrate against ORB-v3
(Orbital Materials) as a drop-in alternative to MACE-MP0, primarily to
**benchmark speed** (ORB and MACE-MPA-0 are ~accuracy-equivalent).

Paper: *Orb-v3: atomistic simulation at scale*, Rhodes et al. 2025,
arXiv:2504.06231. Code: https://github.com/orbital-materials/orb-models
(Apache-2.0).

---

## Verdict: easy. Physics code is untouched; only the calculator-construction sites change.

Every MACE use in tricor goes through the **standard ASE interface**
(`atoms.calc`, `get_potential_energy()`, `get_forces()`). ORB-v3 ships
exactly that (`ORBCalculator`). So the relaxation loops, the
calibration math, and the soft-wall wrapper are all backend-agnostic
already. Only the spots that *build* the calculator are MACE-specific.

### MACE construction sites (the only things to change)

1. **Library** — `src/tricor/_mace_calibrate.py::_load_mace()` (~L61-69),
   called once at `calibrate_to_mace` (~L135). One-line `mace_mp(...)`.
2. **Regen script** — `tricor-docs/scripts/regen_mace_examples.py::_load_mace_calc()`
   (~L277-291), module-level cached calc. Reused at L703 (wall), L754
   (per-stage SP), L768 (orient scoring).
3. **Generated reproducers** — direct `mace_mp(...)` in each
   `docs/_static/mace/<mat>/<regime>_generate.py` (cosmetic; from the
   template in the regen script).

### Already generic — DO NOT touch

- Relaxation loop (`regen_mace_examples.py` ~L693-745): LBFGS / Langevin
  on `atoms.get_potential_energy()` / `get_forces()`.
- Calibration (`_mace_calibrate.py`): FD-Hessian from forces + energy
  scans for Morse fit. Uses only energy/forces, no MACE internals.
- Soft wall (`tricor-docs/scripts/_wall_calculator.py::MinDistanceWallCalculator`):
  generic ASE wrapper around `base_calc` — works with any calculator.
- `mace-torch` is an **optional** dep (guarded import, helpful error).
  Add `orb-models` the same way.

---

## API difference: ORB loader is 3 lines + returns a tuple

```python
# MACE today (1 line):
from mace.calculators import mace_mp
calc = mace_mp(model="medium-mpa-0", device="cpu", default_dtype="float32")

# ORB-v3 equivalent (3 lines, tuple return):
from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.calculator import ORBCalculator
orbff, adapter = pretrained.orb_v3_conservative_inf_omat(
    device="cpu", precision="float32-high")
calc = ORBCalculator(orbff, atoms_adapter=adapter, device="cpu")
```

- `device="cpu"` or `"cuda"`. `precision` ∈ {`float32-high` (default),
  `float32-highest`, `float64`}. ORB README: *"we have not found any
  benefit to using float64."*
- `pip install orb-models`. Graph construction uses `pynanoflann` (CPU)
  / `cuml` (GPU) — **cuML is GPU-only**; verify the CPU neighbor backend
  installs cleanly.
- Model loaders are per-model functions named `orb_v3_{X}_{Y}_{Z}`:
  X ∈ {direct, conservative}, Y ∈ {20, inf} (neighbor cap),
  Z ∈ {omat, mpa}. See MODELS.md in the repo.

---

## Three caveats that matter for the speed test

**1. The 10× is a GPU story; tricor runs MACE on CPU.** All paper
numbers are NVIDIA H200. At 1k atoms (Table 1, steps/s):

| model | steps/s | vs MACE |
|---|--:|--:|
| MACE-MPA-0 | 21.2 | 1× |
| orb-v3-direct-20-mpa | 216.5 | ~10× |
| orb-v3-direct-inf-mpa | 125.0 | ~6× |
| orb-v3-conservative-20-mpa | 41.2 | ~2× |
| orb-v3-conservative-inf-mpa | 28.1 | ~1.3× |

ORB's win comes from non-conservative forces (no backward pass) + sparse
graph + **GPU** graph construction (cuML). On CPU the advantage shrinks
and is unknown — direct models should still beat MACE (skip gradient +
20-neighbor graph) but not by 10×. **Benchmark on a GPU** for a
meaningful comparison; the current `device="cpu"` default will
understate ORB.

**2. Direct (non-conservative) models break calibration, not
relaxation.** `calibrate_to_mace` fits Morse from an *energy* scan AND
stiffness from a *force* FD-Hessian. For direct models forces ≠ −∇E, so
those are mutually inconsistent. **Use `orb-v3-conservative-*` for
calibration.** For plain relaxation either works, but non-conservative
forces make the LBFGS minimum ill-defined — conservative is the safer
apples-to-apples vs MACE.

**3. Accuracy parity holds (slight ORB edge for conservative).**
Matbench F1: conservative-inf 0.906 vs MACE-MPA-0 0.852. Physical-
property MAE (Table 2): `orb-v3-conservative-inf-omat` best across the
board, beating both MACE-MPA-0 and MACE-OMAT-0. Direct slightly worse
but competitive. So "roughly same accuracy" is fair.

### Models to test
- **Calibration + accuracy:** `orb_v3_conservative_inf_omat` (energy-
  consistent, most accurate; direct MACE-MPA-0 replacement).
- **Speed ceiling:** `orb_v3_direct_20_omat` for relax/MD only (not
  calibration), on GPU.
- Prefer `-omat` (OMat24) over `-mpa` (MPtraj+Alexandria; only for
  Matbench-Discovery compatibility).

---

## Implementation sketch (when revived)

1. **Library:** generalize `_load_mace()` → a small backend factory, OR
   have `calibrate_to_*` accept a ready-made `calculator=`. Add
   `backend="mace"|"orb"` (or a generic `calibrate_to_mlip`). Default
   stays MACE.
2. **Regen:** branch `_load_mace_calc()` on an env var / flag to return
   a MACE or ORB ASE calc. Wall wrapper + LBFGS/MD loops unchanged.
3. **Reproducer template:** parameterize the `mace_mp(...)` line
   (optional).

**Recommended first step: NOT the integration.** Write a standalone
head-to-head benchmark (one cleaned structure → MACE vs ORB-conservative
vs ORB-direct: energy, max force, wall-clock, on the target hardware).
It answers the actual question (speed) and de-risks the refactor before
any library change.
