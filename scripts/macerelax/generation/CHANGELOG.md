# macerelax generation — decision & change log

Dated, retrievable record of generation-pipeline decisions. Each entry names
the git tag/commit that captures the relevant code state so any dataset can be
traced back to exactly the code that produced it.

---

## 2026-09-04 — Packing and relaxation overhaul; harmonic wall becomes the default

### What

**1. Hard-core floors are pair-resolved.** Overlap removal and the pre-relax
push used a single scalar cutoff, `_dup_frac * min(pair_hard_min)`. For SiO2
that is 1.3275 A applied to every pair, so an O-O contact anywhere in
1.33-2.18 A was never examined against the 2.425 A O-O floor. Both steps now
use the full `pair_hard_min` matrix. This applies to every grain build, not
only when `protect_crystallites=True`.

On the four `Supercell.PRESETS` (SiO2 mp-10851, 24 A, seed 42, 972 atoms) the
sub-floor pair count goes 1678 -> 1285. `liquid` is bit-identical (no grain
path). `SRO` and `MRO` improve on both count and worst contact. `amorphous`
trades a 28% drop in violations for one contact 0.046 A tighter (1.3338 ->
1.2881 A): removing more overlaps leaves a larger shortfall, which padding
refills at a relaxed floor, recorded in `atoms.info["padding_report"]`.

Exact per-species atom counts are now best-effort — the loose-placement
fallback that guaranteed them is replaced by that floor back-off.

**2. `protect_crystallites=True` (new, opt-in).** `relative_density` then
describes the amorphous matrix only; crystalline grains stay at full crystal
density and are frozen through the quench. Without it the global trim punches
vacancies into the grains: at 10 A grains with `relative_density=0.78` the
grain came out at 85.6% of crystal density, CN(Al-O) 4.785 against 6.000.

**3. Wall defaults: quartic k=1000 -> harmonic k=50**, `exponent` 4 -> 2,
`margin` 0.0 -> 0.1. Not a softening: at 0.1 A penetration the quartic gave
4 eV/A where harmonic k=50 gives 10. The reason is conditioning — quartic
curvature grows as delta^2, harmonic is constant at 2k. Generators that pass
`WALL_K`/`WALL_EXPONENT` explicitly still override the class default.

`MinDistanceWallCalculator.calculate` also now raises on non-finite positions
rather than letting `ase.neighbor_list` cast inf->int and host-OOM the worker.

**4. `tricor.crn` (new package).** Continuous random networks by
Wooten-Winer-Weaire bond switching under a Keating potential — the topological
route to an amorphous network, complementary to the Voronoi packing above.
Tetravalent cations only; AX2 networks are built as a cation-only CRN and then
decorated with bridging anions.

`t_max` is an energy, so its useful range is a material property: a silicon
switch costs ~1.3 eV and acceptance is 0% at 0.25 eV. Use `calibrate_t_melt()`
to measure the network's own melting scale and drive the anneal with the
dimensionless ratio. On 216-atom diamond Si, `t_melt` = 1.425 eV: at ratio 0.6
the network stays crystalline, 1.0 gives a CRN (1.9% of switches accepted,
E/atom 0.51, z=4 for 100% of atoms), and 1.5 starts collapsing geometry
(min distance 1.052 A) while coordination still reads as perfect.

### Verification

`pytest tests/`: 30 passed / 9 skipped on `upstream/dev` and on this branch.
`examples/generate_end_to_end.py` runs both pathways, with and without the
MACE leg.

---

## 2026-06-29 — MRO regime relative_density was an inversion (0.88 → 0.92)

### What

Across every macerelax-family generator, `DENSITY_BY_REGIME["MRO"] = 0.88` —
the **lowest** density of any regime. This is physically backwards. Density in
this pipeline tracks structural order via `grain_size`, and that ladder is
monotonic:

| regime | grain_size | rel_density (buggy) | should be |
|---|---|---|---|
| crystalline_30 | 30 | 0.98 | 0.98 |
| nanocrystalline | 20 | 0.96 | 0.96 |
| LRO | 18 | 0.92 | 0.92 |
| **MRO** | **13** | **0.88** ⚠️ | **0.92** |
| SRO | 10 | 0.92 | 0.92 |
| amorphous | 6 | 0.92 | 0.92 |

MRO has *larger* grains / more order than SRO (10) and amorphous (6), yet was
assigned a *lower* density than both. The other three disordered regimes are a
flat 0.92; MRO is the lone outlier. Diagnosed as a typo — intended value 0.92.

### Affected datasets (all carry MRO @ 0.88)

| dataset | path | type | generator / SOURCE_TAG | scale |
|---|---|---|---|---|
| big_exp | `/pscratch/sd/e/ehrdt/tricor/big_exp` | MACE traj (training) | `safemem_generate_mace_perl.py` / `mace_big_v1` | 6,418 |
| cnos_1e100meV | `/pscratch/sd/e/ehrdt/tricor/cnos_1e100meV` | MACE traj, 2 seeds/regime | `mace_cnos` | ~527 |
| generated_cnos_v1 | `/pscratch/sd/e/ehrdt/macerelax/generated_cnos_v1` | student-model XYZ | `generate_with_student*.py` | 2,732 |
| big_extras_quarantine | `/pscratch/sd/e/ehrdt/tricor/big_extras_quarantine` | MACE traj (from big_exp) | `safemem_*` | 151 |

Source files containing the buggy literal (`"MRO": 0.88`):
`safemem_generate_mace_perl.py`, `generate_mace_trajectories.py`,
`generate_mace_trajectories_perlmutter.py`, `generate_with_student.py`,
`generate_with_student_prod.py`, `generate_with_student_test.py`,
`diagnose_generate.py`.

The relaxml-family scripts (`scripts/relaxml/*`) use a different grain-size +
quota scheme and are **not** affected by this literal.

### Retrievable code states

- **Pre-fix (as-built, MRO @ 0.88):** tag **`big_exp-asbuilt`** on the commit
  that first tracks `safemem_generate_mace_perl.py`. `git checkout big_exp-asbuilt`
  reproduces the exact generator that produced `big_exp` (and matches the
  density table used by the cnos runs). The buggy 0.88 lives there intact.
- **Fix:** the commit adding `generate_mro_v2_perl.py` (search log for
  "MRO-fix generator fork").

### Fix applied

- **big_exp:** corrected MRO regenerated at 0.92 into a separate dataset
  `big_exp_mro_v2/` via `generate_mro_v2_perl.py` (idx=5, same seed/cell, only
  density changes). `big_exp`'s idx=5 @0.88 is **superseded**. See
  `big_exp/PROVENANCE.md` for the canonical assembly rule.
### Remediation policy (decided 2026-06-29)

- **No source-literal edits.** Every existing generator is left as-built with
  `MRO = 0.88`. The as-built code state is the reproducible record (tag
  `big_exp-asbuilt`); editing the literal in place would muddy that. All MRO
  corrections are applied *only* through dedicated `*_mro_v2` forks per dataset
  (each a config-only fork: `REGIME_STRATA=["MRO"]`, density 0.92, idx=5, same
  seed/cell, separate output root). This also keeps the big_exp *tail* re-run
  consistent with its bulk (still 0.88), with MRO corrected uniformly afterward.

### Open / deliberately deferred

- **`cnos_1e100meV`** MACE trajectories: documented, not corrected. The same
  mro_v2 fork pattern applies when picked up — point the fork's `CIF_DIR` /
  `DATASET_ROOT` / `CALIBRATION_LOAD_PATH` at the cnos set (note: 2 seeds per
  regime, so idx 5 and 11).
- **`generated_cnos_v1`** student-model output: density feeds the model's
  `weight_vector`, AND the student was trained on `big_exp` (also 0.88). The
  bug is self-consistent train↔inference, so correcting the conditioning alone
  would create a mismatch — a clean fix needs retraining on corrected data.
  Deferred pending a retrain decision.
