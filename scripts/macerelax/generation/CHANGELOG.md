# macerelax generation — decision & change log

Dated, retrievable record of generation-pipeline decisions. Each entry names
the git tag/commit that captures the relevant code state so any dataset can be
traced back to exactly the code that produced it.

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
