# Perlmutter setup for MACE+wall trajectory generation

End-to-end checklist for getting `generate_mace_trajectories_perlmutter.py`
running on NERSC Perlmutter. Assumes you already have a Perlmutter account,
SSH access set up, and the repo checked out somewhere under `$HOME`.

## 1. Conda environment

Two paths.  Pick the one that matches your starting state.

### 1a. Extend an existing `tricor` env (recommended if it exists)

A workstation `tricor` env probed today has every relaxml dep already
installed — torch 2.11.0+cu130, numpy, scipy, ase 3.28, tricor 0.1.0 —
so the macerelax delta is just two pip installs:

```bash
module load conda
conda activate tricor             # whatever you named yours

# Preview what pip wants (catches torch-version conflicts before they happen).
pip install --dry-run mace-torch cuequivariance cuequivariance-torch

# If the dry-run looks clean (no torch bump, no other big deps changing):
pip install mace-torch cuequivariance cuequivariance-torch
```

**If pip wants to bump `torch`**: the safest path is to accept the bump,
then verify relaxml still imports cleanly:

```bash
python -c "from tricor.relaxml.model_shelltgt import LitRelaxML; print('relaxml still works')"
```

If relaxml's import breaks after the bump, force pinning:

```bash
pip install --no-deps mace-torch cuequivariance cuequivariance-torch
pip install e3nn matscipy prettytable      # MACE's hard deps we'd otherwise miss
```

### 1b. Fresh `macerelax_gen` env (if no tricor env on Perlmutter)

```bash
module load conda
conda create -n macerelax_gen python=3.12 -y
conda activate macerelax_gen

# PyTorch first — use the cu12-tagged build matched to Perlmutter's
# cudatoolkit/12.x module.
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# MACE + cuequivariance.
pip install mace-torch cuequivariance cuequivariance-torch

# Tricor (editable so script imports work).
cd $HOME/macerelax     # or wherever the repo lives
pip install -e .
```

### Verification (either path)

```bash
python -c "
import torch, mace, tricor, ase, cuequivariance, cuequivariance_torch
from mace.calculators import mace_mp
print('torch          :', torch.__version__)
print('mace           :', mace.__version__)
print('tricor         :', tricor.__version__)
print('ase            :', ase.__version__)
print('cuequivariance :', cuequivariance.__version__)
print('cuda available :', torch.cuda.is_available())
print('gpu count      :', torch.cuda.device_count())
# Smoke MACE-MP cold-start (~30s first time, downloads checkpoint).
calc = mace_mp(model='medium-mpa-0', device='cuda', default_dtype='float32')
print('mace_mp ready  : OK')
"
```

If everything prints + the MACE cold-start finishes without a stack trace,
the env is good to go.  Whichever env name you ended up with, update the
sbatch's `conda activate <env_name>` line to match.

If anything red-lines, NERSC's PyTorch docs are at
<https://docs.nersc.gov/machinelearning/pytorch/> — their curated PyTorch
module is an alternative to manual pip install.

If anything red-lines: NERSC's docs at
<https://docs.nersc.gov/machinelearning/pytorch/> have the blessed PyTorch
install path; preferring NERSC's curated module over a manual pip install
is fine, the script doesn't care which way it got installed.

## 2. CIF library

Two cases.

**Case A: the relaxml CIF lib is already on Perlmutter.** Find it (`find
$CFS/<project>/ -name 'mp-149_Si.cif' 2>/dev/null` is one way). Once
located, point `CIF_DIR` inside `generate_mace_trajectories_perlmutter.py`
at it. Done.

**Case B: needs transfer from the workstation.** Copy the 520-CIF training
symlink dir (~5 MB resolved, ~600 KB compressed) into `$SCRATCH`:

```bash
# From the workstation, push to Perlmutter:
rsync -avzh --copy-links \
    /wigeon/users/ehrdt/prod/cifs_mp_cnos_le100meV_training/ \
    USERNAME@perlmutter.nersc.gov:$SCRATCH/macerelax/cifs/
# --copy-links resolves the symlinks so Perlmutter doesn't see broken refs
# pointing at the workstation's filesystem.
```

Then on Perlmutter:

```bash
ls $SCRATCH/macerelax/cifs/ | wc -l       # expect ~520
```

Set `CIF_DIR = Path("$SCRATCH/macerelax/cifs")` inside the script.
**Note: `$SCRATCH` doesn't auto-expand inside Python paths** — write the
absolute path (typically `/pscratch/sd/<letter>/<user>/macerelax/cifs`).

## 3. Output paths

Inside `generate_mace_trajectories_perlmutter.py`, set:

```python
DATASET_ROOT = Path("/pscratch/sd/<letter>/<user>/macerelax/big_v1")
```

`$SCRATCH` is the right place during generation — fast I/O, big quota.
**Don't use `$CFS` for the live dataset**: it's tuned for archival, not for
the 10k×12MB write pattern during generation. After the run completes,
`cp -r` (or `tar | gzip`) to `$CFS` for permanence.

## 4. Account placeholder in the sbatch

Open `scripts/macerelax/generation/perlmutter_generate.sbatch` and replace
`<FILL_ME>` with your NERSC project ID (the `mNNNN` thing). For example:

```bash
#SBATCH --account=m1234_g
```

(The `_g` suffix is conventional for the GPU sub-allocation.)

## 5. Smoke pre-flight on Perlmutter

Before the multi-node job, validate the env + paths on a single GPU with
a 5-CIF interactive session:

```bash
# Grab an interactive GPU node (single A100, ~15 min):
salloc -A m1234_g -C gpu -q interactive -t 0:15:00 --gpus=1

# Once allocated:
module load conda && conda activate macerelax_gen
cd $HOME/macerelax

# Quick override: set MAX_CIFS=5 either by editing the script or by
# exporting an env var the script reads (currently it's a hardcoded
# constant; just edit the line).  Then:
python scripts/macerelax/generation/generate_mace_trajectories_perlmutter.py
```

Expected outcome: a `manifest_all_pg0.csv`, a `calibrated_cells.csv`
(probably 5 rows, mostly CELL_SIZE fits), and 5 per-CIF dirs with their
own manifest + ~20 NPZs each. Wall time ≈ 10-30 min depending on CIF
density.

If the smoke run looks right, revert `MAX_CIFS = None` and submit the
big job.

## 6. Multi-node job

```bash
cd $HOME/macerelax
sbatch scripts/macerelax/generation/perlmutter_generate.sbatch
squeue --me                       # watch
```

Quick checks while running:

```bash
# Live tail of all node logs:
tail -F logs/slurm-*.out scripts/macerelax/generation/logs/*.log

# How many trajectories on disk so far?
find /pscratch/sd/<letter>/<user>/macerelax/big_v1 -name '*.npz' | wc -l

# Calibration outliers — CIFs that have been auto-shrunk:
awk -F, '$6 != "" && $6 != "50.00"' \
    /pscratch/sd/<letter>/<user>/macerelax/big_v1/calibrated_cells.csv

# OOMs that hit MIN_CELL (rare; would be empty calibrated_cell column):
awk -F, '$6 == ""' \
    /pscratch/sd/<letter>/<user>/macerelax/big_v1/calibrated_cells.csv
```

## 7. Resuming across walltime caps

Perlmutter regular queue caps at 24 h. The script is fully resumable —
re-submitting picks up where the previous job stopped:

- Existing NPZs are skipped (per-trajectory `outfile.is_file()` check).
- CIFs with a complete per-CIF manifest are fully skipped (avoids the
  ~20s cache-build cost).
- Calibration cache is consulted, so previously-shrunk CIFs start at the
  known-good cell size, no re-probe.

Chain jobs with `--dependency=afterany:<prev_jobid>`:

```bash
JOB1=$(sbatch --parsable perlmutter_generate.sbatch)
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 perlmutter_generate.sbatch)
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 perlmutter_generate.sbatch)
# etc.; the chain breaks gracefully if everything's done.
```

A 4-node × 4-GPU job at ~3 min/trajectory does ~7700 trajectories per 24-h
window, so 2-3 chained jobs is enough for the full 10,000-trajectory set.

## 8. After the run completes

Archive to `$CFS`:

```bash
PROJECT=/global/cfs/projectdirs/<project>/<user>/macerelax
mkdir -p $PROJECT
rsync -avzh /pscratch/sd/<letter>/<user>/macerelax/big_v1/ $PROJECT/big_v1/
```

Or tarball:

```bash
tar -czf $PROJECT/big_v1.tar.gz -C /pscratch/sd/<letter>/<user>/macerelax big_v1
```

## 9. Common failure modes

- **"out of memory" but the script keeps going**: working as designed —
  the OOM-driven calibration shrinks the offending CIF and retries.
- **"out of memory" hits MIN_CELL**: CIF is too dense for the GPU even at
  CELL=40. Logged to `failures.csv` with stage `oom_below_min_cell`;
  script moves on. The CIF is excluded from the dataset.
- **Job dies at 24h walltime**: expected on regular queue. Resubmit; the
  resume logic picks up where it stopped.
- **PyTorch import fails with cuda mismatch**: probably mismatched CUDA
  module + PyTorch wheel. NERSC recommends matching the cu-tagged wheel
  to the `cudatoolkit` module version. See their PyTorch docs.
- **Manifest CSVs scattered with `_pn0_g0`, `_pn1_g0`, … suffixes**:
  expected. Each (node, GPU) writes its own manifest. Merge after the
  run completes by concatenating CSVs (taking one header) — or just have
  `make_experiment_manifests.py` walk `*.npz` directly and ignore the
  partial manifests (which it already does).
