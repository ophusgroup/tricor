"""Train the conditional flow matching model on a directory of structure files.

Edit the CONFIG section below, then run:
    python train_flowmatch.py
"""

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from tricor.flowmatch import LitFlowMatch, FlowMatchDataModule

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

# Data
DATA_DIR = "/pscratch/sd/e/ehrdt/mcstructgen/smallcell/"   # directory of xyz/extxyz/vasp/cif files
SPECIES = [7, 14]                       # atomic numbers (N, Si for Si3N4)
CUTOFF = 5.0                            # graph construction cutoff (A)
DUP = 128                               # noise replicas per structure
USE_OT = True                           # per-species optimal transport assignment
VAL_FRACTION = 0.1

# Spectral labels (must match what you use at inference)
R_MAX = 10.0                            # PDF cutoff (A)
R_STEP = 0.05                           # radial bin width (A)
PHI_NUM_BINS = 90                       # angular bins
SIGMA_R = 0.15                          # PDF Gaussian bandwidth (A)
SIGMA_PHI = 0.1                         # ADF Gaussian bandwidth (rad)

# Model
NUM_CONVS = 6                           # message-passing layers
DIM = 256                               # hidden dimension
COND_DIM = 128                          # conditioning vector dimension
EMA_DECAY = 0.9999
LR = 5e-4

# Training
MAX_EPOCHS = 100
BATCH_SIZE = 16
NUM_WORKERS = 0                          # 0 avoids pickle errors with Python 3.14
GPUS = 1
LOG_DIR = "./lightning_logs"
RUN_NAME = "flowmatch-si3n4"
RESUME_CKPT = None

# ══════════════════════════════════════════════════════════════════════════════


def main():
    num_species = len(SPECIES)
    num_r = int(round(R_MAX / R_STEP))

    print(f"Training conditional flow matching model")
    print(f"  Data: {DATA_DIR}")
    print(f"  Species: {SPECIES} ({num_species} types)")
    print(f"  Model: dim={DIM}, cond_dim={COND_DIM}, num_convs={NUM_CONVS}")
    print(f"  Training: lr={LR}, batch_size={BATCH_SIZE}, max_epochs={MAX_EPOCHS}")
    print(f"  Spectral: r_max={R_MAX}, r_step={R_STEP}, phi_bins={PHI_NUM_BINS}")

    datamodule = FlowMatchDataModule(
        structures=DATA_DIR,
        cutoff=CUTOFF,
        r_max=R_MAX,
        r_step=R_STEP,
        phi_num_bins=PHI_NUM_BINS,
        sigma_r=SIGMA_R,
        sigma_phi=SIGMA_PHI,
        species=SPECIES,
        dup=DUP,
        use_ot=USE_OT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_fraction=VAL_FRACTION,
    )

    model = LitFlowMatch(
        num_species=num_species,
        num_convs=NUM_CONVS,
        dim=DIM,
        cond_dim=COND_DIM,
        num_r=num_r,
        num_phi=PHI_NUM_BINS,
        ema_decay=EMA_DECAY,
        learn_rate=LR,
    )

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="{epoch}-{val_loss:.4f}",
    )

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if GPUS > 0 else "cpu",
        devices=GPUS if GPUS > 0 else "auto",
        logger=TensorBoardLogger(save_dir=LOG_DIR, name=RUN_NAME),
        callbacks=[checkpoint_cb, TQDMProgressBar(refresh_rate=10)],
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule, ckpt_path=RESUME_CKPT)
    print(f"Training complete. Best checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
