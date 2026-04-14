"""Train the unconditional flow matching model.

No spectral labels needed — just a directory of structure files.
Guidance is applied at inference time, not during training.

Edit the CONFIG section below, then run:
    python train_flowmatch_uncond.py
"""

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from tricor.flowmatch.velocity_model_uncond import LitUncondFlowMatch
from tricor.flowmatch.data_uncond import UncondFlowMatchDataModule

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

# Data — directory of xyz/extxyz/vasp/cif files
DATA_DIR = "/pscratch/sd/e/ehrdt/mcstructgen/smallcell/"
SPECIES = [7, 14]                       # atomic numbers (N, Si for Si3N4)
CUTOFF = 5.0                            # graph construction cutoff (A)
DUP = 4                               # noise replicas per structure
USE_OT = True                           # per-species optimal transport
VAL_FRACTION = 0.1

# Model (GLASS-matched defaults)
NUM_CONVS = 3                           # message-passing layers
DIM = 128                               # hidden dimension
EMA_DECAY = 0.9999
LR = 1e-3

# Training
MAX_EPOCHS = 10
BATCH_SIZE = 4
NUM_WORKERS = 0
GPUS = 1
LOG_DIR = "./lightning_logs"
RUN_NAME = "flowmatch-uncond-si3n4"
RESUME_CKPT = None

# ══════════════════════════════════════════════════════════════════════════════


def main():
    num_species = len(SPECIES)
    print(f"Training unconditional flow matching model")
    print(f"  Data: {DATA_DIR}")
    print(f"  Species: {SPECIES} ({num_species} types)")
    print(f"  Model: dim={DIM}, num_convs={NUM_CONVS}")
    print(f"  Training: lr={LR}, batch_size={BATCH_SIZE}, max_epochs={MAX_EPOCHS}")

    datamodule = UncondFlowMatchDataModule(
        structures=DATA_DIR,
        cutoff=CUTOFF,
        species=SPECIES,
        dup=DUP,
        use_ot=USE_OT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_fraction=VAL_FRACTION,
    )

    model = LitUncondFlowMatch(
        num_species=num_species,
        num_convs=NUM_CONVS,
        dim=DIM,
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
