"""Train the GLASS score model on a directory of structure files.

Edit the CONFIG section below, then run:
    python train_score_model.py
"""

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from tricor.glass import LitScoreNet, StructureDataModule

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

# Data
DATA_DIR = "/path/to/25A_xyz_files/"   # directory of xyz/extxyz/vasp/cif files
SPECIES = [7, 14]                       # atomic numbers (N, Si for Si3N4)
CUTOFF = 5.0                            # neighbor cutoff for graph construction (A)
K = 0.8                                 # VE-SDE max noise level (A)
DUP = 128                               # noise replicas per structure
VAL_FRACTION = 0.1                      # fraction of structures for validation

# Model (GLASS defaults from Sec. S1.4)
NUM_CONVS = 5                           # message-passing layers
DIM = 200                               # hidden dimension
EMA_DECAY = 0.9999                      # EMA decay rate
LR = 1e-3                               # learning rate

# Training
MAX_EPOCHS = 12000
BATCH_SIZE = 32
NUM_WORKERS = 8
GPUS = 1
LOG_DIR = "./lightning_logs"
RUN_NAME = "glass-si3n4"
RESUME_CKPT = None                      # set to checkpoint path to resume

# ══════════════════════════════════════════════════════════════════════════════


def main():
    num_species = len(SPECIES)
    print(f"Training GLASS score model")
    print(f"  Data: {DATA_DIR}")
    print(f"  Species: {SPECIES} ({num_species} types)")
    print(f"  Model: dim={DIM}, num_convs={NUM_CONVS}")
    print(f"  Training: lr={LR}, batch_size={BATCH_SIZE}, max_epochs={MAX_EPOCHS}, k={K}")

    datamodule = StructureDataModule(
        structures=DATA_DIR,
        cutoff=CUTOFF,
        k=K,
        dup=DUP,
        species=SPECIES,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_fraction=VAL_FRACTION,
    )

    score_net = LitScoreNet(
        num_species=num_species,
        num_convs=NUM_CONVS,
        dim=DIM,
        ema_decay=EMA_DECAY,
        learn_rate=LR,
    )

    print(f"  Parameters: {sum(p.numel() for p in score_net.parameters()):,}")

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

    trainer.fit(score_net, datamodule, ckpt_path=RESUME_CKPT)
    print(f"Training complete. Best checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
