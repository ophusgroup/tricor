"""Load a trained macerelax checkpoint, run one batch through the model,
and report whether the model is outputting:
  (a) zero (collapsed),
  (b) right magnitude but random direction (rotation-augmentation / receptive-field problem),
  (c) right magnitude and aligned with target (learning genuine structure).

Outputs cosine similarity, pred / target norms, and the inner-product
breakdown of train_relative_loss so we can see exactly why rel sits at
~1 instead of descending below.

Run:
    /home/ehrdt/miniforge3/envs/tricor/bin/python scratch/diagnose_pred_target.py
"""
from __future__ import annotations

import os

# === CONFIG ============================================================
GPU_ID = 0   # GPUs 0 + 1 currently free; pick whichever isn't busy when you run
CKPT_PATH = (
    "/home/ehrdt/tricor/scripts/macerelax/lightning_logs/"
    "composition_test_stride5/version_0/checkpoints/last.ckpt"
)
TRAIN_MANIFEST = (
    "/home/ehrdt/tricor/mace/data/pilot_v1/manifests/composition_test_train.csv"
)
K_STRIDE_SNAPSHOTS = 5
CUTOFF = 5.0
BATCH_SIZE = 4
USE_EMA_WEIGHTS = True
# How many batches to average over.  4 is enough to smooth per-batch noise.
N_BATCHES = 4
# =======================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

import sys
sys.path.insert(0, "/home/ehrdt/tricor/src")

import numpy as np
import torch
import torch.nn.functional as F

from tricor.macerelax.model import LitRelaxML
from tricor.macerelax.data import RelaxMLDataModule


def _load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k.replace("._orig_mod.", "."): v
          for k, v in state["state_dict"].items()}
    lit = LitRelaxML(**state["hyper_parameters"])
    lit.load_state_dict(sd, strict=True)
    lit.eval().to(device)
    if USE_EMA_WEIGHTS and hasattr(lit, "ema_model"):
        lit.ema_model.eval()
        return lit.ema_model.module.to(device)
    return lit.model.to(device)


def _forward(model, batch):
    return model(
        batch.z, batch.edge_index, batch.edge_attr, batch.w, batch.batch,
        batch.shell_pair_species, batch.shell_pair_features,
        batch.shell_pair_batch,
        batch.shell_trip_species, batch.shell_trip_features,
        batch.shell_trip_batch,
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CKPT_PATH}")

    model = _load_model(CKPT_PATH, device)

    # Build the same DataModule the training used, but with rotate=False
    # so we measure intrinsic alignment without augmentation noise per call.
    dm_no_rot = RelaxMLDataModule(
        manifest_path=TRAIN_MANIFEST, cutoff=CUTOFF,
        k_stride_snapshots=K_STRIDE_SNAPSHOTS, rotate=False,
        batch_size=BATCH_SIZE, num_workers=0,
        val_fraction=0.1, split_seed=42,
    )
    dm_no_rot.setup()

    # Also build the rotate=True version so we can see if the rotation
    # itself is the issue (model performs worse with augmentation = the
    # model isn't equivariant, only invariant on average).
    dm_rot = RelaxMLDataModule(
        manifest_path=TRAIN_MANIFEST, cutoff=CUTOFF,
        k_stride_snapshots=K_STRIDE_SNAPSHOTS, rotate=True,
        batch_size=BATCH_SIZE, num_workers=0,
        val_fraction=0.1, split_seed=42,
    )
    dm_rot.setup()

    print(f"Train items: {len(dm_no_rot.train_set)}    "
          f"Val items: {len(dm_no_rot.val_set)}")
    print(f"Probing first {N_BATCHES} val batches "
          f"(batch_size={BATCH_SIZE})\n")

    for name, dm in [("rotate=False", dm_no_rot), ("rotate=True", dm_rot)]:
        loader = dm.val_dataloader()
        pred_norm_acc, tgt_norm_acc, cos_acc, dot_acc = [], [], [], []
        pred_sq_acc, tgt_sq_acc, mse_acc = [], [], []
        it = iter(loader)
        for i in range(N_BATCHES):
            batch = next(it).to(device)
            with torch.no_grad():
                pred = _forward(model, batch)
            target = batch.target_displacement
            pn = pred.norm(dim=-1)
            tn = target.norm(dim=-1)
            cos = F.cosine_similarity(pred, target, dim=-1)
            dot = (pred * target).sum(dim=-1)
            pred_norm_acc.append(pn.mean().item())
            tgt_norm_acc.append(tn.mean().item())
            cos_acc.append(cos.mean().item())
            dot_acc.append(dot.mean().item())
            pred_sq_acc.append(pn.pow(2).mean().item())
            tgt_sq_acc.append(tn.pow(2).mean().item())
            mse_acc.append(((pred - target).pow(2).sum(-1)).mean().item())

        pred_norm = float(np.mean(pred_norm_acc))
        tgt_norm  = float(np.mean(tgt_norm_acc))
        cos_mean  = float(np.mean(cos_acc))
        dot_mean  = float(np.mean(dot_acc))
        pred_sq   = float(np.mean(pred_sq_acc))
        tgt_sq    = float(np.mean(tgt_sq_acc))
        mse       = float(np.mean(mse_acc))
        rel_loss  = mse / tgt_sq

        print(f"=== {name} (loader matched training; eval mode, EMA weights) ===")
        print(f"  per-atom |pred|  mean:  {pred_norm:.5f} Å")
        print(f"  per-atom |target| mean: {tgt_norm:.5f} Å")
        print(f"  ratio |pred|/|target|:  {pred_norm / tgt_norm:.3f}")
        print(f"  cosine(pred, target) mean: {cos_mean:+.4f}")
        print(f"  E[pred·target] / E[|target|²]: {dot_mean / tgt_sq:+.4f}")
        print(f"  E[|pred|²]     / E[|target|²]: {pred_sq / tgt_sq:.4f}")
        print(f"  reconstructed rel_loss:        {rel_loss:.4f}")
        print(f"     = 1 - 2·E[p·t]/E[|t|²] + E[|p|²]/E[|t|²]")
        print(f"     = 1 - {2 * dot_mean / tgt_sq:+.4f}  +  {pred_sq / tgt_sq:.4f}")
        print()


if __name__ == "__main__":
    main()
