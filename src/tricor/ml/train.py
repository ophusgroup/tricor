"""Training loop for the EGNN regressor.

Usage from the repo root:

.. code-block:: shell

    python -m tricor.ml.train \
        --data src/tricor/ml/data/sio2/train.h5 \
        --val  src/tricor/ml/data/sio2/val.h5 \
        --out  src/tricor/ml/data/sio2/checkpoint.pt \
        --n-species 2 \
        --hidden-dim 64 \
        --n-layers 4 \
        --epochs 50 \
        --batch-size 4 \
        --lr 1e-3 \
        --device auto

Predicts per-atom displacement.  Loss is MSE on ``(fire_pos -
voronoi_pos)`` predicted vs actual.  Optionally includes a hard-core
penalty so the model is encouraged not to produce sub-NN pair
distances.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import TricorMLDataset, collate_cells
from .egnn import EGNN


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def displacement_loss(pred_pos: torch.Tensor,
                      target_pos: torch.Tensor,
                      voronoi_pos: torch.Tensor,
                      box_dim: torch.Tensor | None = None,
                      batch: torch.Tensor | None = None) -> torch.Tensor:
    """MSE on the per-atom displacement, PBC-corrected.

    Predicting the residual ``(target - voronoi)`` rather than the
    absolute position gives a smaller, better-conditioned target.

    Parameters
    ----------
    pred_pos    : (sum_N, 3) predicted positions
    target_pos  : (sum_N, 3) FIRE-final positions
    voronoi_pos : (sum_N, 3) Voronoi-tile positions (input)
    box_dim     : optional (B, 3) per-cell orthorhombic box.  Required
                  for correct loss — atoms can wrap during FIRE and
                  the resulting raw displacement (~ box_size, 20+ Å)
                  dominates the MSE and destroys the gradient signal
                  if not min-image-corrected.
    batch       : optional (sum_N,) cell index per atom — required
                  together with ``box_dim``.
    """
    pred_disp = pred_pos - voronoi_pos
    target_disp = target_pos - voronoi_pos
    if box_dim is not None and batch is not None:
        box_per_atom = box_dim[batch]                              # (N, 3)
        target_disp = target_disp - torch.round(
            target_disp / box_per_atom
        ) * box_per_atom
        # Also PBC-correct the prediction so the network can satisfy
        # the constraint by simply moving the atom to either wrap.
        pred_disp = pred_disp - torch.round(
            pred_disp / box_per_atom
        ) * box_per_atom
    return torch.nn.functional.mse_loss(pred_disp, target_disp)


def train_one_epoch(
    model: EGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        # Move tensors to device
        positions = batch["positions"].to(device)
        target_pos = batch["target_pos"].to(device)
        species_idx = batch["species_idx"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_vec = batch["edge_vec"].to(device)
        cond = batch["cond"].to(device)
        batch_idx = batch["batch"].to(device)
        box_dim = batch["box_dim"].to(device)

        pred_pos = model(
            positions=positions,
            species_idx=species_idx,
            edge_index=edge_index,
            edge_vec=edge_vec,
            cond=cond,
            batch=batch_idx,
        )
        loss = displacement_loss(
            pred_pos, target_pos, positions,
            box_dim=box_dim, batch=batch_idx,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.detach())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model: EGNN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        positions = batch["positions"].to(device)
        target_pos = batch["target_pos"].to(device)
        species_idx = batch["species_idx"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_vec = batch["edge_vec"].to(device)
        cond = batch["cond"].to(device)
        batch_idx = batch["batch"].to(device)
        box_dim = batch["box_dim"].to(device)

        pred_pos = model(
            positions=positions,
            species_idx=species_idx,
            edge_index=edge_index,
            edge_vec=edge_vec,
            cond=cond,
            batch=batch_idx,
        )
        loss = displacement_loss(
            pred_pos, target_pos, positions,
            box_dim=box_dim, batch=batch_idx,
        )
        total_loss += float(loss.detach())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _peek_n_species(h5_path: str | Path) -> int:
    """Look at the first cell in the HDF5 file to determine n_species."""
    with h5py.File(h5_path, "r") as f:
        keys = [k for k in f.keys() if k.startswith("cell_")]
        if not keys:
            raise ValueError(f"No cells in {h5_path}")
        species_max = 0
        for k in keys[:50]:  # sample first 50 cells
            s = np.asarray(f[k]["species_idx"])
            species_max = max(species_max, int(s.max()))
    return species_max + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="HDF5 dataset")
    parser.add_argument("--val", default=None,
                        help="Optional separate val HDF5; "
                             "else 10%% of --data is held out")
    parser.add_argument("--out", required=True, help="Checkpoint output (.pt)")
    parser.add_argument("--n-species", type=int, default=None,
                        help="Override autodetect")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--species-embedding-dim", type=int, default=16)
    parser.add_argument("--cond-dim", type=int, default=8)
    parser.add_argument("--cond-input-dim", type=int, default=1)
    parser.add_argument("--r-cut", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-trajectory-frames", action="store_true",
                        help="Use every Nth FIRE frame as an extra training "
                             "target (much more data per cell).")
    parser.add_argument("--max-box-side", type=float, default=None,
                        help="Restrict dataset to cells with box side "
                             "<= this many Å (e.g. 20 to drop 25³ cells "
                             "for faster training).")
    parser.add_argument("--regime-filter", nargs="+", default=None,
                        help="Only train on these regimes "
                             "(e.g. amorphous nanocrystalline).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)

    n_species = args.n_species or _peek_n_species(args.data)
    print(f"n_species (autodetected): {n_species}", flush=True)
    print(f"device: {device}", flush=True)

    dataset = TricorMLDataset(
        args.data, r_cut=args.r_cut,
        use_trajectory_frames=args.use_trajectory_frames,
        max_box_side=args.max_box_side,
        regime_filter=args.regime_filter,
    )
    print(f"dataset size: {len(dataset)} samples", flush=True)

    if args.val:
        val_dataset = TricorMLDataset(args.val, r_cut=args.r_cut)
        train_dataset = dataset
    else:
        n_val = max(1, len(dataset) // 10)
        train_dataset, val_dataset = random_split(
            dataset, [len(dataset) - n_val, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )

    def _collate(batch):
        return collate_cells(batch, r_cut=args.r_cut)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=0,
    )

    model = EGNN(
        n_species=n_species,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        species_embedding_dim=args.species_embedding_dim,
        cond_dim=args.cond_dim,
        cond_input_dim=args.cond_input_dim,
    ).to(device)
    print(f"model params: {model.num_parameters():,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "wall_s": dt})
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "n_species": n_species,
                    "hidden_dim": args.hidden_dim,
                    "n_layers": args.n_layers,
                    "species_embedding_dim": args.species_embedding_dim,
                    "cond_dim": args.cond_dim,
                    "cond_input_dim": args.cond_input_dim,
                    "r_cut": args.r_cut,
                },
            }, out_path)
            marker = "  [best]"
        print(f"epoch {epoch:>3d}/{args.epochs}  "
              f"train {train_loss:.5f}  val {val_loss:.5f}  "
              f"({dt:.1f}s){marker}", flush=True)

    # Save history alongside
    hist_path = out_path.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"saved checkpoint to {out_path}", flush=True)
    print(f"saved history   to {hist_path}", flush=True)


if __name__ == "__main__":
    main()
