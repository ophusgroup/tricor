"""Smoke test for the multi-species RelaxMLModel.

Verifies that:
  1. The model constructs with max_z=120.
  2. A forward pass on a real Si .npz produces the right output shape, no NaNs.
  3. Swapping half the atoms to a different species (Z=7) changes the output
     — confirms species info actually flows through both NodeEncoder and the
     pair-aware EdgeEncoder.
  4. The Embedding lookup returns finite values for arbitrary Z.

Run on mallard (where graphite is installed):
    python smoke_test.py [path/to/some.npz]

Defaults to a Si trajectory under data/.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
import torch

from torch_geometric.data import Data, Batch

from tricor.flowmatch.flow_utils import periodic_radius_graph_cell_list
from tricor.relaxml.model import LitRelaxML
from tricor.relaxml.data import NUM_WEIGHT_FEATURES


DEFAULT_NPZ = "./data/si_trajectories_v2/si_amorphous_cell050_idx00018_seed000131018.npz"


def main() -> None:
    npz_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NPZ)
    if not npz_path.is_file():
        sys.exit(f"npz not found: {npz_path}")

    torch.manual_seed(0)
    torch.set_num_threads(1)

    npz = np.load(npz_path)
    pos = torch.tensor(npz["positions"][0], dtype=torch.float32)
    cell = torch.tensor(npz["cell"], dtype=torch.float32)
    species = torch.tensor(npz["species_numbers"], dtype=torch.long)
    print(f"loaded {npz_path.name}  N={pos.shape[0]}  unique Z={sorted(set(species.tolist()))}")

    ei, ev = periodic_radius_graph_cell_list(pos, 5.0, cell)
    edge_attr = torch.hstack([ev, ev.norm(dim=-1, keepdim=True)])
    w = torch.zeros(NUM_WEIGHT_FEATURES, dtype=torch.float32).unsqueeze(0)

    # Small model so this runs fast on CPU as well.
    lit = LitRelaxML(
        max_z=120, node_dim=64, edge_dim=64, num_convs=2,
        species_pair_dim=8, weight_encoder_hidden=32,
    )
    lit.eval()
    n_params = sum(p.numel() for p in lit.model.parameters())
    print(f"model params: {n_params:,}")

    # 1) Forward on real Si data
    data = Data(z=species, pos=pos, edge_index=ei, edge_attr=edge_attr, w=w)
    batch = Batch.from_data_list([data])
    with torch.no_grad():
        out_si = lit.model(batch.z, batch.edge_index, batch.edge_attr,
                           batch.w, batch.batch)
    assert out_si.shape == pos.shape, f"shape mismatch: {out_si.shape} vs {pos.shape}"
    assert torch.isfinite(out_si).all(), "NaN/Inf in Si output"
    print(f"[ok] Si forward: shape={tuple(out_si.shape)}  "
          f"mean|out|={out_si.norm(dim=-1).mean():.4f}")

    # 2) Swap half atoms to a different species — output must change
    N = pos.shape[0]
    species_bin = torch.cat([
        torch.full((N // 2,), 14, dtype=torch.long),
        torch.full((N - N // 2,), 7, dtype=torch.long),
    ])
    data2 = Data(z=species_bin, pos=pos, edge_index=ei, edge_attr=edge_attr, w=w)
    batch2 = Batch.from_data_list([data2])
    with torch.no_grad():
        out_bin = lit.model(batch2.z, batch2.edge_index, batch2.edge_attr,
                            batch2.w, batch2.batch)
    assert torch.isfinite(out_bin).all(), "NaN/Inf in binary output"
    diff = (out_si - out_bin).norm(dim=-1).mean().item()
    assert diff > 1e-6, "binary output identical to Si — species channel inert!"
    print(f"[ok] Binary forward: shape={tuple(out_bin.shape)}  "
          f"mean per-atom diff vs Si-only={diff:.4f}")

    # 3) Embedding sanity check across the full Z range
    all_z = torch.arange(120, dtype=torch.long)
    emb = lit.model.node_encoder(all_z)
    pair = lit.model.pair_species_embed(all_z)
    assert torch.isfinite(emb).all() and torch.isfinite(pair).all()
    print(f"[ok] Embedding lookup over Z=1..120  "
          f"node_emb shape={tuple(emb.shape)}  pair_emb shape={tuple(pair.shape)}")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
