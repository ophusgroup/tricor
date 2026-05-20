"""Smoke test for the physics-feature shell_target RelaxMLModel variant.

Verifies that:
  1. The model constructs (max_z=120, SpeciesEncoder + ShellTargetEncoder).
  2. A forward pass on a real .npz with shell_target arrays produces
     the right output shape, no NaNs.
  3. Swapping the shell_target arrays for a fake "different phase"
     changes the output — confirms the conditioning channel is flowing.
  4. Swapping atomic numbers also changes the output — confirms the
     SpeciesEncoder pathway is live.
  5. The periodic-table buffer is loaded and the encoder produces
     finite outputs across Z = 0..119.
  6. Unseen-element check: train a 1-step gradient on a Si-only batch,
     then verify the embedding for an OOD element (N) has changed.
     This is the hypothesis-validation step — confirms parameter sharing
     across the periodic-table MLP, which the baseline nn.Embedding lacks.

Run on mallard (where graphite is installed):
    python smoke_test.py [path/to/some.npz]

Defaults to a multi-species trajectory under data/.  The .npz MUST
contain shell_pair_species and friends — run
``add_shell_target_to_npz.py`` first if it doesn't.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
import torch

from torch_geometric.data import Batch

from tricor.flowmatch.flow_utils import periodic_radius_graph_cell_list
from tricor.relaxml.shelltgt_phys import LitRelaxML
from tricor.relaxml.data_shelltgt import (
    NUM_WEIGHT_FEATURES, ShellTargetData,
)


DEFAULT_NPZ = "../data/multi_species_v1/Si_trajectories/Si_amorphous_cell050_idx00143_seed000400143.npz"


def _make_data(pos, ei, edge_attr, species, w, shell):
    P = shell["shell_pair_species"].shape[0]
    T = shell["shell_triplet_species"].shape[0]
    return ShellTargetData(
        z=species,
        pos=pos,
        edge_index=ei,
        edge_attr=edge_attr,
        w=w,
        shell_pair_species=torch.tensor(shell["shell_pair_species"], dtype=torch.long),
        shell_pair_features=torch.tensor(shell["shell_pair_features"], dtype=torch.float32),
        shell_pair_batch=torch.zeros(P, dtype=torch.long),
        shell_trip_species=torch.tensor(shell["shell_triplet_species"], dtype=torch.long),
        shell_trip_features=torch.tensor(shell["shell_triplet_features"], dtype=torch.float32),
        shell_trip_batch=torch.zeros(T, dtype=torch.long),
    )


def _run_forward(lit, batch):
    return lit.model(
        batch.z, batch.edge_index, batch.edge_attr,
        batch.w, batch.batch,
        batch.shell_pair_species, batch.shell_pair_features,
        batch.shell_pair_batch,
        batch.shell_trip_species, batch.shell_trip_features,
        batch.shell_trip_batch,
    )


def main() -> None:
    npz_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NPZ)
    if not npz_path.is_file():
        sys.exit(f"npz not found: {npz_path}")

    torch.manual_seed(0)
    torch.set_num_threads(1)

    npz = np.load(npz_path)
    if "shell_pair_species" not in npz.files:
        sys.exit(
            f"{npz_path.name} has no shell_target arrays.  Run "
            f"add_shell_target_to_npz.py against its source dir first."
        )

    pos = torch.tensor(npz["positions"][0], dtype=torch.float32)
    cell = torch.tensor(npz["cell"], dtype=torch.float32)
    species = torch.tensor(npz["species_numbers"], dtype=torch.long)
    shell = {k: np.asarray(npz[k]) for k in (
        "shell_pair_species", "shell_pair_features",
        "shell_triplet_species", "shell_triplet_features",
    )}
    print(f"loaded {npz_path.name}  N={pos.shape[0]}  unique Z={sorted(set(species.tolist()))}")
    print(f"shell_target: P={shell['shell_pair_species'].shape[0]}  "
          f"T={shell['shell_triplet_species'].shape[0]}")

    ei, ev = periodic_radius_graph_cell_list(pos, 5.0, cell)
    edge_attr = torch.hstack([ev, ev.norm(dim=-1, keepdim=True)])
    w = torch.zeros(NUM_WEIGHT_FEATURES, dtype=torch.float32).unsqueeze(0)

    # Small model so this runs fast on CPU as well.
    lit = LitRelaxML(
        max_z=120, node_dim=64, edge_dim=64, num_convs=2,
        species_pair_dim=8, weight_encoder_hidden=32,
        species_hidden=64,
        shell_target_species_dim=8, shell_target_hidden=32,
    )
    lit.eval()
    n_params = sum(p.numel() for p in lit.model.parameters())
    print(f"model params: {n_params:,}")

    # 1) Forward with the real shell_target
    data = _make_data(pos, ei, edge_attr, species, w, shell)
    batch = Batch.from_data_list([data])
    with torch.no_grad():
        out_real = _run_forward(lit, batch)
    assert out_real.shape == pos.shape, f"shape mismatch: {out_real.shape} vs {pos.shape}"
    assert torch.isfinite(out_real).all(), "NaN/Inf in real-shell output"
    print(f"[ok] Real shell_target forward: shape={tuple(out_real.shape)}  "
          f"mean|out|={out_real.norm(dim=-1).mean():.4f}")

    # 2) Perturb shell_pair_features and verify the output changes.
    fake_shell = {
        "shell_pair_species":     shell["shell_pair_species"].copy(),
        "shell_pair_features":    shell["shell_pair_features"].copy(),
        "shell_triplet_species":  shell["shell_triplet_species"].copy(),
        "shell_triplet_features": shell["shell_triplet_features"].copy(),
    }
    fake_shell["shell_pair_features"][:, 0] += 0.5
    data_fake = _make_data(pos, ei, edge_attr, species, w, fake_shell)
    batch_fake = Batch.from_data_list([data_fake])
    with torch.no_grad():
        out_fake = _run_forward(lit, batch_fake)
    diff_shell = (out_real - out_fake).norm(dim=-1).mean().item()
    assert diff_shell > 1e-6, "fake-shell output identical to real — ShellTargetEncoder inert!"
    print(f"[ok] Perturbed shell_target forward: "
          f"mean per-atom diff vs real shell={diff_shell:.4f}")

    # 3) Z-swap: half atoms to a different species — output must change.
    N = pos.shape[0]
    species_bin = torch.cat([
        torch.full((N // 2,), 14, dtype=torch.long),
        torch.full((N - N // 2,), 7, dtype=torch.long),
    ])
    data_bin = _make_data(pos, ei, edge_attr, species_bin, w, shell)
    batch_bin = Batch.from_data_list([data_bin])
    with torch.no_grad():
        out_bin = _run_forward(lit, batch_bin)
    diff_z = (out_real - out_bin).norm(dim=-1).mean().item()
    assert diff_z > 1e-6, "Z-swap output identical — species channel inert!"
    print(f"[ok] Z-swap forward: mean per-atom diff vs real Z={diff_z:.4f}")

    # 4) SpeciesEncoder lookup over the full Z range — all outputs finite.
    # Exercises every Z=0..119, including super-heavy synthetics (Rf..Og)
    # whose physics-feature rows are imputed from nearest-group neighbours
    # by build_periodic_table_features.  If pymatgen returns garbage for
    # any element, _validate_table catches it at module load.
    all_z = torch.arange(120, dtype=torch.long)
    node_emb = lit.model.node_encoder(all_z)
    pair_emb = lit.model.pair_species_embed(all_z)
    st_emb = lit.model.shell_target_encoder.species_emb(all_z)
    assert (
        torch.isfinite(node_emb).all()
        and torch.isfinite(pair_emb).all()
        and torch.isfinite(st_emb).all()
    ), "SpeciesEncoder produced NaN/Inf for some Z in 0..119"
    print(f"[ok] Embedding lookup over Z=0..119  "
          f"node={tuple(node_emb.shape)}  pair={tuple(pair_emb.shape)}  "
          f"shelltgt={tuple(st_emb.shape)}")

    # 4b) Per-element distinguishability: distinct Z values must produce
    # distinct embeddings.  If two elements collide (e.g. broken
    # imputation), the model can't tell them apart.  Check across
    # main-group + transition + lanthanide + actinide.
    sentinel_z = [1, 6, 7, 8, 13, 14, 26, 47, 57, 79, 92, 95, 100, 118]
    embs = lit.model.node_encoder(torch.tensor(sentinel_z, dtype=torch.long))
    pairwise_diffs = (embs.unsqueeze(0) - embs.unsqueeze(1)).norm(dim=-1)
    # Off-diagonal entries should all be nonzero (distinct embeddings).
    n = embs.shape[0]
    off_diag = pairwise_diffs[~torch.eye(n, dtype=torch.bool)]
    assert (off_diag > 1e-5).all(), (
        f"some sentinel-Z embeddings collided: min off-diag distance "
        f"{off_diag.min().item():.2e} (Z values: {sentinel_z})"
    )
    print(f"[ok] sentinel-Z distinguishability: "
          f"min off-diag |emb_a - emb_b| = {off_diag.min().item():.4f}")

    # 5) Buffer presence check.  The periodic-table table should be saved
    # in the state_dict so checkpoints survive without re-running
    # build_periodic_table_features.
    sd = lit.state_dict()
    table_keys = [k for k in sd if k.endswith(".table")]
    assert len(table_keys) >= 3, (
        f"expected at least 3 SpeciesEncoder.table buffers, found {len(table_keys)}"
    )
    print(f"[ok] periodic-table buffers in state_dict: {len(table_keys)}")

    # 6) Cross-element gradient propagation check — the hypothesis we're
    # actually testing.  Train one step on a Si-only batch, verify that
    # the N-row embedding has shifted.  In the baseline (nn.Embedding)
    # model this would NOT happen — the N row would receive zero
    # gradient.  In the SpeciesEncoder model the N embedding is
    # produced by the same MLP that processed Si features, so it must
    # change.
    lit.train()
    n_si = pos.shape[0]
    species_si = torch.full((n_si,), 14, dtype=torch.long)  # Si only
    data_si = _make_data(pos, ei, edge_attr, species_si, w, shell)
    batch_si = Batch.from_data_list([data_si])

    n_atom = torch.tensor([7], dtype=torch.long)  # nitrogen
    with torch.no_grad():
        emb_n_before = lit.model.node_encoder(n_atom).clone()

    # One gradient step on a fake target.
    target = torch.randn_like(pos) * 0.1
    pred = _run_forward(lit, batch_si)
    loss = (pred - target).pow(2).mean()
    opt = torch.optim.SGD(lit.model.parameters(), lr=0.1)
    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        emb_n_after = lit.model.node_encoder(n_atom).clone()
    delta_n = (emb_n_after - emb_n_before).norm().item()
    assert delta_n > 1e-7, (
        f"N embedding did not change after Si-only training step "
        f"(|Δemb_N|={delta_n:.2e}) — SpeciesEncoder MLP is not getting "
        f"cross-element gradient propagation!"
    )
    print(f"[ok] cross-element gradient propagation: "
          f"|Δemb_N| after Si-only step = {delta_n:.4f}")

    print("\nAll shell_target_phys smoke checks passed.")


if __name__ == "__main__":
    main()
