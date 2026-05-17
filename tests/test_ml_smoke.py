"""Smoke tests for the ML acceleration backend.

These tests verify the *plumbing* — graph construction, EGNN forward
pass, end-to-end ``Supercell.generate(backend="ml")`` — but do NOT
test prediction quality (that needs trained weights, which are out
of scope for unit tests).

Full acceptance-gate tests (g3 / polyhedra count / no overlaps) live
in ``tests/test_ml_sio2.py`` once weights are trained.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
    from tricor.ml import EGNN, predict_positions
    from tricor.ml.dataset import build_pbc_graph, collate_cells
    from tricor.ml.egnn import EGNNLayer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


def test_pbc_graph_basic():
    """Two atoms in a 5 Å box should give 2 directed edges (i→j, j→i)
    if within cutoff, both vectors min-image-correct."""
    pos = np.array([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32)
    box = (5.0, 5.0, 5.0)
    ei, ev = build_pbc_graph(pos, box, r_cut=2.5)
    # Min-image displacement is 5 - 3 = 2, which is the wrap-around
    # direction (closer than the direct 3.0).
    assert ei.shape[1] == 2
    norms = torch.linalg.norm(ev, dim=-1).tolist()
    assert all(abs(n - 2.0) < 1e-4 for n in norms)


def test_pbc_graph_respects_cutoff():
    pos = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32)
    box = (10.0, 10.0, 10.0)
    # Distance is 4 Å, both directions.  Cutoff 3 → no edges.
    ei, ev = build_pbc_graph(pos, box, r_cut=3.0)
    assert ei.shape[1] == 0
    # Cutoff 5 → 2 edges
    ei, ev = build_pbc_graph(pos, box, r_cut=5.0)
    assert ei.shape[1] == 2


def test_egnn_layer_shapes():
    layer = EGNNLayer(hidden_dim=16, cond_dim=4)
    n, e = 5, 8
    h = torch.randn(n, 16)
    x = torch.randn(n, 3)
    edge_index = torch.randint(0, n, (2, e), dtype=torch.long)
    edge_vec = torch.randn(e, 3)
    cond = torch.randn(e, 4)
    h2, x2 = layer(h, x, edge_index, edge_vec, cond)
    assert h2.shape == (n, 16)
    assert x2.shape == (n, 3)


def test_egnn_forward_pass():
    model = EGNN(n_species=2, hidden_dim=32, n_layers=2,
                 species_embedding_dim=8, cond_dim=4)
    n, e = 10, 20
    pos = torch.randn(n, 3)
    species = torch.randint(0, 2, (n,))
    edge_index = torch.randint(0, n, (2, e), dtype=torch.long)
    edge_vec = torch.randn(e, 3)
    cond = torch.tensor([[2.0]], dtype=torch.float32)
    out = model(positions=pos, species_idx=species,
                edge_index=edge_index, edge_vec=edge_vec,
                cond=cond, batch=None)
    assert out.shape == (n, 3)
    assert torch.isfinite(out).all()


def test_egnn_translation_equivariance():
    """Translating the input by a constant vector should translate
    the output by the same vector."""
    torch.manual_seed(0)
    model = EGNN(n_species=2, hidden_dim=16, n_layers=2,
                 species_embedding_dim=8, cond_dim=0,
                 cond_input_dim=1).eval()
    n, e = 6, 10
    pos = torch.randn(n, 3)
    species = torch.randint(0, 2, (n,))
    edge_index = torch.randint(0, n, (2, e), dtype=torch.long)
    edge_vec = torch.randn(e, 3)  # invariant under translation of pos

    with torch.no_grad():
        out_a = model(positions=pos, species_idx=species,
                      edge_index=edge_index, edge_vec=edge_vec,
                      cond=None, batch=None)
        shift = torch.tensor([1.5, -2.0, 0.7])
        out_b = model(positions=pos + shift, species_idx=species,
                      edge_index=edge_index, edge_vec=edge_vec,
                      cond=None, batch=None)
    assert torch.allclose(out_b - out_a, shift.expand(n, 3), atol=1e-5)


def test_collate_cells_graph_offsets():
    """Two cells of different sizes should give a stitched graph with
    correctly offset edge indices."""
    s1 = {
        "voronoi_pos": torch.randn(4, 3),
        "target_pos": torch.randn(4, 3),
        "species_idx": torch.zeros(4, dtype=torch.long),
        "grain_size": 10.0,
        "box_dim": torch.tensor([5.0, 5.0, 5.0]),
        "regime": "test",
    }
    s2 = {
        "voronoi_pos": torch.randn(6, 3),
        "target_pos": torch.randn(6, 3),
        "species_idx": torch.zeros(6, dtype=torch.long),
        "grain_size": 12.0,
        "box_dim": torch.tensor([5.0, 5.0, 5.0]),
        "regime": "test",
    }
    batch = collate_cells([s1, s2], r_cut=3.0)
    assert batch["positions"].shape[0] == 10
    assert batch["batch"].tolist() == [0]*4 + [1]*6
    # All edge indices must point into [0, 10)
    assert int(batch["edge_index"].max()) < 10
    assert int(batch["edge_index"].min()) >= 0
    assert batch["cond"].shape == (2, 1)


def test_supercell_generate_backend_ml(atoms_sio2):
    """Build a small SiO₂ cell with a randomly-initialised EGNN to
    confirm the ``backend='ml'`` plumbing works end-to-end (does NOT
    test quality — model is untrained)."""
    import tricor as tc
    from tricor.shells import CoordinationShellTarget

    shell = CoordinationShellTarget.from_atoms(atoms_sio2, phi_num_bins=24)
    cell = tc.Supercell.from_atoms(
        atoms_sio2, cell_dim_angstroms=(12, 12, 12),
        r_max=6.0, r_step=0.1, phi_num_bins=24, rng_seed=42,
    )
    # Build untrained EGNN, save to disk, load via backend="ml"
    model = EGNN(n_species=2, hidden_dim=16, n_layers=2,
                 species_embedding_dim=8, cond_dim=4)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save({
            "model_state": model.state_dict(),
            "config": {
                "n_species": 2, "hidden_dim": 16, "n_layers": 2,
                "species_embedding_dim": 8, "cond_dim": 4,
                "cond_input_dim": 1, "r_cut": 5.0,
            },
        }, f.name)
        ckpt = f.name

    try:
        cell.generate(
            shell, grain_size=8.0,
            bond_weight=1.65, angle_weight=1.35, repulsion_weight=1.3,
            hard_core_scale=0.82, nonbond_push_scale=0.72,
            displacement_sigma=0.011,
            backend="ml", ml_model=ckpt,
            show_progress=False,
        )
        # Just verify the cell has reasonable atom count and is
        # inside the box.
        assert len(cell.atoms) > 0
        box = np.diag(np.asarray(cell.atoms.cell.array))
        assert (cell.atoms.positions >= 0).all()
        assert (cell.atoms.positions < box[None, :] + 1e-3).all()
    finally:
        Path(ckpt).unlink(missing_ok=True)
