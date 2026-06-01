"""Step-by-step MACE+wall relaxation surrogate — fork of
tricor.relaxml.model_shelltgt.

Architecturally identical to the relaxml baseline:
  - global per-graph conditioning vector (NUM_WEIGHT_FEATURES, see
    .data — 6 for the MACE pilot vs 9 for the shell_relax baseline)
  - per-graph shell_target conditioning (ShellTargetEncoder; same as
    relaxml — composition-intrinsic, works for MACE trajectories too)

The WeightEncoder input dim adapts automatically via num_weight_features
default. No other model code changes vs relaxml. See MACE_RELAX_PILOT.md
for the reasoning on what conditioning fields apply to MACE+wall.
"""

from __future__ import annotations

from typing import Tuple

import lightning as L
import torch
import torch._dynamo as _dynamo
from torch import Tensor, nn

from graphite.nn import MLP
from graphite.nn.convs.mgn import MeshGraphNetsConv
from graphite.nn.models.mgn import Decoder
from torch_geometric.utils import scatter

from .data import NUM_WEIGHT_FEATURES
# shell_target helpers stay in the relaxml package — composition-intrinsic,
# works identically for MACE-source trajectories once the NPZs have the
# shell_target arrays appended by mace/add_shell_target_to_pilot.py.
from tricor.relaxml.shell_target import NUM_PAIR_FEATURES, NUM_TRIPLET_FEATURES


# Periodic table covers Z = 1..118; sized to 120 gives a small safety margin.
DEFAULT_MAX_Z: int = 120


# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────────────────────


class NodeEncoder(nn.Module):
    """Atomic number Z -> node embedding via lookup table."""

    def __init__(self, max_z: int, node_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(max_z, node_dim)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, z: Tensor) -> Tensor:
        return self.norm(self.embed(z))


class EdgeEncoder(nn.Module):
    """[dx, dy, dz, r, src_pair_emb, dst_pair_emb] -> edge embedding.

    The geometric features `[dx, dy, dz, r]` are concatenated with a
    small (`species_pair_dim`) embedding of the source and destination
    atomic numbers so the network sees pair identity directly.
    """

    def __init__(
        self,
        edge_dim: int,
        species_pair_dim: int,
        init_geom_dim: int = 4,
    ) -> None:
        super().__init__()
        in_dim = init_geom_dim + 2 * species_pair_dim
        self.embed = nn.Sequential(
            MLP([in_dim, edge_dim, edge_dim], act=nn.SiLU()),
            nn.LayerNorm(edge_dim),
        )

    def forward(
        self, edge_attr: Tensor, src_pair: Tensor, dst_pair: Tensor,
    ) -> Tensor:
        return self.embed(torch.cat([edge_attr, src_pair, dst_pair], dim=-1))


class WeightEncoder(nn.Module):
    """Global weight-parameter vector (NUM_WEIGHT_FEATURES,) -> node-space embedding.

    Called once per graph.  The result is broadcast over every atom in
    that graph and added to the node embeddings before message passing.
    """

    def __init__(
        self,
        node_dim: int,
        num_weight_features: int = NUM_WEIGHT_FEATURES,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            MLP([num_weight_features, hidden_dim, node_dim], act=nn.SiLU()),
            nn.LayerNorm(node_dim),
        )

    def forward(self, w: Tensor) -> Tensor:
        """
        Args:
            w: (B, num_weight_features) — one row per graph in the batch.

        Returns:
            (B, node_dim) — one embedding per graph.
        """
        return self.embed(w)


class ShellTargetEncoder(nn.Module):
    """Deep-set encoder for shell_target arrays.

    Each (pair, triplet) tuple is mapped to a fixed-size vector via a
    small MLP, then summed per-graph.  Pair and triplet pools are
    concatenated and projected to ``out_dim``.  Result is one vector
    per graph in the batch.

    Forward inputs
    --------------
    pair_species   (P_total, 2) long       atomic numbers (Z_a, Z_b) per pair
    pair_features  (P_total, 4) float      [target_r, sigma, n_ab, n_ba]
    pair_batch     (P_total,)   long       per-pair graph index in [0, B)
    trip_species   (T_total, 3) long       atomic numbers (Z_a, Z_b, Z_c)
    trip_features  (T_total, 2) float      [angle_mode_rad, mass_weight]
    trip_batch     (T_total,)   long       per-triplet graph index in [0, B)
    num_graphs     int                     B (the batch size)

    Returns
    -------
    (B, out_dim) — one shell_target embedding per graph.

    Notes on batching: when a graph has zero pairs (or zero triplets), the
    sum scatters to zero for that graph slot, which is the right
    permutation-invariant aggregation for an empty set.
    """

    def __init__(
        self,
        max_z: int,
        out_dim: int,
        species_emb_dim: int = 8,
        pair_hidden: int = 64,
        triplet_hidden: int = 64,
    ) -> None:
        super().__init__()
        # Independent species embedding (small) so the encoder doesn't pull
        # on the node_encoder's parameters and so it stays cheap.
        self.species_emb = nn.Embedding(max_z, species_emb_dim)
        pair_in = 2 * species_emb_dim + NUM_PAIR_FEATURES
        trip_in = 3 * species_emb_dim + NUM_TRIPLET_FEATURES
        self.pair_mlp = nn.Sequential(
            MLP([pair_in, pair_hidden, pair_hidden], act=nn.SiLU()),
        )
        self.triplet_mlp = nn.Sequential(
            MLP([trip_in, triplet_hidden, triplet_hidden], act=nn.SiLU()),
        )
        self.out = nn.Sequential(
            MLP([pair_hidden + triplet_hidden, out_dim, out_dim], act=nn.SiLU()),
            nn.LayerNorm(out_dim),
        )

    # @_dynamo.disable: shell_target tensors are variable-length per
    # batch (different compounds contribute different P / T counts), and
    # leaving this branch in eager mode prevents torch.compile from
    # repeatedly recompiling the surrounding model when shapes change.
    # The MGN backbone (which dominates compute and benefits most from
    # compile) keeps its 2× speedup; this small encoder runs in eager.
    @_dynamo.disable
    def forward(
        self,
        pair_species: Tensor,
        pair_features: Tensor,
        pair_batch: Tensor,
        trip_species: Tensor,
        trip_features: Tensor,
        trip_batch: Tensor,
        num_graphs: int,
    ) -> Tensor:
        # Pair branch: encode each (Z_a, Z_b, target_r, sigma, n_ab, n_ba)
        # tuple, then sum over pairs in each graph.
        za = self.species_emb(pair_species[:, 0])
        zb = self.species_emb(pair_species[:, 1])
        h_pair = self.pair_mlp(torch.cat([za, zb, pair_features], dim=-1))
        pair_pool = scatter(
            h_pair, pair_batch, dim=0, dim_size=num_graphs, reduce="sum",
        )

        # Triplet branch: same idea over triplet tuples.
        za = self.species_emb(trip_species[:, 0])
        zb = self.species_emb(trip_species[:, 1])
        zc = self.species_emb(trip_species[:, 2])
        h_trip = self.triplet_mlp(torch.cat([za, zb, zc, trip_features], dim=-1))
        trip_pool = scatter(
            h_trip, trip_batch, dim=0, dim_size=num_graphs, reduce="sum",
        )

        return self.out(torch.cat([pair_pool, trip_pool], dim=-1))


class Processor(nn.Module):
    """N MeshGraphNetsConv layers with residual + LayerNorm."""

    def __init__(self, num_convs: int, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            MeshGraphNetsConv(node_dim, edge_dim) for _ in range(num_convs)
        ])
        self.node_norms = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(num_convs)
        ])
        self.edge_norms = nn.ModuleList([
            nn.LayerNorm(edge_dim) for _ in range(num_convs)
        ])

    def forward(
        self, h_node: Tensor, edge_index: Tensor, h_edge: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        for conv, n_norm, e_norm in zip(self.convs, self.node_norms, self.edge_norms):
            dn, de = conv(h_node, edge_index, h_edge)
            h_node = n_norm(h_node + dn)
            h_edge = e_norm(h_edge + de)
        return h_node, h_edge


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────


class RelaxMLModel(nn.Module):
    """Step-by-step relaxation surrogate with shell_target conditioning.

    Predicts per-atom min-image displacement over k tricor steps given
    the current positions, species, graph, global weight parameters, and
    shell_target (per-pair distances + counts and per-triplet angles).

    Forward inputs
    --------------
    z              (N,)            long  atomic numbers
    edge_index     (2, E)
    edge_attr      (E, 4)          [dx, dy, dz, r]
    w              (B, num_weight_features)  per-graph regime weights
    batch          (N,)            node→graph index
    pair_species   (P_total, 2)    long  per-pair (Z_a, Z_b)
    pair_features  (P_total, 4)    [target_r, sigma, n_ab, n_ba]
    pair_batch     (P_total,)      pair→graph index
    trip_species   (T_total, 3)    long  per-triplet (Z_a, Z_b, Z_c)
    trip_features  (T_total, 2)    [angle_mode_rad, mass_weight]
    trip_batch     (T_total,)      triplet→graph index

    Returns
    -------
    displacement  (N, 3)
    """

    def __init__(
        self,
        max_z: int = DEFAULT_MAX_Z,
        num_weight_features: int = NUM_WEIGHT_FEATURES,
        node_dim: int = 128,
        edge_dim: int = 128,
        num_convs: int = 4,
        weight_encoder_hidden: int = 64,
        species_pair_dim: int = 16,
        shell_target_species_dim: int = 8,
        shell_target_hidden: int = 64,
        shell_target_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_z = int(max_z)
        # Per-graph dropout on the shell_target conditioning signal during
        # training (classifier-free-guidance style).  When >0, with this
        # probability the shell_target encoder's output for a graph is
        # zeroed before being added to the per-atom features — forcing
        # the model to produce useful predictions both with and without
        # conditioning, which in turn forces it to actually USE the
        # conditioning when present (instead of shortcutting via species).
        # Disabled at eval/inference (uses self.training flag).
        self.shell_target_dropout = float(shell_target_dropout)
        self.node_encoder = NodeEncoder(max_z, node_dim)
        # Separate small embedding fed into the edge encoder so pair
        # identity is explicit on every edge.  Independent of the node
        # embedding so the edge MLP doesn't grow with node_dim.
        self.pair_species_embed = nn.Embedding(max_z, species_pair_dim)
        self.edge_encoder = EdgeEncoder(
            edge_dim=edge_dim, species_pair_dim=species_pair_dim,
        )
        self.weight_encoder = WeightEncoder(
            node_dim=node_dim,
            num_weight_features=num_weight_features,
            hidden_dim=weight_encoder_hidden,
        )
        # Phase conditioning: encodes the shell_target driving this
        # trajectory into a per-graph vector that gets added to the
        # weight-encoder output.  Same broadcasting pattern.
        self.shell_target_encoder = ShellTargetEncoder(
            max_z=max_z,
            out_dim=node_dim,
            species_emb_dim=shell_target_species_dim,
            pair_hidden=shell_target_hidden,
            triplet_hidden=shell_target_hidden,
        )
        self.processor = Processor(num_convs, node_dim, edge_dim)
        self.decoder = Decoder(node_dim, 3)

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        w: Tensor,
        batch: Tensor,
        pair_species: Tensor,
        pair_features: Tensor,
        pair_batch: Tensor,
        trip_species: Tensor,
        trip_features: Tensor,
        trip_batch: Tensor,
    ) -> Tensor:
        h_node = self.node_encoder(z)

        # Edge features get explicit src/dst species identity.
        pair_emb = self.pair_species_embed(z)               # (N, species_pair_dim)
        src, dst = edge_index[0], edge_index[1]
        h_edge = self.edge_encoder(edge_attr, pair_emb[src], pair_emb[dst])

        # Per-graph conditioning: regime weights + shell_target.  Both
        # produce (B, node_dim) vectors which are summed and broadcast
        # across the atoms of each graph.
        num_graphs = int(w.shape[0])
        w_emb = self.weight_encoder(w)                      # (B, node_dim)
        st_emb = self.shell_target_encoder(
            pair_species, pair_features, pair_batch,
            trip_species, trip_features, trip_batch,
            num_graphs,
        )                                                   # (B, node_dim)

        # Conditioning dropout: zero the shell_target embedding for a
        # subset of graphs at training time so the model is forced to
        # produce useful predictions both with and without conditioning.
        # That asymmetry is what makes the encoder informative — if the
        # model could ignore conditioning, the with-conditioning batches
        # would have higher loss than the without-conditioning ones.
        if self.training and self.shell_target_dropout > 0.0:
            keep = (
                torch.rand(num_graphs, device=st_emb.device)
                > self.shell_target_dropout
            ).to(st_emb.dtype).unsqueeze(-1)                # (B, 1)
            st_emb = st_emb * keep
        h_node = h_node + w_emb[batch] + st_emb[batch]

        h_node, _ = self.processor(h_node, edge_index, h_edge)
        return self.decoder(h_node)


# ──────────────────────────────────────────────────────────────────────────────
# Lightning training module
# ──────────────────────────────────────────────────────────────────────────────


class LitRelaxML(L.LightningModule):
    """Lightning training wrapper for the shell_target-conditioned RelaxMLModel.

    Hyperparameters (sensible defaults for multi-species + phase-conditioned
    training):
      - max_z: 120 (full periodic table + safety margin)
      - num_convs: 4
      - node_dim / edge_dim: 128
      - species_pair_dim: 16 (size of edge-side species embedding)
      - weight_encoder_hidden: 64
      - shell_target_species_dim: 8 (small embedding inside ShellTargetEncoder)
      - shell_target_hidden: 64
      - ema_decay: 0.9999
      - learn_rate: 1e-3
    """

    def __init__(
        self,
        max_z: int = DEFAULT_MAX_Z,
        num_weight_features: int = NUM_WEIGHT_FEATURES,
        node_dim: int = 128,
        edge_dim: int = 128,
        num_convs: int = 4,
        weight_encoder_hidden: int = 64,
        species_pair_dim: int = 16,
        shell_target_species_dim: int = 8,
        shell_target_hidden: int = 64,
        shell_target_dropout: float = 0.0,
        ema_decay: float = 0.9999,
        learn_rate: float = 1e-3,
        lr_schedule: str = "cosine",    # "none" | "cosine"
        lr_min_ratio: float = 0.01,
        warmup_steps: int = 500,
        weight_decay: float = 0.0,      # decoupled L2 via AdamW; 0 = vanilla Adam
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = RelaxMLModel(
            max_z=max_z,
            num_weight_features=num_weight_features,
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_convs=num_convs,
            weight_encoder_hidden=weight_encoder_hidden,
            species_pair_dim=species_pair_dim,
            shell_target_species_dim=shell_target_species_dim,
            shell_target_hidden=shell_target_hidden,
            shell_target_dropout=shell_target_dropout,
        )

        # EMA weights — standard trick from the flowmatch training loop.
        _ema = float(ema_decay)
        ema_avg = lambda avg_p, p, num_avg: _ema * avg_p + (1 - _ema) * p
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model, avg_fn=ema_avg,
        )

        self.learn_rate = learn_rate
        self.lr_schedule = lr_schedule
        self.lr_min_ratio = lr_min_ratio
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

    def _shared_step(self, batch) -> tuple[Tensor, Tensor]:
        """Returns (loss, zero_baseline).  Zero-baseline = the loss the
        same batch would score if the model output zero everywhere; it's
        the natural comparison target for "is the model actually doing
        anything?" since per-step displacements are tiny."""
        pred = self.model(
            batch.z, batch.edge_index, batch.edge_attr,
            batch.w, batch.batch,
            batch.shell_pair_species, batch.shell_pair_features,
            batch.shell_pair_batch,
            batch.shell_trip_species, batch.shell_trip_features,
            batch.shell_trip_batch,
        )
        loss = (pred - batch.target_displacement).pow(2).sum(dim=-1).mean()
        zero_baseline = batch.target_displacement.pow(2).sum(dim=-1).mean()
        return loss, zero_baseline

    def training_step(self, batch, batch_idx):
        loss, zero_baseline = self._shared_step(batch)
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        # relative_loss = 1.0 means model predicts zero (baseline);
        # 0.0 means perfect.  Much more interpretable than raw MSE.
        rel = loss / zero_baseline.clamp(min=1e-12)
        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        self.log("train_relative_loss", rel, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, zero_baseline = self._shared_step(batch)
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        rel = loss / zero_baseline.clamp(min=1e-12)
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        self.log("val_relative_loss", rel, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        return loss

    def configure_optimizers(self):
        # AdamW decouples weight decay from the gradient update.  With
        # weight_decay=0 it's mathematically equivalent to vanilla Adam.
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learn_rate,
            weight_decay=self.weight_decay,
        )
        if self.lr_schedule == "none":
            return opt

        total_steps = int(self.trainer.estimated_stepping_batches)
        eta_min = self.learn_rate * self.lr_min_ratio

        if self.lr_schedule == "cosine":
            cosine_steps = max(1, total_steps - self.warmup_steps)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cosine_steps, eta_min=eta_min,
            )
            if self.warmup_steps > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=1e-3, end_factor=1.0,
                    total_iters=self.warmup_steps,
                )
                sched = torch.optim.lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[warmup, cosine],
                    milestones=[self.warmup_steps],
                )
            else:
                sched = cosine
        else:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema_model.update_parameters(self.model)
