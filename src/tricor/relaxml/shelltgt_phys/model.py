"""Shell_target-conditioned relaxml surrogate with physics-feature species
embedding.

Variant of ``tricor.relaxml.model_shelltgt`` with one architectural
change: every ``nn.Embedding(MAX_Z, dim)`` lookup is replaced by a
``SpeciesEncoder`` — a shared MLP over a fixed periodic-table feature
table (group, period, electronegativity, covalent radius, ionization
energy, electron affinity, valence count, atomic mass, atomic volume,
plus block / period / group one-hots; see ``shelltgt_phys.species``).

The motivating problem: a learned ``nn.Embedding`` allocates one
independently-trained row per element.  Elements absent from training
stay at random init, so a model trained on Si + SiC + SiO2 cannot
embed N atoms when asked to predict Si3N4 — the row never received
gradient.  Sharing one MLP across all 120 element rows fixes that:
gradient from any Si-bearing graph updates the same MLP weights that
produce the embedding for N (via the shared physics-feature input).
The expected payoff is cross-composition generalization; the test
design is in scripts/relaxml/shelltgt_phys/train.py and
RELAXML_PHYS_NEXTSTEPS.md.

Three replacement sites
-----------------------
  * NodeEncoder.embed                 (per-atom node feature, dim 64-128)
  * RelaxMLModel.pair_species_embed   (per-atom edge species feature, dim 16)
  * ShellTargetEncoder.species_emb    (small per-atom feature for the
                                       shell-target deep-set encoder, dim 8)

The three encoders have independent MLP weights.  This is intentional
for now: it preserves the current information flow between the three
encoder paths and matches their differing output dimensionalities
without coupling.  A future refactor could share a single backbone with
three projection heads — measure benefit before doing it.

Forward / data signatures are identical to ``model_shelltgt`` — this
module re-uses ``data_shelltgt.RelaxMLDataModule`` unchanged.
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

from ..data_shelltgt import NUM_WEIGHT_FEATURES
from ..shell_target import NUM_PAIR_FEATURES, NUM_TRIPLET_FEATURES
from .species import SpeciesEncoder


DEFAULT_MAX_Z: int = 120


# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────────────────────


class NodeEncoder(nn.Module):
    """Atomic number Z -> node embedding via shared physics-feature MLP."""

    def __init__(
        self, max_z: int, node_dim: int, species_hidden: int = 128,
    ) -> None:
        super().__init__()
        # max_z is accepted for API parity with model_shelltgt.NodeEncoder;
        # SpeciesEncoder uses its own internal MAX_Z (120) for the buffer.
        self.embed = SpeciesEncoder(out_dim=node_dim, hidden_dim=species_hidden)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, z: Tensor) -> Tensor:
        return self.norm(self.embed(z))


class EdgeEncoder(nn.Module):
    """[dx, dy, dz, r, src_pair_emb, dst_pair_emb] -> edge embedding.

    Identical to model_shelltgt.EdgeEncoder — only the source of the
    pair embeddings changes (a SpeciesEncoder rather than nn.Embedding).
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

    Unchanged from model_shelltgt.
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
        return self.embed(w)


class ShellTargetEncoder(nn.Module):
    """Deep-set encoder for shell_target arrays.

    Identical to model_shelltgt.ShellTargetEncoder except that the
    per-atom species lookup is now a SpeciesEncoder (shared MLP over
    physics features).
    """

    def __init__(
        self,
        max_z: int,
        out_dim: int,
        species_emb_dim: int = 8,
        species_hidden: int = 128,
        pair_hidden: int = 64,
        triplet_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.species_emb = SpeciesEncoder(
            out_dim=species_emb_dim, hidden_dim=species_hidden,
        )
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
        za = self.species_emb(pair_species[:, 0])
        zb = self.species_emb(pair_species[:, 1])
        h_pair = self.pair_mlp(torch.cat([za, zb, pair_features], dim=-1))
        pair_pool = scatter(
            h_pair, pair_batch, dim=0, dim_size=num_graphs, reduce="sum",
        )

        za = self.species_emb(trip_species[:, 0])
        zb = self.species_emb(trip_species[:, 1])
        zc = self.species_emb(trip_species[:, 2])
        h_trip = self.triplet_mlp(torch.cat([za, zb, zc, trip_features], dim=-1))
        trip_pool = scatter(
            h_trip, trip_batch, dim=0, dim_size=num_graphs, reduce="sum",
        )

        return self.out(torch.cat([pair_pool, trip_pool], dim=-1))


class Processor(nn.Module):
    """N MeshGraphNetsConv layers with residual + LayerNorm.  Unchanged."""

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
    """Step-by-step relaxation surrogate with shell_target conditioning
    and physics-feature species embeddings.

    Forward inputs / outputs are identical to model_shelltgt.RelaxMLModel.
    The only difference is the species representation under the hood.
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
        species_hidden: int = 128,
        shell_target_species_dim: int = 8,
        shell_target_hidden: int = 64,
        shell_target_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_z = int(max_z)
        # Per-graph dropout on the shell_target conditioning signal during
        # training (classifier-free-guidance trick).  Same semantics as
        # model_shelltgt — see that module's docstring for the rationale.
        self.shell_target_dropout = float(shell_target_dropout)

        self.node_encoder = NodeEncoder(
            max_z=max_z, node_dim=node_dim, species_hidden=species_hidden,
        )
        # Per-edge species encoder: same role as model_shelltgt's
        # nn.Embedding(max_z, species_pair_dim), now a shared MLP so the
        # output for any Z benefits from gradient on every other Z.
        self.pair_species_embed = SpeciesEncoder(
            out_dim=species_pair_dim, hidden_dim=species_hidden,
        )
        self.edge_encoder = EdgeEncoder(
            edge_dim=edge_dim, species_pair_dim=species_pair_dim,
        )
        self.weight_encoder = WeightEncoder(
            node_dim=node_dim,
            num_weight_features=num_weight_features,
            hidden_dim=weight_encoder_hidden,
        )
        self.shell_target_encoder = ShellTargetEncoder(
            max_z=max_z,
            out_dim=node_dim,
            species_emb_dim=shell_target_species_dim,
            species_hidden=species_hidden,
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

        pair_emb = self.pair_species_embed(z)
        src, dst = edge_index[0], edge_index[1]
        h_edge = self.edge_encoder(edge_attr, pair_emb[src], pair_emb[dst])

        num_graphs = int(w.shape[0])
        w_emb = self.weight_encoder(w)
        st_emb = self.shell_target_encoder(
            pair_species, pair_features, pair_batch,
            trip_species, trip_features, trip_batch,
            num_graphs,
        )

        if self.training and self.shell_target_dropout > 0.0:
            keep = (
                torch.rand(num_graphs, device=st_emb.device)
                > self.shell_target_dropout
            ).to(st_emb.dtype).unsqueeze(-1)
            st_emb = st_emb * keep
        h_node = h_node + w_emb[batch] + st_emb[batch]

        h_node, _ = self.processor(h_node, edge_index, h_edge)
        return self.decoder(h_node)


# ──────────────────────────────────────────────────────────────────────────────
# Lightning training module
# ──────────────────────────────────────────────────────────────────────────────


class LitRelaxML(L.LightningModule):
    """Lightning training wrapper for the physics-feature variant.

    Hyperparameter signature is a superset of model_shelltgt.LitRelaxML —
    adds ``species_hidden`` (the MLP hidden width inside SpeciesEncoder).
    Defaults match what produced sensible cross-polymorph behavior in
    the baseline model.
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
        species_hidden: int = 128,
        shell_target_species_dim: int = 8,
        shell_target_hidden: int = 64,
        shell_target_dropout: float = 0.0,
        # Auxiliary bond-length loss weight.  When > 0, _shared_step
        # adds ``aux_bond_weight * mean(((‖new_dxyz‖ - target_r) /
        # sigma)²)`` over edges whose species pair has a shell_target
        # entry, where ``new_dxyz`` is the predicted post-step edge
        # vector and ``target_r``/``sigma`` come from
        # shell_pair_features.  This directly supervises shell_target
        # use at the bond-length level — independent of how the model
        # internally encodes shell_target (this v1 path uses the
        # per-graph ShellTargetEncoder which the slope diagnostic
        # showed was inert at slope=0; aux loss provides supervision
        # that should force the deep-set encoder to actually carry
        # the signal).  See RELAXML_SESSION.txt §13.5/13.6.
        aux_bond_weight: float = 0.0,
        ema_decay: float = 0.9999,
        learn_rate: float = 1e-3,
        lr_schedule: str = "cosine",
        lr_min_ratio: float = 0.01,
        warmup_steps: int = 500,
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
            species_hidden=species_hidden,
            shell_target_species_dim=shell_target_species_dim,
            shell_target_hidden=shell_target_hidden,
            shell_target_dropout=shell_target_dropout,
        )

        _ema = float(ema_decay)
        ema_avg = lambda avg_p, p, num_avg: _ema * avg_p + (1 - _ema) * p
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model, avg_fn=ema_avg,
        )
        # EMA parameters are deep-copied from self.model and updated
        # manually via ``.data.copy_`` in ``optimizer_step`` — they never
        # receive gradients.  DDP would otherwise flag them as "unused
        # parameters" and refuse to run; setting requires_grad=False
        # excludes them from DDP's reducer registration cleanly.
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self.aux_bond_weight = float(aux_bond_weight)
        self.learn_rate = learn_rate
        self.lr_schedule = lr_schedule
        self.lr_min_ratio = lr_min_ratio
        self.warmup_steps = warmup_steps

    def _aux_bond_length_loss(self, batch, pred) -> Tensor:
        """Predicted-vs-target bond-length penalty on edges with a
        shell_target entry.

        For each directed edge ``(src, dst)`` whose species pair has a
        shell_target row, compute the predicted post-step distance
        ``‖dx_old + (pred_dst - pred_src)‖`` and penalize its deviation
        from ``target_r``, normalized by the pair's ``sigma``.

        v1's RelaxMLModel doesn't expose per-edge target_r/sigma in its
        forward path (it uses the per-graph deep-set ShellTargetEncoder).
        We reuse ``shelltgt_phys_v2.edge_features.build_per_edge_shell_target``
        purely as a *lookup utility* — it computes ``target_r`` and
        ``sigma`` per edge from ``shell_pair_species`` and
        ``shell_pair_features``, no learnable parameters.  v1's
        conditioning pathway is still the deep-set encoder; only the
        aux supervision uses the per-edge lookup.

        Returns a scalar (mean over targeted edges).  Returns 0 if no
        edges in the batch have a shell_target entry.
        """
        if pred.shape[0] == 0 or batch.edge_index.shape[1] == 0:
            return pred.new_zeros(())

        # Local import to avoid a hard dependency of v1 on v2 at module
        # load — aux loss is opt-in via aux_bond_weight > 0.
        from tricor.relaxml.shelltgt_phys_v2.edge_features import (
            SIGMA_SCALE, TARGET_R_SCALE, build_per_edge_shell_target,
        )

        num_graphs = int(batch.w.shape[0])
        per_edge = build_per_edge_shell_target(
            z=batch.z,
            edge_index=batch.edge_index,
            batch=batch.batch,
            pair_species=batch.shell_pair_species,
            pair_features=batch.shell_pair_features,
            pair_batch=batch.shell_pair_batch,
            num_graphs=num_graphs,
            max_z=int(self.hparams.get("max_z", DEFAULT_MAX_Z)),
            dropout=0.0,
            training=False,
        )

        target_r = per_edge[:, 0] * TARGET_R_SCALE
        sigma = per_edge[:, 1] * SIGMA_SCALE
        has_target = per_edge[:, 4]

        src, dst = batch.edge_index[0], batch.edge_index[1]
        # edge_attr[:, :3] is the rotated old src→dst displacement;
        # pred[src] / pred[dst] are per-atom displacements in the same
        # rotated frame.  Distance is rotation-invariant.
        new_dxyz = batch.edge_attr[:, :3] + (pred[dst] - pred[src])
        new_r = new_dxyz.norm(dim=-1)

        sigma_safe = sigma.clamp(min=1e-3)
        err = (new_r - target_r) / sigma_safe
        n_targeted = has_target.sum().clamp(min=1.0)
        return (err.pow(2) * has_target).sum() / n_targeted

    def _shared_step(self, batch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute (total_loss, displacement_loss, aux_bond_loss, zero_baseline).

        ``total_loss`` is what backprop runs on.  The other three are
        returned for logging so we can monitor each term independently.
        """
        pred = self.model(
            batch.z, batch.edge_index, batch.edge_attr,
            batch.w, batch.batch,
            batch.shell_pair_species, batch.shell_pair_features,
            batch.shell_pair_batch,
            batch.shell_trip_species, batch.shell_trip_features,
            batch.shell_trip_batch,
        )
        per_atom_sq = (pred - batch.target_displacement).pow(2).sum(dim=-1)
        displacement_loss = per_atom_sq.mean()
        zero_baseline = batch.target_displacement.pow(2).sum(dim=-1).mean()

        if self.aux_bond_weight > 0.0:
            aux_bond_loss = self._aux_bond_length_loss(batch, pred)
        else:
            aux_bond_loss = pred.new_zeros(())

        total_loss = displacement_loss + self.aux_bond_weight * aux_bond_loss
        return total_loss, displacement_loss, aux_bond_loss, zero_baseline

    def training_step(self, batch, batch_idx):
        total, disp, aux, zero_baseline = self._shared_step(batch)
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        rel = disp / zero_baseline.clamp(min=1e-12)
        # sync_dist averages epoch-level scalars across DDP ranks before
        # logging; without it each rank logs its own values and only
        # rank 0's land in TensorBoard.  Tiny per-step NCCL cost.
        self.log("train_loss", total, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs, sync_dist=True)
        self.log("train_disp_loss", disp, on_step=False, on_epoch=True,
                 prog_bar=False, batch_size=bs, sync_dist=True)
        self.log("train_aux_bond_loss", aux, on_step=False, on_epoch=True,
                 prog_bar=False, batch_size=bs, sync_dist=True)
        self.log("train_relative_loss", rel, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs, sync_dist=True)
        return total

    def validation_step(self, batch, batch_idx):
        total, disp, aux, zero_baseline = self._shared_step(batch)
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        rel = disp / zero_baseline.clamp(min=1e-12)
        self.log("val_loss", total, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs, sync_dist=True)
        self.log("val_disp_loss", disp, on_step=False, on_epoch=True,
                 prog_bar=False, batch_size=bs, sync_dist=True)
        self.log("val_aux_bond_loss", aux, on_step=False, on_epoch=True,
                 prog_bar=False, batch_size=bs, sync_dist=True)
        self.log("val_relative_loss", rel, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs, sync_dist=True)
        return total

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)
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
