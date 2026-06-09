"""Per-edge shell_target injection variant of the relaxml surrogate.

The v1 ``shelltgt_phys`` model encoded shell_target as a per-graph
deep-set vector that was added to every node embedding.  Perturbation
diagnostics showed that the deep-set encoder was inert (slope=0 on
both pair target_r and triplet angle perturbations on the dropout=0.2
checkpoint).  The underlying cause: with classifier-free-guidance
dropout active during training, the encoder could minimize loss by
outputting a constant — making "dropout-on" and "dropout-off" batches
indistinguishable.  See ``RELAXML_SESSION.txt`` §4.5.

The v2 fix moves *pair* shell_target features (target_r, sigma,
coordination numbers) onto each graph edge whose species pair matches a
shell_target entry.  The edge encoder MLP — already responsible for
producing the geometric / species edge embedding — gets direct numeric
access to the bond-length and coordination targets it should be
predicting toward.  The per-graph deep-set encoder stays for *triplet*
features only (angle modes + mass weights), since triplet info doesn't
map cleanly to a single edge.

Scope of this v2 step
---------------------
**Pair-only**.  We test the per-edge injection hypothesis on bond
lengths first.  Triplet conditioning still goes through a per-graph
deep-set (``TripletTargetEncoder``) which the v1 angle diagnostic
showed was inert at slope=0.  After v2 trains, re-run the pair
perturbation diagnostic on the new checkpoint — if pair slope > 0,
the architectural fix worked; then decide separately whether and how
to fix the angle channel based on whatever the angle diagnostic shows
on v2.

What's identical to ``shelltgt_phys.model``
-------------------------------------------
  * Data signature (re-uses ``RelaxMLDataModule`` unchanged)
  * NodeEncoder, WeightEncoder, Processor, Decoder, Lightning wrapper
  * SpeciesEncoder shared MLP for all three species lookup sites
  * Forward output shape and meaning

What changes
------------
  * EdgeEncoder gains ``NUM_PER_EDGE_FEATURES`` extra input channels
    fed by ``build_per_edge_shell_target``.
  * ``TripletTargetEncoder`` replaces ``ShellTargetEncoder``: it
    encodes only the triplet half of the deep-set (pairs no longer go
    through it).
  * ``shell_target_dropout`` is applied PER GRAPH at the per-edge
    feature step (and also to the triplet encoder output, for the same
    CFG semantics as v1).

Hyperparameter names mirror v1 wherever the architectural meaning is
the same, so most training-config dicts port over without changes.
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
from ..shell_target import NUM_TRIPLET_FEATURES
from ..shelltgt_phys.species import SpeciesEncoder
from .edge_features import (
    NUM_PER_EDGE_FEATURES,
    SIGMA_SCALE,
    TARGET_R_SCALE,
    build_per_edge_shell_target,
)


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
        self.embed = SpeciesEncoder(out_dim=node_dim, hidden_dim=species_hidden)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, z: Tensor) -> Tensor:
        return self.norm(self.embed(z))


class EdgeEncoder(nn.Module):
    """[edge_attr, src_pair_emb, dst_pair_emb, per_edge_shell_target] -> edge embedding.

    The shell_target features are appended to the edge encoder input so
    the MLP has direct numeric access to the bond-length / coordination
    targets it's supposed to drive predictions toward.
    """

    def __init__(
        self,
        edge_dim: int,
        species_pair_dim: int,
        init_geom_dim: int = 4,
        per_edge_shell_dim: int = NUM_PER_EDGE_FEATURES,
    ) -> None:
        super().__init__()
        in_dim = init_geom_dim + 2 * species_pair_dim + per_edge_shell_dim
        self.embed = nn.Sequential(
            MLP([in_dim, edge_dim, edge_dim], act=nn.SiLU()),
            nn.LayerNorm(edge_dim),
        )

    def forward(
        self,
        edge_attr: Tensor,
        src_pair: Tensor,
        dst_pair: Tensor,
        per_edge_shell: Tensor,
    ) -> Tensor:
        return self.embed(
            torch.cat([edge_attr, src_pair, dst_pair, per_edge_shell], dim=-1)
        )


class WeightEncoder(nn.Module):
    """Global weight-parameter vector -> node-space embedding.  Unchanged from v1."""

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


class TripletTargetEncoder(nn.Module):
    """Deep-set encoder for the *triplet* half of shell_target.

    Pairs go directly to edges (see ``edge_features.py``); only triplets
    (angle modes + mass weights) are still encoded as a per-graph deep
    set and broadcast.  Same SpeciesEncoder-backed species lookup as v1.
    """

    def __init__(
        self,
        out_dim: int,
        species_emb_dim: int = 8,
        species_hidden: int = 128,
        triplet_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.species_emb = SpeciesEncoder(
            out_dim=species_emb_dim, hidden_dim=species_hidden,
        )
        trip_in = 3 * species_emb_dim + NUM_TRIPLET_FEATURES
        self.triplet_mlp = nn.Sequential(
            MLP([trip_in, triplet_hidden, triplet_hidden], act=nn.SiLU()),
        )
        self.out = nn.Sequential(
            MLP([triplet_hidden, out_dim, out_dim], act=nn.SiLU()),
            nn.LayerNorm(out_dim),
        )

    @_dynamo.disable
    def forward(
        self,
        trip_species: Tensor,
        trip_features: Tensor,
        trip_batch: Tensor,
        num_graphs: int,
    ) -> Tensor:
        # shell_triplet_species rows are (Z_centre, Z_nbr_1, Z_nbr_2);
        # this matches the convention used by tricor.shells and the
        # diagnose_shell_target_angle_conditioning.py diagnostic.
        za = self.species_emb(trip_species[:, 0])
        zb = self.species_emb(trip_species[:, 1])
        zc = self.species_emb(trip_species[:, 2])
        h_trip = self.triplet_mlp(torch.cat([za, zb, zc, trip_features], dim=-1))
        trip_pool = scatter(
            h_trip, trip_batch, dim=0, dim_size=num_graphs, reduce="sum",
        )
        return self.out(trip_pool)


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
    """Step-by-step relaxation surrogate, v2 architecture.

    Pair shell_target features feed the edge encoder directly; triplet
    features go through a per-graph deep-set encoder.  Forward inputs
    and outputs are identical to ``shelltgt_phys.RelaxMLModel``, so the
    same data layer (``RelaxMLDataModule``) and iterative-inference
    code work unchanged.
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
        self.shell_target_dropout = float(shell_target_dropout)

        self.node_encoder = NodeEncoder(
            max_z=max_z, node_dim=node_dim, species_hidden=species_hidden,
        )
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
        self.triplet_target_encoder = TripletTargetEncoder(
            out_dim=node_dim,
            species_emb_dim=shell_target_species_dim,
            species_hidden=species_hidden,
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

        num_graphs = int(w.shape[0])

        # Per-edge shell_target features.  CFG dropout (per graph) is
        # applied inside the lookup so that the entire (5,) vector goes
        # to zero for dropout-picked graphs.
        per_edge_shell = build_per_edge_shell_target(
            z=z,
            edge_index=edge_index,
            batch=batch,
            pair_species=pair_species,
            pair_features=pair_features,
            pair_batch=pair_batch,
            num_graphs=num_graphs,
            max_z=self.max_z,
            dropout=self.shell_target_dropout,
            training=self.training,
        )

        h_edge = self.edge_encoder(
            edge_attr, pair_emb[src], pair_emb[dst], per_edge_shell,
        )

        w_emb = self.weight_encoder(w)
        trip_emb = self.triplet_target_encoder(
            trip_species, trip_features, trip_batch, num_graphs,
        )

        # Per-graph CFG dropout on the triplet conditioning, mirroring
        # the original ShellTargetEncoder dropout semantics.  Per-edge
        # dropout has already been applied above; we use an
        # independent draw here so the model also learns to handle "no
        # triplet info" cases.
        if self.training and self.shell_target_dropout > 0.0:
            keep = (
                torch.rand(num_graphs, device=trip_emb.device)
                > self.shell_target_dropout
            ).to(trip_emb.dtype).unsqueeze(-1)
            trip_emb = trip_emb * keep
        h_node = h_node + w_emb[batch] + trip_emb[batch]

        h_node, _ = self.processor(h_node, edge_index, h_edge)
        return self.decoder(h_node)


# ──────────────────────────────────────────────────────────────────────────────
# Lightning training module
# ──────────────────────────────────────────────────────────────────────────────


class LitRelaxML(L.LightningModule):
    """Lightning training wrapper for the v2 (per-edge injection) model.

    Hyperparameter signature matches ``shelltgt_phys.LitRelaxML`` so
    porting training configs between v1 and v2 is just an import swap.
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
        # vector and ``target_r`` / ``sigma`` come from
        # shell_pair_features.  Directly supervises shell_target use
        # at the bond-length level — the v2 pair diagnostic on the
        # initial coord_phys_rinject checkpoint showed slope ≈ 0
        # (per-edge channel structurally present but downstream layers
        # underweight it), so this aux loss is the Branch A fix.  See
        # RELAXML_SESSION.txt §13.5/13.6.
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

        Returns a scalar (mean over targeted edges).  Returns 0 if no
        edges in the batch have a shell_target entry (or if there are
        no edges at all, which shouldn't happen but is guarded).

        The lookup uses ``dropout=0.0`` regardless of the model's
        CFG-dropout setting — aux supervision needs the *true*
        target_r for every targeted edge in every graph, not the
        dropout-zeroed version the conditioning channel sees.
        """
        if pred.shape[0] == 0 or batch.edge_index.shape[1] == 0:
            return pred.new_zeros(())

        num_graphs = int(batch.w.shape[0])
        per_edge = build_per_edge_shell_target(
            z=batch.z,
            edge_index=batch.edge_index,
            batch=batch.batch,
            pair_species=batch.shell_pair_species,
            pair_features=batch.shell_pair_features,
            pair_batch=batch.shell_pair_batch,
            num_graphs=num_graphs,
            max_z=self.model.max_z,
            dropout=0.0,
            training=False,
        )  # (E, NUM_PER_EDGE_FEATURES)

        target_r = per_edge[:, 0] * TARGET_R_SCALE  # unscale to Å
        sigma = per_edge[:, 1] * SIGMA_SCALE         # unscale to Å
        has_target = per_edge[:, 4]                  # 0/1

        src, dst = batch.edge_index[0], batch.edge_index[1]
        # Predicted post-step edge vector in the data's (possibly
        # rotated) frame.  edge_attr[:, :3] is the rotated old
        # displacement src→dst; pred[src]/pred[dst] are per-atom
        # displacements in the same rotated frame.  Distance is
        # rotation-invariant, so this is consistent with the un-rotated
        # geometry.
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
        ``aux_bond_loss`` is always returned even when
        ``aux_bond_weight=0`` (in which case it's a 0-tensor), so the
        logging path is shape-stable.
        """
        pred = self.model(
            batch.z, batch.edge_index, batch.edge_attr,
            batch.w, batch.batch,
            batch.shell_pair_species, batch.shell_pair_features,
            batch.shell_pair_batch,
            batch.shell_trip_species, batch.shell_trip_features,
            batch.shell_trip_batch,
        )
        displacement_loss = (
            pred - batch.target_displacement
        ).pow(2).sum(dim=-1).mean()
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
