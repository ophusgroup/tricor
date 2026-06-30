"""DDP-aware variant of tricor.macerelax.model for multi-GPU training.

Architecturally identical to model.py.  The only changes are inside
LitRelaxML to make Lightning's training loop behave correctly under DDP:

  1. Every self.log() call in training_step / validation_step adds
     ``sync_dist=True`` so the metric is all-reduced across ranks before
     being shown in TensorBoard / consumed by EarlyStopping +
     ModelCheckpoint.  Without sync_dist=True, monitored metrics use
     only rank-0's local batches — usable but noisier and slightly
     wrong as a "global val_loss" signal.

  2. A brief note on EMA + DDP behavior in optimizer_step (no code
     change, just documenting the subtlety).

Use this from scripts/macerelax/train_perlmutter.py:

    from tricor.macerelax.model_perlmutter import LitRelaxML

For single-GPU training, the original model.py is unchanged and still
the right import — sync_dist=True has no effect on single-GPU runs but
adds a small per-step coordination cost, so keeping the variants
separate avoids paying it unnecessarily on the workstation.
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
from tricor.macerelax.shell_target import NUM_PAIR_FEATURES, NUM_TRIPLET_FEATURES


# Periodic table covers Z = 1..118; sized to 120 gives a small safety margin.
DEFAULT_MAX_Z: int = 120


# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules (identical to model.py)
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
    """[dx, dy, dz, r, src_pair_emb, dst_pair_emb] -> edge embedding."""

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
    """Global weight-parameter vector -> node-space embedding."""

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
    """Deep-set encoder for shell_target arrays."""

    def __init__(
        self,
        max_z: int,
        out_dim: int,
        species_emb_dim: int = 8,
        pair_hidden: int = 64,
        triplet_hidden: int = 64,
    ) -> None:
        super().__init__()
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
# Full model (identical to model.py)
# ──────────────────────────────────────────────────────────────────────────────


class RelaxMLModel(nn.Module):
    """Step-by-step relaxation surrogate with shell_target conditioning.

    See tricor.macerelax.model.RelaxMLModel for full docs — this is the
    same class re-exported under the perlmutter variant so the import
    sits next to the DDP-aware LitRelaxML in the same module.
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
        self.shell_target_dropout = float(shell_target_dropout)
        self.node_encoder = NodeEncoder(max_z, node_dim)
        self.pair_species_embed = nn.Embedding(max_z, species_pair_dim)
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
# Lightning training module — DDP-aware
# ──────────────────────────────────────────────────────────────────────────────


class LitRelaxML(L.LightningModule):
    """DDP-aware Lightning training wrapper for RelaxMLModel.

    The only behavioral difference from tricor.macerelax.model.LitRelaxML
    is sync_dist=True on every self.log() call.  Lightning needs that to
    all-reduce metrics across ranks so that:

      - tensorboard sees a global average rather than rank-0's local view,
      - EarlyStopping's patience counter triggers on the global val_loss
        minimum (not rank-0's noisy local one),
      - ModelCheckpoint saves the global-best checkpoint.

    EMA + DDP note (no code change required, just documenting):
      Under DDP, parameters are synchronized via gradient averaging in the
      backward pass.  After optimizer.step() all ranks have bit-identical
      parameters, so EMA updates (which read from those parameters) yield
      bit-identical EMA state on every rank.  Lightning's checkpoint logic
      saves only rank-0's state, which is the correct one.
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
        lr_schedule: str = "cosine",
        lr_min_ratio: float = 0.01,
        warmup_steps: int = 500,
        weight_decay: float = 0.0,
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
        rel = loss / zero_baseline.clamp(min=1e-12)
        # sync_dist=True: average the per-epoch metric across all ranks
        # before logging.  Without it, tensorboard would show only rank-0.
        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs, sync_dist=True)
        self.log("train_relative_loss", rel, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, zero_baseline = self._shared_step(batch)
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        rel = loss / zero_baseline.clamp(min=1e-12)
        # sync_dist=True is especially important here because val_loss is
        # what EarlyStopping + ModelCheckpoint monitor.  A local rank-0
        # view of val_loss would cause those callbacks to make decisions
        # on noisier, sample-biased values.
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs, sync_dist=True)
        self.log("val_relative_loss", rel, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # AdamW: decoupled weight decay.  With weight_decay=0, same as Adam.
        # estimated_stepping_batches is DDP-aware: Lightning divides the
        # dataset by world_size for the per-rank step count.  No manual
        # scaling needed here.
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
        # Under DDP: post-step parameters are bit-identical across ranks
        # (gradient sync in backward + same optimizer state on each rank),
        # so update_parameters yields identical EMA tensors everywhere.
        self.ema_model.update_parameters(self.model)
