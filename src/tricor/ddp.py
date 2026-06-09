"""DDP training utilities — generic, reusable across tricor projects.

Adapted from quantem.core.ml.ddp.DDPMixin.  Provides:

  * DDPMixin              — mixin for any trainer class; handles process-group
                            init, device pinning, DDP wrapping, and
                            collective helpers (all-reduce, broadcast).
  * SizeBalancedDistributedSampler — DistributedSampler replacement that
                            balances per-rank batch workload by interleaving
                            samples sorted by a per-sample size key.  Useful
                            when sample compute cost varies and naive random
                            sampling causes straggler hangs / OOM spikes.
  * make_distributed_loader — convenience builder around DataLoader that
                            wires up the correct sampler given world_size.

Designed to be project-agnostic.  The trainer subclass owns the training
loop, model construction, optimizer config, etc.  The mixin only handles
distributed-training plumbing.

Launch pattern (torchrun, single or multi-node):

    srun -l torchrun --nnodes=$SLURM_JOB_NUM_NODES \\
         --nproc-per-node=$SLURM_GPUS_PER_NODE \\
         --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \\
         train_script.py

torchrun sets RANK / LOCAL_RANK / WORLD_SIZE / MASTER_* in each worker's
env, which DDPMixin.setup_distributed reads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler


# ─────────────────────────────────────────────────────────────────────────────
# Size-balanced sampler
# ─────────────────────────────────────────────────────────────────────────────


class SizeBalancedDistributedSampler(Sampler[int]):
    """DistributedSampler that interleaves by sorted per-sample size.

    Standard DistributedSampler shards uniformly at random, which means one
    rank can land an outlier-heavy batch (e.g. 10 of the biggest cells in a
    graph dataset) while others sit idle.  This sampler:

      1. Shuffles the full dataset (epoch-seeded) for ordering randomness.
      2. Sorts each bucket of (batch_size * num_replicas) adjacent shuffled
         samples by the size key.
      3. Within each sorted bucket, gives each rank one sample at stride
         num_replicas — every rank's batch contains ~one sample from each
         size slot of the bucket.

    Net effect: per-rank batch totals are within ~one-bucket-spread of each
    other every step, eliminating the straggler problem and reducing the
    worst-case OOM risk (because monster samples are distributed across
    ranks rather than concentrated on one).

    Parameters
    ----------
    sizes : Sequence[int] or np.ndarray
        One size key per sample, length == len(dataset).  For graph datasets
        this is typically n_atoms or n_edges per sample.
    num_replicas : int
        World size.
    rank : int
        Global rank of this process.
    batch_size : int
        Per-rank batch size — must match the DataLoader's batch_size.
    shuffle : bool, default True
        Whether to permute samples per epoch.  Pass False for val/test.
    seed : int, default 0
        Base seed for the per-epoch shuffle.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        num_replicas: int,
        rank: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.sizes        = np.asarray(sizes)
        self.num_replicas = int(num_replicas)
        self.rank         = int(rank)
        self.batch_size   = int(batch_size)
        self.shuffle      = shuffle
        self.seed         = seed
        self.epoch        = 0
        n = len(self.sizes)
        bucket = self.batch_size * self.num_replicas
        # Drop the tail so every bucket is full, mirroring drop_last=True.
        self._num_buckets = n // bucket
        self._num_samples = self._num_buckets * self.batch_size

    def __iter__(self):
        n = len(self.sizes)
        g = np.random.default_rng(self.seed + self.epoch) if self.shuffle else None
        idx = np.arange(n)
        if self.shuffle:
            idx = g.permutation(idx)

        bucket = self.batch_size * self.num_replicas
        out = []
        for start in range(0, self._num_buckets * bucket, bucket):
            chunk = idx[start:start + bucket]
            order = np.argsort(self.sizes[chunk])         # ascending size
            chunk_sorted = chunk[order]
            # Each rank picks every Nth sorted element starting at its rank.
            out.extend(chunk_sorted[self.rank::self.num_replicas].tolist())
        return iter(out)

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader helper
# ─────────────────────────────────────────────────────────────────────────────


def make_distributed_loader(
    dataset: Dataset,
    *,
    world_size: int,
    global_rank: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    sizes: Optional[Sequence[int]] = None,
    sampler_seed: int = 0,
    loader_cls=DataLoader,
    **extra_loader_kwargs,
) -> tuple[DataLoader, Optional[Sampler]]:
    """Construct a DataLoader with the right sampler for the world size.

    Choices:
      * world_size > 1, sizes is None      → DistributedSampler (random shard)
      * world_size > 1, sizes is not None  → SizeBalancedDistributedSampler
      * world_size == 1                     → no sampler (DataLoader shuffles)

    Returns (loader, sampler).  Sampler is None when world_size == 1.

    Parameters
    ----------
    sizes : optional sequence of per-sample size keys.  When provided, uses
        SizeBalancedDistributedSampler keyed on this.
    loader_cls : DataLoader class to use.  Default torch.utils.data.DataLoader;
        for PyG graph datasets, pass torch_geometric.loader.DataLoader.
    """
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if prefetch_factor is None and num_workers > 0:
        prefetch_factor = 4

    if world_size > 1:
        if sizes is not None:
            sampler = SizeBalancedDistributedSampler(
                sizes,
                num_replicas=world_size, rank=global_rank,
                batch_size=batch_size, shuffle=shuffle, seed=sampler_seed,
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size, rank=global_rank,
                shuffle=shuffle, drop_last=drop_last,
            )
        loader_shuffle = False
    else:
        sampler = None
        loader_shuffle = shuffle

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=loader_shuffle,
        pin_memory=pin_memory,
        drop_last=(drop_last if sampler is None else False),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    loader_kwargs.update(extra_loader_kwargs)

    return loader_cls(dataset, **loader_kwargs), sampler


# ─────────────────────────────────────────────────────────────────────────────
# DDPMixin
# ─────────────────────────────────────────────────────────────────────────────


class DDPMixin:
    """Mixin for trainer classes that use PyTorch DDP.

    A trainer class inheriting from DDPMixin should call ``setup_distributed()``
    early in its construction (or fit() entry).  After that, the mixin
    populates the following attributes:

      * self.device       — torch.device for this rank's GPU (or cpu)
      * self.world_size   — int, total number of ranks
      * self.global_rank  — int, this rank's global rank (0..world_size-1)
      * self.local_rank   — int, this rank's local rank on its node
      * self.is_main      — bool property, True iff global_rank == 0

    And exposes helpers:

      * distribute_model(model)       — wrap an nn.Module in DDP
      * all_reduce_mean(value)        — average a scalar across ranks
      * broadcast_bool(value)         — rank-0 → all-ranks bool broadcast
      * main_print(*args, **kwargs)   — print only on rank 0 (with flush=True)
      * barrier()                     — distributed barrier (no-op if not init)
      * save_on_main(obj, path)       — torch.save on rank 0, no-op elsewhere
      * cleanup()                     — destroy process group

    Detection: when ``RANK`` is in os.environ (set by torchrun), we initialize
    a process group.  Otherwise we configure a single-process run on cuda:0
    (or cpu).  This makes the same trainer class work for both
    multi-GPU torchrun launches and quick single-process debugging.
    """

    # Populated by setup_distributed.
    device:      torch.device
    world_size:  int
    global_rank: int
    local_rank:  int

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def setup_distributed(
        self,
        *,
        single_device: Optional[str | torch.device] = None,
        enable_tf32: bool = True,
    ) -> None:
        """Initialize the process group (if under torchrun) and pin the device.

        Idempotent — safe to call once at the top of __init__.
        """
        if "RANK" in os.environ:
            self.global_rank = int(os.environ["RANK"])
            self.local_rank  = int(os.environ["LOCAL_RANK"])
            self.world_size  = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            if not dist.is_initialized():
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(
                    backend=backend, init_method="env://",
                    device_id=self.device,
                )
        else:
            self.global_rank = 0
            self.local_rank  = 0
            self.world_size  = 1
            if torch.cuda.is_available():
                dev = torch.device("cuda:0") if single_device is None \
                      else torch.device(single_device)
                torch.cuda.set_device(dev.index if dev.index is not None else 0)
                self.device = dev
            else:
                self.device = torch.device("cpu")

        if enable_tf32 and self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def cleanup(self) -> None:
        """Tear down the process group.  Safe to call multiple times."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def is_main(self) -> bool:
        return self.global_rank == 0

    @property
    def is_distributed(self) -> bool:
        return dist.is_initialized() and self.world_size > 1

    # ── Model + dataloader plumbing ────────────────────────────────────────

    def distribute_model(
        self,
        model: nn.Module,
        *,
        find_unused_parameters: bool = False,
        broadcast_buffers: bool = False,
        bucket_cap_mb: int = 100,
        gradient_as_bucket_view: bool = True,
    ) -> nn.Module:
        """Move model to this rank's device and wrap in DDP if world_size > 1.

        Returns the wrapped module (or the bare module if single-process).
        """
        model = model.to(self.device)
        if self.is_distributed:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=find_unused_parameters,
                broadcast_buffers=broadcast_buffers,
                bucket_cap_mb=bucket_cap_mb,
                gradient_as_bucket_view=gradient_as_bucket_view,
            )
        return model

    def make_dataloader(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        pin_memory: bool = True,
        sizes: Optional[Sequence[int]] = None,
        sampler_seed: int = 0,
        loader_cls=DataLoader,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: Optional[int] = None,
        **extra_loader_kwargs,
    ) -> tuple[DataLoader, Optional[Sampler]]:
        """Thin wrapper around make_distributed_loader using this rank's world."""
        return make_distributed_loader(
            dataset,
            world_size=self.world_size,
            global_rank=self.global_rank,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            sizes=sizes,
            sampler_seed=sampler_seed,
            loader_cls=loader_cls,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            **extra_loader_kwargs,
        )

    # ── Collective helpers ─────────────────────────────────────────────────

    def all_reduce_mean(self, value: float) -> float:
        """Average a scalar float across all ranks.

        Single-process: returns the value unchanged.  Useful for reporting
        per-epoch losses that were accumulated rank-locally.
        """
        if not self.is_distributed:
            return float(value)
        t = torch.tensor([value], dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item() / self.world_size)

    def all_reduce_sum(self, value: float) -> float:
        """Sum a scalar float across all ranks."""
        if not self.is_distributed:
            return float(value)
        t = torch.tensor([value], dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())

    def broadcast_bool(self, value: bool) -> bool:
        """Broadcast rank-0's bool to all ranks."""
        if not self.is_distributed:
            return bool(value)
        t = torch.tensor([1 if value else 0], dtype=torch.int32, device=self.device)
        dist.broadcast(t, src=0)
        return bool(t.item())

    # ── I/O helpers ────────────────────────────────────────────────────────

    def main_print(self, *args, **kwargs) -> None:
        """print() that only fires on rank 0, with flush=True by default."""
        if self.is_main:
            kwargs.setdefault("flush", True)
            print(*args, **kwargs)

    def save_on_main(self, obj, path: str | Path) -> None:
        """torch.save on rank 0, no-op on other ranks.

        Note: caller should make sure obj's tensors are on a save-friendly
        device (typically CPU or the rank's GPU).
        """
        if self.is_main:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(obj, path)
