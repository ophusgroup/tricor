"""Per-edge shell_target feature lookup for the v2 architecture.

Background
----------
The v1 ``shelltgt_phys`` model encoded shell_target as a per-graph
deep-set vector that was broadcast (added) to every node embedding.
Perturbation diagnostics on the dropout=0.2 checkpoint showed slope=0
in both pair (target_r) and triplet (angle) directions: the deep-set
encoder output was effectively constant across all perturbations, so
the network learned to ignore it (cf. RELAXML_SESSION.txt §4.5).

The v2 fix is to inject pair shell_target features (target_r, sigma,
coord numbers) *directly* into each graph edge whose species pair
matches the shell_target entry.  The bond-length / coordination MLP in
the edge encoder then has numeric access to "you should aim for r=1.78
Å here, with coord(Si around O) = 2" — no per-graph average, no
broadcast that gets averaged out, no learnable layer between data and
edge feature that can collapse to zero.

This module produces the (E, NUM_PER_EDGE_FEATURES) tensor that
``model.py`` concatenates to the edge encoder input.

Scope
-----
This is the **pair-only** v2 step.  The triplet (angle) channel keeps
its v1 per-graph deep-set encoder, which the angle diagnostic showed
was inert at slope=0 on the v1 checkpoint.  The deliberate plan: train
v2 with the per-edge bond-length fix alone, re-run the pair perturbation
diagnostic to confirm slope > 0, then decide whether to escalate the
angle conditioning fix (per-edge angle context or per-triplet message
passing) based on training results.

Schema produced by :func:`build_per_edge_shell_target`
------------------------------------------------------
For each edge (src, dst) with atomic numbers (z_src, z_dst):

    [0] target_r / TARGET_R_SCALE        — peak distance for the bond
    [1] target_sigma / SIGMA_SCALE       — peak width
    [2] n_src_to_dst / COORD_SCALE       — target #(dst-species around src-species)
    [3] n_dst_to_src / COORD_SCALE       — target #(src-species around dst-species)
    [4] has_target                       — 1.0 if a pair entry exists for
                                           (z_src, z_dst) in this graph,
                                           0.0 otherwise.  Lets the
                                           network distinguish "no
                                           spring is installed for this
                                           pair" from "small spring at
                                           r=0, sigma=0".

Scaling
-------
Raw shell features mix Å distances (~1.5-3), unitless sigmas
(~0.05-0.3) and small integers (~0-12).  We divide each by a fixed
constant so that all four numeric channels enter the edge encoder at
~unit scale.  Has_target is left at 0/1.  These scales match the rough
range seen in the .npz files used so far; they are not learned, so a
checkpoint trained with one set of scales must be evaluated with the
same set.  Bump the constants here only on a clean retraining run.

CFG dropout
-----------
``classifier_free_guidance_dropout`` is applied PER GRAPH (not per
edge): with probability ``p_drop`` an entire graph's per-edge features
get zeroed (and has_target -> 0).  This matches the per-graph dropout
behavior of ``shelltgt_phys.model`` and lets us train the model to
function with and without shell_target conditioning.

Implementation
--------------
For each batch we build an in-memory lookup table

    table : (num_graphs, max_z+1, max_z+1, NUM_PER_EDGE_FEATURES)

zero-initialized, then scatter the pair entries into both (Z_a, Z_b)
and (Z_b, Z_a) slots (with n_ab / n_ba swapped for the reverse
direction).  Per-edge lookup is one fancy-index into the table.
Memory cost at max_z=120: ~70 KB / graph / feature; at
NUM_PER_EDGE_FEATURES=5 and batch_size=8, ~2 MB — negligible.
"""

from __future__ import annotations

import torch
import torch._dynamo as _dynamo
from torch import Tensor


# Number of per-edge shell_target features (see schema above).  Exposed
# so model.py can size the edge encoder's first linear correctly.
NUM_PER_EDGE_FEATURES: int = 5


# Per-channel scales applied at lookup time so each numeric feature
# arrives at the edge encoder near unit magnitude.  Tweak only on a
# clean retraining run — a checkpoint trained with one set of scales
# requires the same scales at inference.
TARGET_R_SCALE: float = 2.5      # Å — typical bond range is 1.5–3.0
SIGMA_SCALE:    float = 0.15     # Å — typical width range is 0.05–0.3
COORD_SCALE:    float = 12.0     # max sensible coordination number


@_dynamo.disable
def build_per_edge_shell_target(
    z: Tensor,
    edge_index: Tensor,
    batch: Tensor,
    pair_species: Tensor,
    pair_features: Tensor,
    pair_batch: Tensor,
    num_graphs: int,
    max_z: int = 120,
    dropout: float = 0.0,
    training: bool = False,
) -> Tensor:
    """Build the (E, NUM_PER_EDGE_FEATURES) per-edge feature tensor.

    Parameters
    ----------
    z
        ``(N,)`` atomic numbers (long).
    edge_index
        ``(2, E)`` ``[src, dst]`` indices into ``z``.
    batch
        ``(N,)`` graph index per atom (PyG style).
    pair_species
        ``(P, 2)`` atomic numbers of each shell_target pair, in
        canonical (a <= b) order (see ``shell_target.py``).
    pair_features
        ``(P, 4)`` ``[target_r, sigma, n_ab, n_ba]`` per pair.
    pair_batch
        ``(P,)`` graph index per pair entry.  Each graph's pairs share
        the same graph id; PyG provides this via ``ShellTargetData``.
    num_graphs
        Total number of graphs in this batch.
    max_z
        Upper bound on atomic numbers.  120 covers all real elements.
    dropout
        Per-graph CFG dropout probability.  With probability ``dropout``
        each graph's per-edge features are zeroed out, including
        ``has_target``.  Only active when ``training=True``.
    training
        Whether the model is in training mode.  Passed in explicitly
        (rather than read off a module) so this function stays a pure
        utility.

    Returns
    -------
    Tensor of shape ``(E, NUM_PER_EDGE_FEATURES)``.  Features for edges
    whose species pair has no shell_target entry are all zero (including
    has_target).  Features for graphs picked by CFG dropout are also
    all zero.
    """
    device = z.device
    dtype = pair_features.dtype

    # (num_graphs, max_z+1, max_z+1, NUM_PER_EDGE_FEATURES) — zero-init
    # means "no shell_target entry for this (graph, z_src, z_dst)".
    table = torch.zeros(
        (num_graphs, max_z + 1, max_z + 1, NUM_PER_EDGE_FEATURES),
        dtype=dtype,
        device=device,
    )

    if pair_species.numel() > 0:
        g  = pair_batch.long()
        za = pair_species[:, 0].long()
        zb = pair_species[:, 1].long()

        target_r = pair_features[:, 0] / TARGET_R_SCALE
        sigma    = pair_features[:, 1] / SIGMA_SCALE
        n_ab     = pair_features[:, 2] / COORD_SCALE
        n_ba     = pair_features[:, 3] / COORD_SCALE
        ones     = torch.ones_like(target_r)

        # Canonical (Z_a, Z_b) entry.  Conventions:
        #   pair_features[:, 2] = n_ab = "#(species b around species a)"
        #   pair_features[:, 3] = n_ba = "#(species a around species b)"
        # For an edge whose src has species a and dst has species b,
        # "src→dst coord" = #(b around a) = n_ab.
        table[g, za, zb, 0] = target_r
        table[g, za, zb, 1] = sigma
        table[g, za, zb, 2] = n_ab          # n_src→dst when src=a, dst=b
        table[g, za, zb, 3] = n_ba          # n_dst→src when src=a, dst=b
        table[g, za, zb, 4] = ones

        # Reverse direction.  For an edge whose src has species b and
        # dst has species a, "src→dst coord" = #(a around b) = n_ba, so
        # the second / third columns swap.  When a == b this is a no-op
        # (n_ab == n_ba by symmetry of the coordination matrix).
        table[g, zb, za, 0] = target_r
        table[g, zb, za, 1] = sigma
        table[g, zb, za, 2] = n_ba          # n_src→dst when src=b, dst=a
        table[g, zb, za, 3] = n_ab          # n_dst→src when src=b, dst=a
        table[g, zb, za, 4] = ones

    # Optional per-graph CFG dropout.  Build a (num_graphs,) keep-mask
    # and use it to scale the table before the per-edge lookup so the
    # zero propagates through has_target as well.
    if training and dropout > 0.0:
        keep = (
            torch.rand(num_graphs, device=device) > dropout
        ).to(dtype).view(num_graphs, 1, 1, 1)
        table = table * keep

    src = edge_index[0]
    dst = edge_index[1]
    e_graph = batch[src]                    # (E,) -- src and dst share a graph
    return table[e_graph, z[src], z[dst]]   # (E, NUM_PER_EDGE_FEATURES)
