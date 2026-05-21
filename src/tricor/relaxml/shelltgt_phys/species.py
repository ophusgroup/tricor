"""Physics-feature periodic-table encoder for relaxml.

Drop-in replacement for ``nn.Embedding(MAX_Z, dim)`` lookups that uses a
fixed periodic-table feature table fed through a small shared MLP.  The
key win over learned-from-Z embeddings is parameter sharing: every
gradient step from any Si-bearing graph updates the same MLP weights
that produce the embedding for N — so a model trained on Si + SiC + SiO2
+ BN + AlN can produce a meaningful embedding for the never-seen-paired
(Si, N) edge in Si3N4.

Why this matters
----------------
With ``nn.Embedding(120, dim)``, each element row receives gradient only
from training graphs that contain that element.  An element absent from
training stays at random init.  Sharing one MLP across all 120 element
rows fixes that: gradient from any Si-bearing graph updates the same
MLP weights that produce the embedding for N (via the shared
physics-feature input).  See ``RELAXML_SESSION.txt`` for the motivating
Si3N4 cross-composition failure.

Data source — pymatgen
----------------------
All elemental properties come from
``pymatgen.core.periodic_table.Element``, the canonical materials-science
periodic-table source.  This means:

  * **No hand-typed tables.**  Off-by-one alignment risks are eliminated.
  * **Coverage through Og (Z=118)** wherever pymatgen has values.
  * **Authoritative values.**  Pauling EN, ionization energy, electron
    affinity, etc. all come from one consistent source.

For the super-heavy synthetics (Rf..Og, Z=104..118) and a handful of
other elements where a measurement / theoretical value doesn't exist
(noble-gas EA etc.), pymatgen returns ``None``.  We do **not** fabricate
a value via group / period extrapolation — that would put guessed
chemistry into the gradient stream as if it were measured fact.
Instead, missing values are left as **zero in z-scored space** (which
equals the column mean post-z-score).  This is the most honest
"information absent" signal that fits in a continuous channel: no
unique encoding for "missing", but no wrong-flavoured guess either.

This means the model cannot strictly distinguish "EN=column-mean"
from "EN=missing" on a single continuous channel.  In practice that
ambiguity is fine for our use case (every realistic CIF in MPtrj is
Z ≤ ~96 where pymatgen has full data) and a separate ``is_known``
mask channel can be added later if it ever bites.

A hard self-consistency check (``_validate_table``) runs at module
construction and asserts that every Z=1..118 row has finite,
in-range values, and that the block / period / group one-hots sum to
exactly 1.  ImportError → broken environment.  ValueError → broken
table.  Either way the training script fails loudly at startup rather
than silently producing garbage.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from graphite.nn import MLP


# Sized to 120 to match DEFAULT_MAX_Z elsewhere.  Row 0 is the "no element"
# placeholder (treated as unknown), rows 1..118 hold real elements, row
# 119 is padding (also imputed).
MAX_Z: int = 120


# ──────────────────────────────────────────────────────────────────────────────
# Periodic-table queries (pymatgen)
# ──────────────────────────────────────────────────────────────────────────────


def _import_pymatgen():
    """Late import of pymatgen with a clear error message if missing."""
    try:
        from pymatgen.core.periodic_table import Element
    except ImportError as e:
        raise ImportError(
            "tricor.relaxml.shelltgt_phys.species requires pymatgen "
            "(`pip install pymatgen`).  This is the authoritative source "
            "of periodic-table data the SpeciesEncoder builds its feature "
            "table from."
        ) from e
    return Element


# Block index lookup, shared across helpers.  's', 'p', 'd', 'f' from
# pymatgen.Element.block; '?' for unknown / Z=0 placeholder.
_BLOCK_INDEX = {"s": 0, "p": 1, "d": 2, "f": 3, "?": 0}
N_BLOCK = 4


def _safe_get(elem, attr: str) -> Optional[float]:
    """Return ``float(getattr(elem, attr))`` or None if the value is
    missing / non-numeric.  pymatgen sometimes returns FloatWithUnit
    objects, sometimes plain floats, sometimes None."""
    try:
        v = getattr(elem, attr)
    except (AttributeError, ValueError, KeyError):
        return None
    if v is None:
        return None
    try:
        # FloatWithUnit and Quantity-like objects implement __float__.
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _atomic_radius(elem) -> Optional[float]:
    """Best-available atomic radius in Å.  Tries calculated atomic
    radius first (Slater values, complete through Z=96), then atomic
    radius (empirical, sparse for super-heavies), then covalent
    radius."""
    for attr in ("atomic_radius_calculated", "atomic_radius"):
        v = _safe_get(elem, attr)
        if v is not None:
            return v
    # Last resort: covalent radius (single-bond, comprehensive).
    try:
        v = elem.atomic_radius
        return float(v) if v is not None else None
    except Exception:
        return None


def _ionization_energy(elem) -> Optional[float]:
    """First ionization energy in eV.  pymatgen stores ionization energies
    on the Element object; first IE is ``ionization_energies[0]`` or
    the legacy ``ionization_energy`` scalar."""
    # pymatgen 2024+ exposes a list; older versions had a scalar attr.
    for attr in ("ionization_energy", "first_ionization_energy"):
        v = _safe_get(elem, attr)
        if v is not None:
            return v
    try:
        ies = getattr(elem, "ionization_energies", None)
        if ies and len(ies) > 0 and ies[0] is not None:
            f = float(ies[0])
            if np.isfinite(f):
                return f
    except Exception:
        pass
    return None


def _electron_affinity(elem) -> Optional[float]:
    """Electron affinity in eV.  Returns None for elements where
    pymatgen has no value (most noble gases, alkaline earths, super-
    heavies)."""
    return _safe_get(elem, "electron_affinity")


def _electronegativity(elem) -> Optional[float]:
    """Pauling electronegativity (dimensionless)."""
    # pymatgen returns NaN (not None) for elements without a Pauling
    # value; _safe_get filters NaN via np.isfinite.
    v = _safe_get(elem, "X")
    return v


def _atomic_volume(elem) -> Optional[float]:
    """Molar volume in cm³/mol."""
    return _safe_get(elem, "molar_volume")


def _atomic_mass(elem) -> Optional[float]:
    """Atomic mass in amu."""
    return _safe_get(elem, "atomic_mass")


def _group(elem) -> int:
    """Group number 1..18.  Lanthanides and actinides are placed in
    group 3 by pymatgen convention."""
    try:
        g = int(elem.group)
        return g if 1 <= g <= 18 else 0
    except Exception:
        return 0


def _period(elem) -> int:
    """Period 1..7 (pymatgen calls it ``row``)."""
    try:
        p = int(elem.row)
        return p if 1 <= p <= 7 else 0
    except Exception:
        return 0


def _block(elem) -> str:
    """'s', 'p', 'd', or 'f'."""
    try:
        b = str(elem.block).lower()
        return b if b in _BLOCK_INDEX else "?"
    except Exception:
        return "?"


def _valence_electrons(elem) -> int:
    """Number of valence electrons.

    Defined precisely:
      * Main-group s-block (groups 1-2): equals group number.
      * Main-group p-block (groups 13-18): equals group - 10.
      * d-block transition metals (groups 3-12): equals group number
        (= n_s + n_(n-1)d valence count in the canonical "valence
        electron" definition for transition metals).
      * f-block (lanthanides, actinides): set to the count of
        f-electrons in the configured ground state plus 2 outer-shell
        s-electrons.  Approximate but consistent.
      * Group 18 noble gases: returns 8 (or 2 for He) per chemistry
        convention.
    """
    g = _group(elem)
    block = _block(elem)
    period = _period(elem)
    if block == "s":
        return g
    if block == "p":
        # He sits in group 18 with block='s' in pymatgen's layout, so we
        # only see group 13-18 here.
        return max(0, g - 10)
    if block == "d":
        # 3..12.  Group number matches s + d valence count in the
        # standard chemistry convention.
        return g
    if block == "f":
        # f-block: count f-electrons (1..14 across the row) + 2 for ns².
        # period=6 -> 4f, period=7 -> 5f.  Lanthanides start at La (Z=57).
        if period == 6:
            n_f = max(0, elem.Z - 56)  # La (Z=57) → 1, Lu (Z=71) → 15
        elif period == 7:
            n_f = max(0, elem.Z - 88)  # Ac (Z=89) → 1, Lr (Z=103) → 15
        else:
            n_f = 0
        return min(n_f, 14) + 2
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Build the feature table
# ──────────────────────────────────────────────────────────────────────────────


# Order of columns in the final feature vector.  Kept stable so
# checkpoints saved with one version of this module don't silently
# misalign when re-loaded.  Total = 10 scalars + 4 block + 7 period
# + 18 group = 39 features.
SCALAR_COLUMNS: Tuple[str, ...] = (
    "atomic_number_norm",
    "group_ordinal",
    "period_ordinal",
    "valence_electrons",
    "atomic_mass",
    "covalent_radius",
    "electronegativity",
    "ionization_energy",
    "electron_affinity",
    "atomic_volume",
)
N_PERIOD = 7
N_GROUP = 18
N_FEATURES: int = len(SCALAR_COLUMNS) + N_BLOCK + N_PERIOD + N_GROUP


def _collect_raw_properties() -> dict:
    """Pull raw per-element properties from pymatgen for Z = 1..MAX_Z-1.

    Returns a dict of length-MAX_Z float arrays (Z=0 is set to NaN so
    the imputation step picks a sensible default).  ``None`` values
    from pymatgen are passed through as NaN.
    """
    Element = _import_pymatgen()
    n = MAX_Z
    out = {
        "atomic_mass":       np.full(n, np.nan, dtype=np.float64),
        "atomic_radius":     np.full(n, np.nan, dtype=np.float64),
        "electronegativity": np.full(n, np.nan, dtype=np.float64),
        "ionization_energy": np.full(n, np.nan, dtype=np.float64),
        "electron_affinity": np.full(n, np.nan, dtype=np.float64),
        "atomic_volume":     np.full(n, np.nan, dtype=np.float64),
        "group":             np.zeros(n, dtype=np.int64),
        "period":            np.zeros(n, dtype=np.int64),
        "block":             np.zeros(n, dtype=np.int64),
        "valence":           np.zeros(n, dtype=np.int64),
    }
    for z in range(1, n):
        try:
            elem = Element.from_Z(z)
        except Exception:
            # Z out of pymatgen's known range — treat as unknown.
            continue
        m = _atomic_mass(elem)
        if m is not None:
            out["atomic_mass"][z] = m
        r = _atomic_radius(elem)
        if r is not None:
            out["atomic_radius"][z] = r
        en = _electronegativity(elem)
        if en is not None:
            out["electronegativity"][z] = en
        ie = _ionization_energy(elem)
        if ie is not None:
            out["ionization_energy"][z] = ie
        ea = _electron_affinity(elem)
        if ea is not None:
            out["electron_affinity"][z] = ea
        v = _atomic_volume(elem)
        if v is not None:
            out["atomic_volume"][z] = v
        out["group"][z] = _group(elem)
        out["period"][z] = _period(elem)
        out["block"][z] = _BLOCK_INDEX.get(_block(elem), 0)
        out["valence"][z] = _valence_electrons(elem)
    return out


def _zscore_known_zero_unknown(values: np.ndarray) -> np.ndarray:
    """Z-score the entries that pymatgen knows; leave unknown entries at 0.

    Statistics (mean, std) are computed over **known values only** so
    pymatgen's ``None``s don't pull the distribution.  Known entries
    are then mapped to ``(x - mu_known) / sigma_known``.  Unknown
    entries (originally NaN) become exactly 0.0 in z-scored space —
    the same numeric value an element with the column mean would get.

    This is the "let data be sparse where it does not exist" policy:
    no fabricated chemistry, no nearest-group fallback, just an
    explicit zero in the slot.  The model cannot uniquely distinguish
    "this element has the column-mean value" from "this element's
    value is unknown", but no fake value enters the gradient.
    """
    known = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float32)
    if not known.any():
        return out
    known_vals = values[known]
    mu = float(known_vals.mean())
    sigma = float(known_vals.std())
    if sigma < 1e-8:
        sigma = 1.0
    out[known] = ((values[known] - mu) / sigma).astype(np.float32)
    return out


def _validate_table(table: np.ndarray) -> None:
    """Hard load-time assertion that every Z=1..118 has a finite,
    in-range feature row.  Raises ValueError with a precise message if
    any column has out-of-range values."""
    if not np.isfinite(table).all():
        bad_z = np.where(~np.isfinite(table).all(axis=1))[0].tolist()
        raise ValueError(
            f"periodic-table feature table has NaN/Inf rows at Z={bad_z}"
        )
    # Z-scored scalars must have moderate range.  Use a generous cap (10σ)
    # so genuine outliers (e.g. H's IE) pass but indexing bugs producing
    # huge values would fail.
    n_scalars = len(SCALAR_COLUMNS)
    scalar_cols = table[:, :n_scalars]
    max_abs = float(np.abs(scalar_cols).max())
    if max_abs > 10.0:
        col, row = np.unravel_index(int(np.abs(scalar_cols).argmax()),
                                    scalar_cols.shape)
        raise ValueError(
            f"periodic-table scalar column {SCALAR_COLUMNS[row]} for "
            f"Z={col} has |z-score|={max_abs:.2f} > 10 — likely "
            f"alignment or unit error"
        )
    # Each block / period / group one-hot row must sum to 0 or 1
    # (Z=0 is all zeros; Z=1..118 is exactly one).
    n_oh = N_BLOCK + N_PERIOD + N_GROUP
    oh_cols = table[:, n_scalars:n_scalars + n_oh]
    # Expect block, period, group one-hots each to be exactly one for
    # Z=1..118.  Sum across each of the three sub-blocks separately.
    block_sums = oh_cols[:, :N_BLOCK].sum(axis=1)
    period_sums = oh_cols[:, N_BLOCK:N_BLOCK + N_PERIOD].sum(axis=1)
    group_sums = oh_cols[:, N_BLOCK + N_PERIOD:].sum(axis=1)
    for name, sums in (("block", block_sums), ("period", period_sums),
                       ("group", group_sums)):
        bad = np.where((sums[1:119] != 1.0))[0] + 1  # check Z=1..118
        if len(bad) > 0:
            raise ValueError(
                f"periodic-table {name} one-hot is not exactly 1 for "
                f"Z={bad.tolist()[:10]}"
            )


def build_periodic_table_features() -> np.ndarray:
    """Return a (MAX_Z, N_FEATURES) z-scored periodic-table table.

    Columns:
        0..(len(SCALAR_COLUMNS)-1):   continuous scalars (z-scored)
        len(SCALAR_COLUMNS)..+3:      block one-hot     (s, p, d, f)
        ...+7:                        period one-hot    (1..7)
        ...+18:                       group one-hot     (1..18)

    Validated at construction time — raises ValueError if any row has
    NaN, Inf, or implausibly large z-scored values.
    """
    raw = _collect_raw_properties()

    z_arr = np.arange(MAX_Z, dtype=np.float64)

    # Categorical-or-derivable properties: well-defined for every
    # Z=1..118 from the periodic-table layout (pymatgen knows all of
    # them).  z-score over the whole array.  z=0 placeholder is the
    # only "missing" entry; it stays close to mean post-z-score.
    groups   = raw["group"].astype(np.float64)
    periods  = raw["period"].astype(np.float64)
    blocks   = raw["block"].astype(np.int64)
    valences = raw["valence"].astype(np.float64)
    # Convert Z=0 group/period/valence to NaN so they don't pull the
    # z-score distribution — they get 0 post-z-score (the "missing"
    # signal).
    groups_in   = np.where(groups   > 0, groups,   np.nan)
    periods_in  = np.where(periods  > 0, periods,  np.nan)
    valences_in = np.where(valences > 0, valences, np.nan)

    # Continuous physics properties: pymatgen returns None for many
    # super-heavy synthetics (Z=104..118) and a handful of others (e.g.
    # noble-gas EA).  z-score uses stats from known values only;
    # unknown entries stay at 0 in z-scored space (= column mean post
    # z-score).  No nearest-group fallback, no fabricated values.

    scalars = np.stack([
        _zscore_known_zero_unknown(z_arr),
        _zscore_known_zero_unknown(groups_in),
        _zscore_known_zero_unknown(periods_in),
        _zscore_known_zero_unknown(valences_in),
        _zscore_known_zero_unknown(raw["atomic_mass"]),
        _zscore_known_zero_unknown(raw["atomic_radius"]),
        _zscore_known_zero_unknown(raw["electronegativity"]),
        _zscore_known_zero_unknown(raw["ionization_energy"]),
        _zscore_known_zero_unknown(raw["electron_affinity"]),
        _zscore_known_zero_unknown(raw["atomic_volume"]),
    ], axis=-1).astype(np.float32)

    block_oh = np.zeros((MAX_Z, N_BLOCK), dtype=np.float32)
    block_oh[np.arange(MAX_Z), blocks.clip(0, N_BLOCK - 1)] = 1.0
    # Z=0 placeholder: zero out its block-one-hot so the validator
    # (which expects exactly 1 for Z=1..118 only) sees Z=0 as all
    # zeros across all three one-hot sub-blocks.
    block_oh[0] = 0.0

    period_oh = np.zeros((MAX_Z, N_PERIOD), dtype=np.float32)
    p_idx = (periods.astype(np.int64) - 1)
    valid_p = (p_idx >= 0) & (p_idx < N_PERIOD)
    period_oh[np.arange(MAX_Z)[valid_p], p_idx[valid_p]] = 1.0

    group_oh = np.zeros((MAX_Z, N_GROUP), dtype=np.float32)
    g_idx = (groups.astype(np.int64) - 1)
    valid_g = (g_idx >= 0) & (g_idx < N_GROUP)
    group_oh[np.arange(MAX_Z)[valid_g], g_idx[valid_g]] = 1.0

    table = np.concatenate([scalars, block_oh, period_oh, group_oh], axis=-1)
    if table.shape != (MAX_Z, N_FEATURES):
        raise ValueError(
            f"periodic-table feature table has wrong shape "
            f"{table.shape}, expected ({MAX_Z}, {N_FEATURES})"
        )
    _validate_table(table)
    return table


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────


class SpeciesEncoder(nn.Module):
    """Atomic-number Z -> embedding via shared MLP over physics features.

    Drop-in replacement for ``nn.Embedding(MAX_Z, out_dim)`` with
    parameter sharing across all 120 elements.  Forward signature
    matches ``nn.Embedding``: takes ``(N,)`` long Z and returns
    ``(N, out_dim)``.

    Args:
        out_dim: output embedding dimensionality.
        hidden_dim: MLP hidden width.  128 is plenty for the 39-D
            input; deeper/wider tends to overfit on 120 rows.
        n_features: width of the periodic-table feature table.
            Defaults to ``N_FEATURES``.
    """

    def __init__(
        self,
        out_dim: int,
        hidden_dim: int = 128,
        n_features: int = N_FEATURES,
    ) -> None:
        super().__init__()
        table = torch.from_numpy(build_periodic_table_features()).float()
        if table.shape[1] != n_features:
            raise ValueError(
                f"periodic table built with {table.shape[1]} features but "
                f"SpeciesEncoder was constructed with n_features={n_features}"
            )
        # Frozen buffer so the table travels with the model state dict
        # (and inference is reproducible without re-running
        # build_periodic_table_features).
        self.register_buffer("table", table, persistent=True)
        self.mlp = nn.Sequential(
            MLP([n_features, hidden_dim, out_dim], act=nn.SiLU()),
        )

    def forward(self, z: Tensor) -> Tensor:
        # Clamp to valid range so an out-of-range Z doesn't index past
        # the buffer.  Anything ≥ MAX_Z gets the row at MAX_Z-1.
        z = z.clamp(min=0, max=MAX_Z - 1)
        return self.mlp(self.table[z])
