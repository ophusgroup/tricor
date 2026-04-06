"""Monte Carlo supercell scaffolding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from ase.atoms import Atoms

from .g3 import G3Distribution


class Supercell:
    """Dummy supercell optimizer driven by a target G3Distribution."""

    def __init__(
        self,
        distribution: G3Distribution,
        cell_dims: int | Sequence[int] | None = None,
        *,
        cell_dim: int | Sequence[int] | None = None,
        label: str | None = None,
        rng_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a prototype supercell optimizer.

        Parameters
        ----------
        distribution
            Target distribution that the eventual Monte Carlo engine will try to
            match.
        cell_dims, cell_dim
            Supercell replication factors. A single integer produces a cubic
            replication, while a length-3 sequence specifies `(na, nb, nc)`.
        label
            Human-readable label for summaries and repr output.
        rng_seed
            Optional random seed used by the placeholder Monte Carlo driver.
        **kwargs
            Extra metadata stored for future Monte Carlo options.
        """
        if cell_dims is None:
            cell_dims = cell_dim
        if cell_dims is None:
            raise ValueError("cell_dims is required.")

        self.distribution = distribution
        self.cell_dims = self._normalize_cell_dims(cell_dims)
        self.label = label or "supercell"
        self.metadata: dict[str, Any] = dict(kwargs)
        self.rng = np.random.default_rng(rng_seed)
        self.atoms = self._build_initial_atoms()
        self.mc_history: dict[str, np.ndarray] | None = None
        self.best_score: float | None = None
        self.last_temperature: float | None = None

    def _normalize_cell_dims(self, cell_dims: int | Sequence[int]) -> tuple[int, int, int]:
        """Validate and normalize the requested supercell dimensions."""
        if isinstance(cell_dims, int):
            if cell_dims <= 0:
                raise ValueError("cell_dims must be positive.")
            return (cell_dims, cell_dims, cell_dims)

        dims = tuple(int(value) for value in cell_dims)
        if len(dims) != 3 or any(value <= 0 for value in dims):
            raise ValueError("cell_dims must be an int or a length-3 sequence of positive integers.")
        return dims

    def _build_initial_atoms(self) -> Atoms | None:
        """Construct the repeated starting structure used by the prototype engine."""
        if self.distribution.atoms is None:
            return None
        return self.distribution.atoms.repeat(self.cell_dims)

    def monte_carlo(
        self,
        num_steps: int = 1_000,
        temperature: float = 1.0,
        *,
        swap_fraction: float = 0.1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a synthetic Monte Carlo history for API prototyping.

        Parameters
        ----------
        num_steps
            Number of Monte Carlo steps to simulate in the placeholder history.
        temperature
            Effective temperature controlling the decay rate of the synthetic
            score trace.
        swap_fraction
            Fraction of steps treated as attempted atom swaps when reporting
            summary statistics.
        **kwargs
            Additional Monte Carlo options preserved in the returned summary.

        Returns
        -------
        dict[str, Any]
            Summary of the synthetic run, including score statistics and swap
            counts.
        """
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.distribution._ensure_plot_data()

        steps = np.arange(num_steps + 1, dtype=np.int32)
        decay_length = max(num_steps / (4.0 * temperature), 1.0)
        raw_score = 0.8 * np.exp(-steps / decay_length) + 0.2
        noise = self.rng.normal(loc=0.0, scale=0.015, size=steps.size)
        score = np.clip(raw_score + noise, 0.0, None)
        best_score = np.minimum.accumulate(score)

        acceptance = np.clip(
            0.65 * np.exp(-steps / max(num_steps / 2.0, 1.0)) * temperature,
            0.05,
            0.95,
        )
        attempted_swaps = int(round(num_steps * swap_fraction))
        accepted_swaps = int(round(attempted_swaps * float(np.mean(acceptance))))

        self.mc_history = {
            "step": steps,
            "score": score.astype(np.float32),
            "best_score": best_score.astype(np.float32),
            "acceptance": acceptance.astype(np.float32),
        }
        self.best_score = float(best_score[-1])
        self.last_temperature = float(temperature)

        summary = {
            "num_steps": num_steps,
            "temperature": float(temperature),
            "attempted_swaps": attempted_swaps,
            "accepted_swaps": accepted_swaps,
            "best_score": self.best_score,
            "cell_dims": self.cell_dims,
        }
        if kwargs:
            summary["mc_kwargs"] = dict(kwargs)
        return summary

    def __repr__(self) -> str:
        atom_count = None if self.atoms is None else len(self.atoms)
        return (
            f"Supercell(label={self.label!r}, cell_dims={self.cell_dims}, "
            f"atoms={atom_count}, best_score={self.best_score})"
        )
