"""Random supercell initialization and Monte Carlo scaffolding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from ase.atoms import Atoms

from .g3 import G3Distribution


class Supercell:
    """Random supercell scaffold driven by a target :class:`G3Distribution`."""

    def __init__(
        self,
        distribution: G3Distribution,
        cell_dim: int | Sequence[int],
        *,
        relative_density: float = 1.0,
        plot_g3_compare: bool = False,
        label: str | None = None,
        rng_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a random supercell with the target composition and density.

        Parameters
        ----------
        distribution
            Target distribution that the eventual Monte Carlo engine will try to
            match.
        cell_dim
            Supercell replication factors. A single integer produces a cubic
            replication, while a length-3 sequence specifies `(na, nb, nc)`.
        relative_density
            Total number density relative to the crystalline reference cell. A
            value of `1.0` preserves the crystalline density, while values below
            one expand the supercell volume isotropically.
        plot_g3_compare
            If `True`, immediately display an interactive comparison between the
            random supercell and the target distribution when running inside
            IPython/Jupyter.
        label
            Human-readable label for summaries and repr output.
        rng_seed
            Optional random seed for reproducible initialization and placeholder
            Monte Carlo traces.
        **kwargs
            Extra metadata stored for future Monte Carlo options.
        """
        if relative_density <= 0:
            raise ValueError("relative_density must be positive.")
        if distribution.atoms is None:
            raise ValueError("Supercell construction requires a distribution with source atoms.")

        self.target_distribution = distribution
        self.distribution = distribution
        self.target_distribution._ensure_plot_data()
        self.cell_dim = self._normalize_cell_dim(cell_dim)
        self.relative_density = float(relative_density)
        self.label = label or "supercell"
        self.metadata: dict[str, Any] = dict(kwargs)
        self.rng = np.random.default_rng(rng_seed)
        self.measure_r_max = float(self.target_distribution.r_max)
        self.measure_r_step = float(self.target_distribution.r_step)
        self.measure_phi_num_bins = int(self.target_distribution.phi_num_bins)

        self.reference_atoms = self._build_reference_atoms()
        self.atoms = self._build_random_atoms()
        self.current_distribution: G3Distribution | None = None
        self.mc_history: dict[str, np.ndarray] | None = None
        self.best_score: float | None = None
        self.last_temperature: float | None = None

        self.measure_g3(show_progress=True)

        if plot_g3_compare:
            self._display_compare_widget()

    def _normalize_cell_dim(self, cell_dim: int | Sequence[int]) -> tuple[int, int, int]:
        """Validate and normalize the requested supercell dimensions."""
        if isinstance(cell_dim, int):
            if cell_dim <= 0:
                raise ValueError("cell_dim must be positive.")
            return (cell_dim, cell_dim, cell_dim)

        dims = tuple(int(value) for value in cell_dim)
        if len(dims) != 3 or any(value <= 0 for value in dims):
            raise ValueError("cell_dim must be an int or a length-3 sequence of positive integers.")
        return dims

    def _build_reference_atoms(self) -> Atoms:
        """Repeat the crystalline reference cell to the requested supercell size."""
        return self.target_distribution.atoms.repeat(self.cell_dim)

    def _build_random_atoms(self) -> Atoms:
        """Construct a random-coordinate supercell at the requested relative density."""
        reference = self.reference_atoms
        scale_factor = self.relative_density ** (-1.0 / 3.0)
        cell = reference.cell.copy()
        cell[:] = reference.cell.array * scale_factor

        numbers = np.array(reference.numbers, copy=True)
        self.rng.shuffle(numbers)
        scaled_positions = self.rng.random((len(numbers), 3))

        atoms = Atoms(
            numbers=numbers,
            cell=cell,
            pbc=reference.pbc,
            scaled_positions=scaled_positions,
        )
        atoms.info["relative_density"] = self.relative_density
        return atoms

    def measure_g3(
        self,
        *,
        force: bool = False,
        show_progress: bool = False,
    ) -> G3Distribution:
        """Measure the current random supercell on the target distribution grid.

        The current implementation always uses the full target discretization so
        the supercell and target histograms can be compared bin-for-bin in raw
        count space during later Monte Carlo updates.

        Parameters
        ----------
        force
            If `True`, discard any cached measurement and recompute the current
            supercell distribution.
        show_progress
            If `True`, display a text progress bar while the supercell histogram
            is accumulated.
        """
        if self.current_distribution is not None and not force:
            return self.current_distribution

        measured = G3Distribution(self.atoms, label=f"{self.label}-measured")
        measured.measure_g3(
            r_max=self.measure_r_max,
            r_step=self.measure_r_step,
            phi_num_bins=self.measure_phi_num_bins,
            show_progress=show_progress,
            progress_label=f"Measuring g3 in {self.label}",
        )
        self.current_distribution = measured
        return measured

    def plot_g3_compare(
        self,
        pair: int | str = 0,
        *,
        normalize: bool = True,
    ):
        """Return an interactive comparison between the current supercell and target."""
        self.target_distribution._ensure_plot_data()
        current = self.measure_g3()
        pair_index = self.target_distribution._resolve_pair_index(pair)

        from .g3_compare_widget import G3CompareWidget

        return G3CompareWidget(
            current_distribution=current,
            target_distribution=self.target_distribution,
            triplet_index=pair_index,
            normalize=normalize,
            supercell_title=f"{self.label} g3 slice",
            target_title=f"{self.target_distribution.label} g3 slice",
            status_prefix=(
                f"density {self.relative_density:.3f} | "
                f"cell_dim {self.cell_dim[0]}x{self.cell_dim[1]}x{self.cell_dim[2]}"
            ),
        )

    def _display_compare_widget(self) -> None:
        """Display the comparison widget immediately when running in IPython."""
        try:
            from IPython.display import display
        except Exception:
            return
        display(self.plot_g3_compare())

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

        self.target_distribution._ensure_plot_data()
        self.measure_g3()

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
            "cell_dim": self.cell_dim,
            "relative_density": self.relative_density,
            "measure_r_max": self.measure_r_max,
            "measure_r_step": self.measure_r_step,
            "measure_phi_num_bins": self.measure_phi_num_bins,
            "num_atoms": len(self.atoms),
        }
        if kwargs:
            summary["mc_kwargs"] = dict(kwargs)
        return summary

    def __repr__(self) -> str:
        atom_count = len(self.atoms)
        return (
            f"Supercell(label={self.label!r}, cell_dim={self.cell_dim}, "
            f"atoms={atom_count}, relative_density={self.relative_density:.3f}, "
            f"measure_r_max={self.measure_r_max:.3f}, best_score={self.best_score})"
        )
