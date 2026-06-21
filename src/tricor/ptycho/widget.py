"""Interactive explorer tying a blurred potential slice to the local g2 / g3.

``PtychoExplorer`` is an :mod:`anywidget` widget with:

* **left** — a greyscale image of the (blurred) potential slice in
  radians, a depth-slice slider, a histogram of the slice values, and a
  draggable red Hann-window box with a size slider;
* **right** — the window-weighted g3 slice (top) and g2 (bottom) for the
  current window, sharing the radial axis, with a g3 colour-range control.

Moving / resizing the window or changing the slice recomputes the
weighted correlations in Python (the random-catalogue denominator is
cached, so each update is fast) and pushes the new arrays to the
front-end.  Computation happens on drag *release* to keep dragging
smooth.
"""

from __future__ import annotations

import pathlib

import anywidget
import numpy as np
import traitlets

from .correlations import local_correlations
from .weighting import WindowSpec, windowed_image

_STATIC = pathlib.Path(__file__).parent.parent / "static"

__all__ = ["PtychoExplorer"]


class PtychoExplorer(anywidget.AnyWidget):
    """Explore local g2 / g3 targets against a blurred potential slice."""

    _esm = _STATIC / "ptycho_explorer.js"
    _css = _STATIC / "ptycho_explorer.css"

    # --- left panel: potential slice ---
    slice_values = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    slice_shape = traitlets.List(trait=traitlets.Int()).tag(sync=True)  # [nx, ny]
    extent = traitlets.List(trait=traitlets.Float()).tag(sync=True)  # [Lx, Ly] (Å)
    slice_index = traitlets.Int(0).tag(sync=True)
    n_slices = traitlets.Int(1).tag(sync=True)
    z0 = traitlets.Float(0.0).tag(sync=True)
    hist_counts = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    hist_edges = traitlets.List(trait=traitlets.Float()).tag(sync=True)

    # --- window ---
    window_cx = traitlets.Float(0.0).tag(sync=True)
    window_cy = traitlets.Float(0.0).tag(sync=True)
    window_side = traitlets.Float(20.0).tag(sync=True)
    window_angle = traitlets.Float(0.0).tag(sync=True)  # input rotation (deg)

    # --- input preview: the rotated, circular-windowed crop the net sees ---
    input_values = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    input_shape = traitlets.List(trait=traitlets.Int()).tag(sync=True)  # [nx, ny]

    # --- right panels: correlations ---
    r = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    phi_deg = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    g2 = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    g3_slice_values = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    g3_slice_shape = traitlets.List(trait=traitlets.Int()).tag(sync=True)  # [phi, r]
    nn_band = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    g3_vmax = traitlets.Float(-1.0).tag(sync=True)  # <=0 -> auto
    r_max = traitlets.Float(10.0).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)
    layout = traitlets.Unicode("side").tag(sync=True)  # "side" | "stacked"
    transpose = traitlets.Bool(False).tag(sync=True)  # long axis horizontal

    def __init__(
        self,
        stack,
        atoms,
        *,
        pair_peak: float | None = None,
        r_max: float = 10.0,
        side: float | None = None,
        sigma_z: float | None = None,
        r_step: float = 0.1,
        phi_num_bins: int = 36,
        scattering_weighted: bool = True,
        layout: str | None = None,
        **kwargs,
    ):
        """Parameters
        ----------
        stack
            A (blurred) :class:`~tricor.ptycho.potential.PotentialStack`.
        atoms
            The same ASE structure the stack was built from.
        pair_peak
            Nearest-neighbour bond distance (Å) for the g3 r01 band.
        r_max
            Fixed maximum correlation radius (Å).  The window is fixed for
            the session (no size slider).
        side
            Hann window side length (Å).  Defaults to ``2 * r_max``.
        sigma_z
            Depth weight width (Å) for the g2 / g3 target.  Defaults to
            the stack's applied depth blur so the input and target agree.
        r_max, r_step, phi_num_bins
            Correlation binning (``r_max`` should be ``<= side / 2``;
            ``phi_num_bins`` defaults to 36 = 5° for the sparse local case).
        scattering_weighted
            Weight atoms by their per-species scattering power so the
            target matches the scattering-weighted potential (default).
        """
        super().__init__(**kwargs)
        self._stack = stack
        self._atoms = atoms
        self._pair_peak = pair_peak
        self._r_step = r_step
        self._phi_num_bins = phi_num_bins
        self._atom_scale = None
        if scattering_weighted:
            try:
                from .potential import scattering_power

                self._atom_scale = scattering_power(atoms.numbers)
            except Exception:  # noqa: BLE001 - fall back to unweighted
                self._atom_scale = None
        self._sigma_z = (
            sigma_z
            if sigma_z is not None
            else (stack.blur[1] if stack.blur else 15.0)
        )
        self._suspend = False

        lx, ly = stack.extent
        self.extent = [float(lx), float(ly)]
        # Wide / non-square cells: stack panels under a full-width slice and
        # put the long axis horizontal.
        aspect = max(lx, ly) / max(min(lx, ly), 1e-9)
        stacked = (aspect > 2.0) if layout is None else (layout == "stacked")
        self.layout = "stacked" if stacked else "side"
        self.transpose = bool(stacked and ly > lx)
        self.n_slices = int(stack.n_slices)
        self.r_max = float(r_max)
        self.window_side = float(side) if side is not None else 2.0 * float(r_max)
        self.window_cx = float(lx) / 2.0
        self.window_cy = float(ly) / 2.0
        self.slice_index = int(stack.n_slices // 2)

        self._update_slice()
        self._update_corr()
        self._update_input()

        self.observe(self._on_slice, names="slice_index")
        self.observe(self._on_window, names=["window_cx", "window_cy"])
        self.observe(self._on_angle, names="window_angle")

    # -- payload builders -------------------------------------------------
    def _update_slice(self) -> None:
        img = np.asarray(self._stack.array[int(self.slice_index)], dtype=np.float64)
        self.slice_shape = [int(img.shape[0]), int(img.shape[1])]
        self.slice_values = img.ravel().tolist()
        self.z0 = float(self._stack.z_centers[int(self.slice_index)])
        counts, edges = np.histogram(img.ravel(), bins=48)
        self.hist_counts = counts.astype(float).tolist()
        self.hist_edges = edges.astype(float).tolist()

    def _update_corr(self) -> None:
        window = WindowSpec(
            center_xy=(float(self.window_cx), float(self.window_cy)),
            side=float(self.window_side),
            z0=float(self.z0),
            sigma_z=float(self._sigma_z),
        )
        try:
            res = local_correlations(
                self._atoms,
                window,
                r_max=float(self.r_max),
                r_step=self._r_step,
                phi_num_bins=self._phi_num_bins,
                pair_peak=self._pair_peak,
                atom_scale=self._atom_scale,
            )
        except Exception as exc:  # noqa: BLE001 - surface to the UI
            self.status = f"error: {exc}"
            return
        self.r = res.r.tolist()
        self.phi_deg = res.phi_deg.tolist()
        self.g2 = np.nan_to_num(res.g2, nan=0.0, posinf=0.0).tolist()
        self.g3_slice_shape = [int(res.g3_slice.shape[0]), int(res.g3_slice.shape[1])]
        self.g3_slice_values = np.nan_to_num(
            res.g3_slice, nan=0.0, posinf=0.0
        ).ravel().tolist()
        self.nn_band = [float(res.nn_band[0]), float(res.nn_band[1])]
        self.status = (
            f"{res.n_window} atoms · z0={self.z0:.1f} Å · "
            f"window {self.window_side:.0f} Å (circular) · "
            f"σz={self._sigma_z:.0f} Å · rot {self.window_angle:.0f}°"
        )

    def _update_input(self) -> None:
        """The rotated, circular-windowed crop the network would see."""
        img = windowed_image(
            self._stack.array[int(self.slice_index)],
            self._stack.sampling,
            (float(self.window_cx), float(self.window_cy)),
            float(self.window_side),
            angle_deg=float(self.window_angle),
        )
        self.input_shape = [int(img.shape[0]), int(img.shape[1])]
        self.input_values = img.ravel().tolist()

    # -- observers --------------------------------------------------------
    def _on_slice(self, _change) -> None:
        if self._suspend:
            return
        self._suspend = True
        try:
            self._update_slice()
            self._update_corr()
            self._update_input()
        finally:
            self._suspend = False

    def _on_window(self, _change) -> None:
        if self._suspend:
            return
        self._suspend = True
        try:
            self._update_corr()
            self._update_input()
        finally:
            self._suspend = False

    def _on_angle(self, _change) -> None:
        # The circular window makes the target rotation-invariant, so only
        # the input preview changes — g2 / g3 stay put.
        if self._suspend:
            return
        self._suspend = True
        try:
            self._update_input()
            self.status = (
                f"z0={self.z0:.1f} Å · window {self.window_side:.0f} Å (circular) · "
                f"σz={self._sigma_z:.0f} Å · rot {self.window_angle:.0f}°"
            )
        finally:
            self._suspend = False
