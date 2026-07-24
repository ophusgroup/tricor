"""Interactive explorer tying an HRTEM frame to the local g2 / g3 block target.

``HRTEMExplorer`` is an :mod:`anywidget` widget over a propagated
:class:`~atomode.ptycho.hrtem.ExitWaveStack`:

* **left** — the HRTEM intensity frame for the current thickness / defocus,
  a draggable circular window, and thickness / defocus / rotation sliders;
* **middle** — the windowed input the network sees (real space) and its
  diffractogram (log ``|FFT|``);
* **right** — the window-weighted g3 slice and g2 for the depth block
  ``[0, t]`` (all atoms the beam has crossed), sharing the radial axis.

Defocus is applied to the cached complex exit waves in numpy, so the
thickness / defocus / rotation sliders are cheap (no multislice re-run).
The g2 / g3 target depends only on thickness + window position (the
circular window makes it rotation-invariant and the defocus-independent),
so it is recomputed only when those change.
"""

from __future__ import annotations

import pathlib

import anywidget
import numpy as np
import traitlets

from .correlations import local_correlations
from .hrtem import default_defocus, hrtem_image, hrtem_input
from .weighting import WindowSpec, windowed_image

_STATIC = pathlib.Path(__file__).parent.parent / "static"

__all__ = ["HRTEMExplorer"]


class HRTEMExplorer(anywidget.AnyWidget):
    """Explore HRTEM block g2 / g3 targets against a thickness / defocus series."""

    _esm = _STATIC / "hrtem_explorer.js"
    _css = _STATIC / "hrtem_explorer.css"

    # --- HRTEM frame ---
    slice_values = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    slice_shape = traitlets.List(trait=traitlets.Int()).tag(sync=True)  # [nx, ny]
    extent = traitlets.List(trait=traitlets.Float()).tag(sync=True)  # [Lx, Ly] (Å)
    layout = traitlets.Unicode("side").tag(sync=True)  # "side" | "stacked"
    transpose = traitlets.Bool(False).tag(sync=True)  # long axis horizontal

    # --- thickness / defocus ---
    thicknesses = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    thickness_index = traitlets.Int(0).tag(sync=True)
    thickness = traitlets.Float(0.0).tag(sync=True)
    defocus_offset = traitlets.Float(0.0).tag(sync=True)
    defocus = traitlets.Float(0.0).tag(sync=True)

    # --- window ---
    window_cx = traitlets.Float(0.0).tag(sync=True)
    window_cy = traitlets.Float(0.0).tag(sync=True)
    window_side = traitlets.Float(20.0).tag(sync=True)
    window_angle = traitlets.Float(0.0).tag(sync=True)

    # --- input previews ---
    input_values = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    input_shape = traitlets.List(trait=traitlets.Int()).tag(sync=True)
    fft_values = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    fft_shape = traitlets.List(trait=traitlets.Int()).tag(sync=True)

    # --- fixed contrast ranges (computed once; histogram-editable in the UI) ---
    frame_clim = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    fft_clim = traitlets.List(trait=traitlets.Float()).tag(sync=True)

    # --- correlations ---
    r = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    phi_deg = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    g2 = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    g3_slice_values = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    g3_slice_shape = traitlets.List(trait=traitlets.Int()).tag(sync=True)  # [phi, r]
    nn_band = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    g3_vmax = traitlets.Float(-1.0).tag(sync=True)
    r_max = traitlets.Float(10.0).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        stack,
        atoms,
        *,
        pair_peak: float | None = None,
        r_max: float = 10.0,
        side: float | None = None,
        semiangle_cutoff: float | None = None,
        cs: float = 0.0,
        r_step: float = 0.1,
        phi_num_bins: int = 36,
        scattering_weighted: bool = True,
        defocus_offset: float = 0.0,
        layout: str | None = None,
        **kwargs,
    ):
        """Parameters
        ----------
        stack
            A propagated :class:`~atomode.ptycho.hrtem.ExitWaveStack`.
        atoms
            The structure the stack was propagated through.
        pair_peak
            Nearest-neighbour bond (Å) for the g3 r01 band.
        r_max, side
            Correlation radius and window diameter (Å); ``side`` defaults to
            ``2 * r_max``.
        semiangle_cutoff, cs
            Objective aperture (mrad) and spherical aberration (Å).
        defocus_offset
            Initial offset (Å) around the per-thickness centre defocus
            ``+t / 2``.
        """
        super().__init__(**kwargs)
        self._stack = stack
        self._atoms = atoms
        self._pair_peak = pair_peak
        self._r_step = r_step
        self._phi_num_bins = phi_num_bins
        self._semiangle = semiangle_cutoff
        self._cs = cs
        self._frame = None
        self._atom_scale = None
        if scattering_weighted:
            try:
                from .potential import scattering_power

                self._atom_scale = scattering_power(atoms.numbers)
            except Exception:  # noqa: BLE001
                self._atom_scale = None
        self._suspend = False

        lx, ly = stack.extent
        self.extent = [float(lx), float(ly)]
        # Wide / non-square cells: stack panels under a full-width frame with
        # the long axis horizontal.
        aspect = max(lx, ly) / max(min(lx, ly), 1e-9)
        stacked = (aspect > 2.0) if layout is None else (layout == "stacked")
        self.layout = "stacked" if stacked else "side"
        self.transpose = bool(stacked and ly > lx)
        self.thicknesses = [float(t) for t in stack.thicknesses]
        self.r_max = float(r_max)
        self.window_side = float(side) if side is not None else 2.0 * float(r_max)
        self.window_cx = float(lx) / 2.0
        self.window_cy = float(ly) / 2.0
        self.defocus_offset = float(defocus_offset)
        self.thickness_index = int(stack.n_thicknesses - 1)
        self._sync_thickness()

        self._update_frame()
        self._update_corr()
        self._update_input()
        self._init_contrast()

        self.observe(self._on_thickness, names="thickness_index")
        self.observe(self._on_defocus, names="defocus_offset")
        self.observe(self._on_window, names=["window_cx", "window_cy"])
        self.observe(self._on_angle, names="window_angle")

    # -- helpers ----------------------------------------------------------
    def _sync_thickness(self) -> None:
        ti = int(self.thickness_index)
        self.thickness = float(self._stack.thicknesses[ti])
        self.defocus = default_defocus(self.thickness) + float(self.defocus_offset)

    def _update_frame(self) -> None:
        img = hrtem_image(self._stack, int(self.thickness_index), float(self.defocus),
                          semiangle_cutoff=self._semiangle, cs=self._cs)
        self._frame = img
        self.slice_shape = [int(img.shape[0]), int(img.shape[1])]
        self.slice_values = img.ravel().tolist()

    def _update_corr(self) -> None:
        window = WindowSpec.block(
            (float(self.window_cx), float(self.window_cy)),
            float(self.window_side), 0.0, float(self.thickness),
        )
        try:
            res = local_correlations(
                self._atoms, window, r_max=float(self.r_max), r_step=self._r_step,
                phi_num_bins=self._phi_num_bins, pair_peak=self._pair_peak,
                atom_scale=self._atom_scale,
            )
        except Exception as exc:  # noqa: BLE001
            self.status = f"error: {exc}"
            return
        self.r = res.r.tolist()
        self.phi_deg = res.phi_deg.tolist()
        self.g2 = np.nan_to_num(res.g2, nan=0.0, posinf=0.0).tolist()
        self.g3_slice_shape = [int(res.g3_slice.shape[0]), int(res.g3_slice.shape[1])]
        self.g3_slice_values = np.nan_to_num(res.g3_slice, nan=0.0, posinf=0.0).ravel().tolist()
        self.nn_band = [float(res.nn_band[0]), float(res.nn_band[1])]
        self.status = (
            f"{res.n_window} atoms in block [0, {self.thickness:.0f}] Å · "
            f"defocus {self.defocus:+.0f} Å (mid-block {default_defocus(self.thickness):+.0f} "
            f"{self.defocus_offset:+.0f}) · window r={self.window_side / 2:.0f} Å · rot "
            f"{self.window_angle:.0f}°"
        )

    def _init_contrast(self) -> None:
        """Fixed contrast ranges from a contrast-rich reference (full thickness,
        underfocused off the mid-block focus), held constant as sliders move."""
        ti = self._stack.n_thicknesses - 1
        t = float(self._stack.thicknesses[ti])
        ref = hrtem_image(self._stack, ti, default_defocus(t) + 60.0,
                          semiangle_cutoff=self._semiangle, cs=self._cs)
        lo, hi = np.percentile(ref, [0.5, 99.5])
        h = max(1.0 - float(lo), float(hi) - 1.0) * 1.05
        self.frame_clim = [1.0 - h, 1.0 + h]  # symmetric about vacuum = 1
        im = windowed_image(ref, self._stack.sampling,
                            (float(self.window_cx), float(self.window_cy)),
                            float(self.window_side), angle_deg=0.0)
        fft = hrtem_input(im)["fft"]
        self.fft_clim = [float(np.percentile(fft, 2.0)), float(np.percentile(fft, 99.7))]

    def _update_input(self) -> None:
        im = windowed_image(
            self._frame, self._stack.sampling,
            (float(self.window_cx), float(self.window_cy)),
            float(self.window_side), angle_deg=float(self.window_angle),
        )
        self.input_shape = [int(im.shape[0]), int(im.shape[1])]
        self.input_values = im.ravel().tolist()
        fft = hrtem_input(im)["fft"]
        self.fft_shape = [int(fft.shape[0]), int(fft.shape[1])]
        self.fft_values = fft.ravel().tolist()

    # -- observers --------------------------------------------------------
    def _guard(self, fn) -> None:
        if self._suspend:
            return
        self._suspend = True
        try:
            fn()
        finally:
            self._suspend = False

    def _on_thickness(self, _change) -> None:
        def go():
            self._sync_thickness()
            self._update_frame()
            self._update_corr()
            self._update_input()
        self._guard(go)

    def _on_defocus(self, _change) -> None:
        def go():
            self._sync_thickness()
            self._update_frame()
            self._update_input()
            self._update_corr()  # refresh status line (defocus shown)
        self._guard(go)

    def _on_window(self, _change) -> None:
        def go():
            self._update_corr()
            self._update_input()
        self._guard(go)

    def _on_angle(self, _change) -> None:
        # Circular window -> target unchanged; only the input preview moves.
        def go():
            self._update_input()
        self._guard(go)
