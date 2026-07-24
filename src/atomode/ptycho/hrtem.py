"""HRTEM exit-wave propagation and image formation for training pairs.

Propagate a plane wave **once** through a frozen-phonon ensemble of a
thick cell, grabbing the complex exit wave at several thicknesses, then
form HRTEM images by applying an objective-lens transfer (defocus) to the
cached exit waves in numpy.  Re-using the stored waves means a whole
thickness / defocus / frozen-phonon sweep costs a handful of FFTs — the
expensive multislice runs only once.

Verified abTEM 1.0.9 conventions (see tests / project notes)
------------------------------------------------------------
* The beam **enters at z = 0** and propagates toward ``+z``; the exit wave
  at thickness ``t`` has integrated exactly the atoms in the block
  ``z ∈ [0, t]``.  The matching g2 / g3 target therefore uses a hard depth
  block (:meth:`atomode.ptycho.WindowSpec.block`), *not* a Gaussian slab.
* ``Potential(exit_planes=...)`` returns the wave at several thicknesses in
  one pass; an explicit tuple must end on the final slice, so the tuple is
  validated and the requested planes are selected back by thickness.
* The contrast transfer is ``χ(k) = π λ Δf k² + ½ π Cs λ³ k⁴`` with
  transfer ``H = aperture · exp(+iχ)`` and image ``|ℱ⁻¹(ℱψ · H)|²`` —
  reproduces ``abtem.CTF`` to a relative MSE of ~1e-13.
* Frozen phonons are averaged **incoherently**: apply the lens to each
  config's complex exit wave, take ``|·|²``, then average the intensities.

Defocus convention (abTEM): the per-thickness centre defocus is ``+t / 2``
(focus on the middle of the block — least contrast; verified empirically as
a sharp minimum at exactly ``+0.5 t``); explore a range of offsets around it.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .weighting import windowed_image

__all__ = [
    "ExitWaveStack",
    "exit_wave_stack",
    "ctf_image",
    "hrtem_image",
    "default_defocus",
    "radial_average",
    "hrtem_input",
]


@dataclasses.dataclass
class ExitWaveStack:
    """Complex HRTEM exit waves from one multislice pass.

    Attributes
    ----------
    waves
        ``(n_frozen_phonons, n_thicknesses, nx, ny)`` complex64 exit waves.
    thicknesses
        ``(n_thicknesses,)`` sample thicknesses (Å) — the block ``[0, t]``
        each wave has propagated through.
    sampling
        In-plane pixel size ``(dx, dy)`` in Å.
    energy
        Beam energy (eV).
    cell_z
        Full cell thickness (Å).
    phonon_sigma
        Frozen-phonon RMS displacement (Å) used.
    """

    waves: np.ndarray
    thicknesses: np.ndarray
    sampling: tuple[float, float]
    energy: float
    cell_z: float
    phonon_sigma: float

    @property
    def n_frozen_phonons(self) -> int:
        return int(self.waves.shape[0])

    @property
    def n_thicknesses(self) -> int:
        return int(self.waves.shape[1])

    @property
    def extent(self) -> tuple[float, float]:
        return (self.waves.shape[2] * self.sampling[0], self.waves.shape[3] * self.sampling[1])

    def thickness_index(self, thickness: float) -> int:
        """Index of the stored plane nearest to ``thickness`` (Å)."""
        return int(np.argmin(np.abs(self.thicknesses - float(thickness))))


def exit_wave_stack(
    atoms,
    *,
    thicknesses=None,
    every: float = 20.0,
    sampling: float = 0.1,
    slice_thickness: float = 2.0,
    energy: float = 300e3,
    num_frozen_phonons: int = 4,
    phonon_sigma: float = 0.076,
    parametrization: str = "lobato",
    projection: str = "infinite",
    device: str = "cpu",
    seed: int = 0,
    show_progress: bool = True,
) -> ExitWaveStack:
    """Propagate once and grab complex exit waves at several thicknesses.

    Parameters
    ----------
    atoms
        ASE ``Atoms`` (orthorhombic, periodic) — a thick atomode supercell,
        e.g. 50 × 50 × 200 Å.
    thicknesses
        Explicit export thicknesses (Å).  If ``None``, every ``every`` Å
        from ``every`` to the full cell thickness.
    every
        Export spacing (Å) when ``thicknesses`` is ``None`` (2 nm default).
    sampling
        In-plane pixel size (Å); abTEM rounds it to integer gpts.
    slice_thickness
        Multislice depth step (Å).
    energy
        Beam energy (eV).
    num_frozen_phonons
        Frozen-phonon configurations (averaged incoherently downstream).
    phonon_sigma
        Frozen-phonon RMS displacement (Å) — ~0.076 Å for Si at 300 K.
    parametrization, projection, device
        Passed to :class:`abtem.Potential`.
    seed
        Frozen-phonon RNG seed.

    Returns
    -------
    ExitWaveStack
    """
    import abtem

    abtem.config.set({"local_diagnostics.progress_bar": bool(show_progress)})

    a = atoms.copy()
    a.pbc = True
    cell_z = float(np.diag(np.asarray(a.cell.array))[2])

    # Number of slices (metadata only — no multislice).
    probe = abtem.Potential(
        a, sampling=sampling, slice_thickness=slice_thickness,
        parametrization=parametrization, projection=projection, device=device,
    )
    n_slices = int(probe.num_slices)

    if thicknesses is None:
        thicknesses = np.arange(every, cell_z + 1e-6, every)
    thicknesses = np.asarray(thicknesses, dtype=float)

    # Requested slice indices (clamped to the cell).  abTEM's explicit-tuple
    # path needs the *final* slice to be an exit plane, so append it if the
    # request stops short; the extra plane is dropped after selection.
    req = sorted({min(max(int(round(t / slice_thickness)) - 1, 0), n_slices - 1) for t in thicknesses})
    idx_full = tuple(req) if req[-1] == n_slices - 1 else tuple(req) + (n_slices - 1,)

    fp = abtem.FrozenPhonons(
        a, num_configs=int(num_frozen_phonons), sigmas=float(phonon_sigma),
        seed=seed, ensemble_mean=False,
    )
    pot = abtem.Potential(
        fp, sampling=sampling, slice_thickness=slice_thickness,
        parametrization=parametrization, projection=projection,
        exit_planes=idx_full, device=device,
    )
    # normalize=False -> unit-amplitude plane wave (vacuum intensity 1).
    waves = abtem.PlaneWave(energy=energy, normalize=False).multislice(pot).compute()

    # Re-order axes to (frozen_phonon, thickness, x, y).
    fp_axis = th_axis = None
    real_axes = []
    th_vals = None
    for i, ax in enumerate(waves.axes_metadata):
        name = type(ax).__name__
        if name == "ThicknessAxis":
            th_axis, th_vals = i, np.asarray(ax.values, dtype=float)
        elif "FrozenPhonon" in name:
            fp_axis = i
        elif name == "RealSpaceAxis":
            real_axes.append(i)
    arr = np.asarray(waves.array)
    order = [ax for ax in (fp_axis, th_axis) if ax is not None] + real_axes
    arr = np.transpose(arr, order)
    if fp_axis is None:
        arr = arr[None]  # single (already-averaged) config

    # Select the planes nearest to the requested thicknesses, in order.
    sel = [int(np.argmin(np.abs(th_vals - t))) for t in thicknesses]
    arr = np.ascontiguousarray(arr[:, sel])
    thick_out = th_vals[sel]

    return ExitWaveStack(
        waves=arr.astype(np.complex64),
        thicknesses=np.asarray(thick_out, dtype=float),
        sampling=(float(waves.sampling[0]), float(waves.sampling[1])),
        energy=float(energy),
        cell_z=cell_z,
        phonon_sigma=float(phonon_sigma),
    )


def ctf_image(psi, sampling, energy, defocus, *, semiangle_cutoff=None, cs=0.0):
    """HRTEM image intensity from a complex exit wave via the lens transfer.

    ``χ(k) = π λ Δf k² + ½ π Cs λ³ k⁴``, transfer ``H = aperture·exp(+iχ)``,
    image ``|ℱ⁻¹(ℱψ·H)|²`` (validated against ``abtem.CTF``).  Operates on
    the last two axes, so ``psi`` may carry a leading frozen-phonon axis.

    Parameters
    ----------
    psi
        Complex exit wave ``(..., nx, ny)``.
    sampling
        Pixel size ``(dx, dy)`` in Å.
    energy
        Beam energy (eV).
    defocus
        Defocus Δf (Å), abTEM convention (positive = underfocus).
    semiangle_cutoff
        Objective-aperture cutoff (mrad).  ``None`` uses the 2/3-Nyquist
        antialiasing band (abTEM's propagated band).
    cs
        Spherical aberration Cs (Å).
    """
    from abtem.core.energy import energy2wavelength

    psi = np.asarray(psi)
    nx, ny = psi.shape[-2], psi.shape[-1]
    lam = float(energy2wavelength(energy))
    kx = np.fft.fftfreq(nx, d=sampling[0])
    ky = np.fft.fftfreq(ny, d=sampling[1])
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    chi = np.pi * lam * defocus * k2 + 0.5 * np.pi * cs * lam ** 3 * k2 ** 2
    H = np.exp(1j * chi)
    if semiangle_cutoff is not None:
        H = H * (lam * np.sqrt(k2) <= semiangle_cutoff * 1e-3)
    else:
        k_aa = (2.0 / 3.0) * 0.5 * min(1.0 / sampling[0], 1.0 / sampling[1])
        H = H * (np.sqrt(k2) <= k_aa)
    out = np.fft.ifft2(np.fft.fft2(psi, axes=(-2, -1)) * H, axes=(-2, -1))
    return out.real ** 2 + out.imag ** 2


def hrtem_image(stack: ExitWaveStack, thickness_index: int, defocus: float,
                *, semiangle_cutoff=None, cs=0.0) -> np.ndarray:
    """Full-frame HRTEM intensity at one thickness / defocus.

    Applies the lens to every frozen-phonon exit wave, then averages the
    intensities (incoherent thermal average).
    """
    psi = stack.waves[:, int(thickness_index)]  # (n_fp, nx, ny)
    img = ctf_image(psi, stack.sampling, stack.energy, defocus,
                    semiangle_cutoff=semiangle_cutoff, cs=cs)
    img = np.asarray(img.mean(axis=0), dtype=np.float64)  # incoherent FP average
    m = float(img.mean())  # vacuum -> 1 (also absorbs aperture power loss)
    return img / m if m > 1e-12 else img


def default_defocus(thickness: float) -> float:
    """Centre defocus for a thickness: focus on the mid-block (least contrast).

    Verified empirically: the minimum-contrast focus sits at ``+t / 2`` (the
    exit wave back-propagated by half the block to image its centroid).  The
    sign is **positive** in abTEM's convention — explore offsets around it.
    """
    return 0.5 * float(thickness)


def radial_average(image: np.ndarray, center=None, n_bins: int | None = None) -> np.ndarray:
    """Azimuthal average of ``image`` about ``center`` (default: image centre)."""
    image = np.asarray(image, dtype=np.float64)
    nx, ny = image.shape
    if center is None:
        center = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    ii, jj = np.mgrid[0:nx, 0:ny]
    rint = np.hypot(ii - center[0], jj - center[1]).astype(int)
    nb = int(n_bins) if n_bins else int(rint.max()) + 1
    total = np.bincount(rint.ravel(), image.ravel(), minlength=nb)[:nb]
    count = np.bincount(rint.ravel(), minlength=nb)[:nb]
    return total / np.maximum(count, 1)


def hrtem_input(image_real: np.ndarray) -> dict:
    """Input representations of a windowed HRTEM crop.

    Returns a dict with the real-space image, its diffractogram (log
    ``|FFT|`` of the mean-subtracted crop), and the radial averages of
    each — the candidate network inputs to compare.
    """
    img = np.asarray(image_real, dtype=np.float64)
    fft = np.fft.fftshift(np.abs(np.fft.fft2(img - img.mean())))
    fft = np.log1p(fft)
    return {
        "real": img,
        "fft": fft,
        "real_radial": radial_average(img),
        "fft_radial": radial_average(fft),
    }


def hrtem_window_image(stack: ExitWaveStack, thickness_index: int, defocus: float,
                       center, side, *, angle_deg=0.0, semiangle_cutoff=None, cs=0.0,
                       full_image=None) -> np.ndarray:
    """Circular-Hann-windowed HRTEM crop the network sees (input image).

    ``full_image`` (a precomputed :func:`hrtem_image`) is reused if given,
    so many window positions over one frame skip recomputing the lens.
    """
    if full_image is None:
        full_image = hrtem_image(stack, thickness_index, defocus,
                                 semiangle_cutoff=semiangle_cutoff, cs=cs)
    return windowed_image(full_image, stack.sampling, center, side, angle_deg=angle_deg)
