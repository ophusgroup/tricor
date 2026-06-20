"""Pseudo-ptychographic input: blurred, depth-resolved projected potential.

The model input is *not* a real ptychographic reconstruction but a cheap
proxy: abTEM's projected electrostatic potential, scaled to phase
(radians) and blurred to the anisotropic resolution a ptychographic
reconstruction would have — fine in-plane (``sigma_xy`` ≈ 0.2–1.0 Å),
coarse in depth (``sigma_z`` ≈ 10–40 Å, set by the probe depth of
field).  Picking one depth slice ``z0`` from the depth-blurred stack is
equivalent to weighting each atom's contribution by ``G(z - z0)`` with
that same ``sigma_z`` — exactly the depth weight the matching g2 / g3
target uses (:mod:`tricor.ptycho.correlations`).

Requires the optional ``abtem`` dependency (``pip install -e '.[training]'``).
A realistic, non-Gaussian probe-CTF depth kernel is a planned follow-up;
the Gaussian here is the v1 proxy.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["PotentialStack", "potential_stack", "blur_stack", "scattering_power"]


@dataclasses.dataclass
class PotentialStack:
    """A depth-resolved projected potential.

    Attributes
    ----------
    array
        ``(n_slices, nx, ny)`` projected potential.  In radians (phase)
        when built with ``to_radians=True``, else eV·Å.
    sampling
        In-plane pixel size ``(dx, dy)`` in Å.
    slice_thickness
        Depth slice spacing ``dz`` in Å.
    energy
        Beam energy in eV (used for the eV→rad interaction parameter).
    units
        ``"rad"`` or ``"eV*A"``.
    blur
        ``(sigma_xy, sigma_z)`` in Å already applied, or ``None`` if raw.
    """

    array: np.ndarray
    sampling: tuple[float, float]
    slice_thickness: float
    energy: float
    units: str = "rad"
    blur: tuple[float, float] | None = None

    @property
    def n_slices(self) -> int:
        return int(self.array.shape[0])

    @property
    def z_centers(self) -> np.ndarray:
        """Depth (Å) of each slice centre."""
        return (np.arange(self.n_slices) + 0.5) * self.slice_thickness

    @property
    def extent(self) -> tuple[float, float]:
        """In-plane extent ``(Lx, Ly)`` in Å."""
        return (self.array.shape[1] * self.sampling[0], self.array.shape[2] * self.sampling[1])

    def slice_index_for_z(self, z0: float) -> int:
        """Index of the slice whose centre is nearest to depth ``z0``."""
        return int(np.argmin(np.abs(self.z_centers - z0)))


def potential_stack(
    atoms,
    *,
    sampling: float = 0.2,
    slice_thickness: float = 2.0,
    energy: float = 300e3,
    parametrization: str = "lobato",
    projection: str = "infinite",
    device: str = "cpu",
    to_radians: bool = True,
) -> PotentialStack:
    """Build a depth-resolved projected potential with abTEM.

    Parameters
    ----------
    atoms
        ASE ``Atoms`` (orthorhombic, periodic) — typically a tricor
        supercell, e.g. 50 × 50 × 200 Å.
    sampling
        Target in-plane pixel size (Å); abTEM rounds it to integer gpts.
    slice_thickness
        Depth slice spacing (Å).
    energy
        Beam energy (eV) for the eV→radian interaction parameter.
    parametrization, projection, device
        Passed through to :class:`abtem.Potential`.
    to_radians
        Scale the potential (eV·Å) to phase (radians) via
        ``abtem.core.energy.energy2sigma`` so the result is
        voltage-explicit and matches a ptychographic phase.

    Returns
    -------
    PotentialStack
    """
    import abtem
    from abtem.core.energy import energy2sigma

    a = atoms.copy()
    a.pbc = True
    pot = abtem.Potential(
        a,
        sampling=sampling,
        slice_thickness=slice_thickness,
        parametrization=parametrization,
        projection=projection,
        device=device,
    )
    built = pot.build().compute()
    arr = np.asarray(built.array, dtype=np.float64)  # (n_slices, nx, ny), eV·Å

    dx, dy = (float(s) for s in pot.sampling)
    lz = float(np.diag(np.asarray(a.cell.array))[2])
    dz = lz / arr.shape[0]

    units = "eV*A"
    if to_radians:
        arr = arr * float(energy2sigma(energy))
        units = "rad"

    return PotentialStack(arr, (dx, dy), dz, float(energy), units=units)


def scattering_power(
    numbers,
    *,
    parametrization: str = "lobato",
    box: float = 6.0,
    sampling: float = 0.04,
) -> np.ndarray:
    """Per-atom total scattering power (integrated projected potential).

    The pseudo-ptychographic potential is dominated by heavier atoms, so
    the matching g2 / g3 target must weight each atom by its total
    scattering cross section.  That is the integral of a single atom's
    projected potential, ``∫∫ V_proj dx dy = ∫ V d³r`` (eV·Å³), which is
    energy-independent (the interaction parameter is a global scalar that
    cancels in the correlation normalisation) and depends only on the
    element and the abTEM ``parametrization``.

    Values are computed once per element via abTEM and cached to
    ``tricor/ptycho/data/scattering_<parametrization>.json``.

    Parameters
    ----------
    numbers
        Atomic numbers (e.g. ``atoms.numbers``).
    parametrization
        abTEM potential parametrization (``"lobato"``, ``"kirkland"``, …).
    box, sampling
        Single-atom box size and pixel size (Å) for the numerical
        integral.  The integral is conserved under periodicity, so a
        modest box suffices.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` scattering power per atom, in eV·Å³.
    """
    import json
    import pathlib

    numbers = np.asarray(np.atleast_1d(numbers), dtype=int)
    uniq = sorted(set(int(z) for z in numbers))

    data_dir = pathlib.Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    cache_file = data_dir / f"scattering_{parametrization}.json"
    table: dict[int, float] = {}
    if cache_file.exists():
        table = {int(k): float(v) for k, v in json.loads(cache_file.read_text()).items()}

    missing = [z for z in uniq if z not in table]
    if missing:
        import abtem
        from ase import Atoms

        for z in missing:
            atom = Atoms(numbers=[z], positions=[[box / 2] * 3], cell=[box] * 3, pbc=True)
            pot = abtem.Potential(
                atom,
                sampling=sampling,
                slice_thickness=box,  # one slice over the whole box
                parametrization=parametrization,
                projection="infinite",
            )
            arr = np.asarray(pot.build().compute().array, dtype=np.float64)
            dx, dy = (float(s) for s in pot.sampling)
            table[z] = float(arr.sum() * dx * dy)  # ∫∫ V_proj dx dy
        cache_file.write_text(json.dumps({str(k): v for k, v in sorted(table.items())}, indent=1))

    return np.array([table[int(z)] for z in numbers], dtype=np.float64)


def blur_stack(stack: PotentialStack, sigma_xy: float, sigma_z: float) -> PotentialStack:
    """Apply anisotropic Gaussian blur (fine in-plane, coarse in depth).

    Parameters
    ----------
    stack
        A raw :class:`PotentialStack`.
    sigma_xy
        In-plane resolution (Å), applied within each slice.
    sigma_z
        Depth resolution (Å), applied across slices.  Picking slice
        ``z0`` from the blurred stack reproduces a ``G(z - z0)`` depth
        weight with this ``sigma_z`` — match it to the g2 / g3 window.

    Returns
    -------
    PotentialStack
        A new stack with the blur applied and ``blur`` recorded.
    """
    from scipy.ndimage import gaussian_filter

    dx, dy = stack.sampling
    sig = (sigma_z / stack.slice_thickness, sigma_xy / dx, sigma_xy / dy)
    # Periodic cell -> wrap in all three axes (consistent with the
    # min-image-free, interior-window correlation convention).
    blurred = gaussian_filter(stack.array, sig, mode="wrap")
    return dataclasses.replace(stack, array=blurred, blur=(sigma_xy, sigma_z))
