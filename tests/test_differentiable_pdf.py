"""
Tests for the differentiable PDF and ADF implementation.

Validates:
1. Gradient flow (autograd works through the full computation)
2. Correctness against known structures (peak positions)
3. Invariance properties (translation, PBC wrapping)
4. Multi-species support
"""

import torch
import numpy as np
import pytest


def make_si_diamond(repeat=(3, 3, 3)):
    """Create a silicon diamond supercell as torch tensors (no ASE)."""
    a = 5.43
    basis_frac = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.50, 0.50, 0.00],
            [0.50, 0.00, 0.50],
            [0.00, 0.50, 0.50],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ]
    )
    nx, ny, nz = repeat
    positions = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                offset = np.array([ix, iy, iz])
                for b in basis_frac:
                    positions.append((b + offset) * a)

    positions = np.array(positions)
    cell = np.diag([a * nx, a * ny, a * nz])
    n_atoms = len(positions)
    species_np = np.full(n_atoms, 14, dtype=np.int64)

    positions_t = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    species_t = torch.tensor(species_np, dtype=torch.long)
    cell_t = torch.tensor(cell, dtype=torch.float64)
    return positions_t, species_t, cell_t


def make_nacl(repeat=(2, 2, 2)):
    """Create a NaCl rocksalt supercell as torch tensors."""
    a = 5.64
    fcc = np.array(
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    )
    basis_na = fcc
    basis_cl = fcc + 0.5

    nx, ny, nz = repeat
    positions = []
    species_list = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                offset = np.array([ix, iy, iz])
                for b in basis_na:
                    positions.append((b + offset) * a)
                    species_list.append(11)
                for b in basis_cl:
                    positions.append((b + offset) * a)
                    species_list.append(17)

    positions = np.array(positions)
    cell = np.diag([a * nx, a * ny, a * nz])

    positions_t = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    species_t = torch.tensor(species_list, dtype=torch.long)
    cell_t = torch.tensor(cell, dtype=torch.float64)
    return positions_t, species_t, cell_t


# ============================================================
# PDF + ADF Tests
# ============================================================


class TestDifferentiablePDFADF:
    def test_gradient_flows_g2(self):
        """Verify that gradients propagate through g2 computation."""
        from tricor.differentiable_pdf import DifferentiablePDFADF

        positions, species, cell = make_si_diamond((2, 2, 2))

        calc = DifferentiablePDFADF(
            r_max=6.0, r_step=0.075, phi_num_bins=45,
            sigma_r=0.15, sigma_phi=0.1, species=[14],
        ).double()
        g2, adf = calc.compute(positions, species, cell)

        loss = g2.sum()
        loss.backward()

        assert positions.grad is not None
        assert not torch.all(positions.grad == 0)
        assert torch.isfinite(positions.grad).all()

    def test_gradient_flows_adf(self):
        """Verify that gradients propagate through ADF computation."""
        from tricor.differentiable_pdf import DifferentiablePDFADF

        positions, species, cell = make_si_diamond((2, 2, 2))

        calc = DifferentiablePDFADF(
            r_max=3.0, r_step=0.1, phi_num_bins=45,
            sigma_r=0.15, sigma_phi=0.1, species=[14],
        ).double()
        g2, adf = calc.compute(positions, species, cell)

        # Use (adf**2).sum() rather than adf.sum() — in a perfect crystal,
        # adf.sum() is approximately invariant to small displacements (the
        # Gaussian kernels shift but their total integral is conserved).
        # Squaring breaks this invariance and ensures non-zero gradients.
        loss = (adf ** 2).sum()
        loss.backward()

        assert positions.grad is not None
        assert not torch.all(positions.grad == 0)
        assert torch.isfinite(positions.grad).all()

    def test_output_shapes_single_species(self):
        """Check output shapes for single-species system."""
        from tricor.differentiable_pdf import DifferentiablePDFADF

        positions, species, cell = make_si_diamond((2, 2, 2))

        calc = DifferentiablePDFADF(
            r_max=6.0, r_step=0.075, phi_num_bins=45,
            sigma_r=0.15, sigma_phi=0.1, species=[14],
        ).double()
        g2, adf = calc.compute(positions, species, cell)

        # Single species: g2 is (1, 1, num_r=80), adf is (1, phi_bins=45)
        assert g2.shape == (1, 1, 80)
        assert adf.shape == (1, 45)

    def test_g2_peak_at_known_distance(self):
        """Si diamond first-neighbor distance is ~2.35 A."""
        from tricor.differentiable_pdf import DifferentiablePDFADF

        positions, species, cell = make_si_diamond((3, 3, 3))

        calc = DifferentiablePDFADF(
            r_max=6.0, r_step=0.03, phi_num_bins=45,
            sigma_r=0.08, sigma_phi=0.1, species=[14],
        ).double()
        g2, _ = calc.compute(positions, species, cell)

        r = calc.r_grid.numpy()
        g2_np = g2[0, 0].detach().numpy()

        # Raw counts grow as r^2 (more atoms at larger distances).
        # Normalize by r^2 to recover the standard g(r) shape.
        g2_norm = g2_np / (r ** 2 + 1e-10)

        # Find the first local maximum above r > 1.5 A (Si diamond has
        # 4 first-neighbors at 2.35 A and 12 second-neighbors at 3.84 A,
        # so the global max may not be the first peak).
        mask = r > 1.5
        g2_sub = g2_norm[mask]
        r_sub = r[mask]
        local_maxima = (
            (g2_sub[1:-1] > g2_sub[:-2]) & (g2_sub[1:-1] > g2_sub[2:])
        )
        first_peak_idx = np.argmax(local_maxima) + 1
        peak_r = r_sub[first_peak_idx]
        assert abs(peak_r - 2.35) < 0.15, f"First peak at {peak_r}, expected ~2.35"

    def test_tetrahedral_angle(self):
        """Si diamond should have ADF peak near tetrahedral angle 109.47 deg."""
        from tricor.differentiable_pdf import DifferentiablePDFADF

        positions, species, cell = make_si_diamond((3, 3, 3))

        # Use tight r_max to select only first neighbors
        calc = DifferentiablePDFADF(
            r_max=2.6, r_step=0.1, phi_num_bins=180,
            sigma_r=0.15, sigma_phi=0.05, species=[14],
        ).double()
        _, adf = calc.compute(positions, species, cell)

        phi_deg = calc.phi_grid.numpy() * 180 / np.pi
        adf_np = adf[0].detach().numpy()

        peak_idx = np.argmax(adf_np)
        peak_angle = phi_deg[peak_idx]
        assert abs(peak_angle - 109.47) < 3.0, (
            f"Peak at {peak_angle} deg, expected ~109.47"
        )

    def test_translation_invariance(self):
        """g2 and ADF should be invariant to rigid translation."""
        from tricor.differentiable_pdf import DifferentiablePDFADF

        positions, species, cell = make_si_diamond((2, 2, 2))

        calc = DifferentiablePDFADF(
            r_max=6.0, r_step=0.1, phi_num_bins=45,
            sigma_r=0.15, sigma_phi=0.1, species=[14],
        ).double()

        g2_orig, adf_orig = calc.compute(positions, species, cell)
        g2_orig, adf_orig = g2_orig.detach(), adf_orig.detach()

        shift = torch.tensor([1.23, -0.45, 2.67], dtype=torch.float64)
        pos_shifted = (positions.detach() + shift).requires_grad_(True)
        g2_shift, adf_shift = calc.compute(pos_shifted, species, cell)
        g2_shift, adf_shift = g2_shift.detach(), adf_shift.detach()

        assert torch.allclose(g2_orig, g2_shift, atol=1e-6)
        assert torch.allclose(adf_orig, adf_shift, atol=1e-6)

    def test_pbc_wrapping_invariance(self):
        """g2 and ADF should be invariant to wrapping atoms across PBC."""
        from tricor.differentiable_pdf import DifferentiablePDFADF

        positions, species, cell = make_si_diamond((2, 2, 2))

        calc = DifferentiablePDFADF(
            r_max=6.0, r_step=0.1, phi_num_bins=45,
            sigma_r=0.15, sigma_phi=0.1, species=[14],
        ).double()

        g2_orig, adf_orig = calc.compute(positions, species, cell)
        g2_orig, adf_orig = g2_orig.detach(), adf_orig.detach()

        pos_wrapped = positions.detach().clone()
        pos_wrapped[0] = pos_wrapped[0] + cell[0]
        pos_wrapped = pos_wrapped.requires_grad_(True)
        g2_wrap, adf_wrap = calc.compute(pos_wrapped, species, cell)
        g2_wrap, adf_wrap = g2_wrap.detach(), adf_wrap.detach()

        assert torch.allclose(g2_orig, g2_wrap, atol=1e-5)
        assert torch.allclose(adf_orig, adf_wrap, atol=1e-5)

    def test_multispecies_shapes(self):
        """Test output shapes and labels with NaCl (two species)."""
        from tricor.differentiable_pdf import DifferentiablePDFADF

        positions, species, cell = make_nacl((2, 2, 2))

        calc = DifferentiablePDFADF(
            r_max=6.0, r_step=0.1, phi_num_bins=45,
            sigma_r=0.15, sigma_phi=0.1, species=[11, 17],
        ).double()
        g2, adf = calc.compute(positions, species, cell)

        # 2 species: g2 is (2, 2, num_r=60), adf has 6 triplet types
        assert g2.shape == (2, 2, 60)
        assert adf.shape == (6, 45)

        assert torch.isfinite(g2).all()
        assert torch.isfinite(adf).all()

        # Gradient should flow through both
        (g2.sum() + adf.sum()).backward()
        assert positions.grad is not None

    def test_multispecies_triplet_labels(self):
        """Verify triplet type count and labels for two species."""
        from tricor.differentiable_pdf import DifferentiablePDFADF

        calc = DifferentiablePDFADF(
            r_max=3.0, r_step=0.1, phi_num_bins=45,
            sigma_r=0.15, sigma_phi=0.1, species=[11, 17],
        )
        # 2 species -> 6 rooted triplet types
        assert calc.num_triplets == 6
        assert len(calc.triplet_labels) == 6

    def test_r_max_divisibility_check(self):
        """Constructor should reject r_max not divisible by r_step."""
        from tricor.differentiable_pdf import DifferentiablePDFADF

        with pytest.raises(ValueError, match="divisible"):
            DifferentiablePDFADF(
                r_max=6.0, r_step=0.07, phi_num_bins=45,
                sigma_r=0.15, sigma_phi=0.1, species=[14],
            )


# ============================================================
# Guidance Loss Tests
# ============================================================


class TestSpectralLoss:
    def test_guidance_gradient(self):
        """Verify compute_guidance returns valid per-atom vectors."""
        from tricor.differentiable_pdf import (
            DifferentiablePDFADF,
            DifferentiableSpectralLoss,
        )

        positions, species, cell = make_si_diamond((2, 2, 2))

        calc = DifferentiablePDFADF(
            r_max=3.0, r_step=0.1, phi_num_bins=30,
            sigma_r=0.15, sigma_phi=0.1, species=[14],
        ).double()

        loss_fn = DifferentiableSpectralLoss(calc)

        # Use current structure's spectra as targets (loss should be ~0)
        with torch.no_grad():
            target_g2, target_adf = calc.compute(positions, species, cell)

        guidance, components = loss_fn.compute_guidance(
            positions, species, cell, target_g2, target_adf
        )

        assert guidance.shape == positions.shape
        assert torch.isfinite(guidance).all()
        # Loss should be near zero when target matches
        assert components["pdf_loss"] < 1e-6
        assert components["adf_loss"] < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
