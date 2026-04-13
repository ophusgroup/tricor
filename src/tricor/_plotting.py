from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.neighborlist import neighbor_list

from .g3 import _EPS, _TextProgressBar

if TYPE_CHECKING:
    from .shells import CoordinationShellTarget
    from .supercell import Supercell


class _PlottingMixin:
    """Plotting methods extracted from Supercell."""

    def view_structure(
        self: "Supercell",
        shell_target: "CoordinationShellTarget | None" = None,
        **kwargs,
    ):
        """Return an interactive 3D structure viewer widget.

        Renders atoms as spheres (coloured by element) and bonds as
        cylinders inside the periodic cell outline.  Uses Three.js
        WebGL with OrbitControls for drag-to-rotate / scroll-to-zoom.

        Parameters
        ----------
        shell_target
            Sets the default bond cutoff from
            ``shell_target.max_pair_outer``.  If ``None``, uses the
            shell_target stored from the last :meth:`generate` call.
        **kwargs
            Forwarded to :class:`StructureWidget` (e.g. ``atom_scale``,
            ``bond_cutoff``, ``show_bonds``, ``slab_x``, etc.).

        Returns
        -------
        StructureWidget
            An anywidget instance for display in Jupyter.
        """
        from .structure_widget import StructureWidget

        if shell_target is None:
            shell_target = getattr(self, "_shell_target", None)

        return StructureWidget(
            self.atoms,
            shell_target=shell_target,
            grain_ids=self._grain_ids,
            **kwargs,
        )

    def plot_g3(
        self: "Supercell",
        pair: int | str = 0,
        *,
        normalize: bool = True,
    ):
        """Return an interactive explorer for the supercell's measured g3.

        Requires :meth:`measure_g3` to have been called first.

        Parameters
        ----------
        pair
            Triplet index or label (e.g. ``0`` or ``"Si-Si-Si"``).
        normalize
            If ``True``, display the reduced (density-normalised) g3.
        """
        if self.current_distribution is None:
            raise ValueError("Call measure_g3() before plot_g3().")
        dist = self.current_distribution
        return dist.plot_g3(pair=pair, normalize=normalize)

    def plot_g3_compare(
        self: "Supercell",
        pair: int | str = 0,
        *,
        normalize: bool = True,
    ):
        """Return an interactive comparison between the current supercell and target."""
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
                "cell_dim_angstroms "
                f"{self.cell_dim_angstroms[0]:.2f}x"
                f"{self.cell_dim_angstroms[1]:.2f}x"
                f"{self.cell_dim_angstroms[2]:.2f} A"
            ),
        )

    def _display_compare_widget(self: "Supercell") -> None:
        """Display the comparison widget immediately when running in IPython."""
        try:
            from IPython.display import display
        except Exception:
            return
        display(self.plot_g3_compare())

    def plot_monte_carlo(
        self: "Supercell",
        *,
        log_y: bool = False,
        show_run_boundaries: bool = True,
    ):
        """Plot the recorded Monte Carlo cost history using Matplotlib."""
        if self.mc_history is None:
            raise ValueError("Run monte_carlo() before plotting the history.")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(self.mc_history["step"], self.mc_history["cost"], lw=1.8, label="cost")
        ax.plot(self.mc_history["step"], self.mc_history["best_cost"], lw=1.4, label="best")
        if show_run_boundaries and "run_index" in self.mc_history:
            run_index = np.asarray(self.mc_history["run_index"], dtype=np.int32)
            step = np.asarray(self.mc_history["step"], dtype=np.int32)
            change_points = np.flatnonzero(np.diff(run_index) > 0)
            for point_index in change_points:
                ax.axvline(
                    float(step[point_index + 1]),
                    color="0.65",
                    lw=0.9,
                    ls="--",
                    alpha=0.7,
                )
        if log_y:
            positive_cost = np.asarray(self.mc_history["cost"], dtype=np.float64)
            positive_cost = positive_cost[positive_cost > 0.0]
            if positive_cost.size:
                ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("cost")
        ax.set_title("Monte Carlo cost history")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig, ax

    def plot_structure(
        self: "Supercell",
        shell_target: "CoordinationShellTarget | None" = None,
        *,
        output: str | None = None,
        width: int = 1024,
        height: int = 1024,
        fps: int = 60,
        duration: float = 6.0,
        elevation: float = 15.0,
        atom_size: float = 10.0,
        bond_cutoff: float | None = None,
        show_cell: bool = True,
        show_atoms: bool = True,
        background: str = "white",
        colormap: str = "Reds",
        tetrahedral_thresh: float = 0.4,
        show_progress: bool = True,
    ):
        """Render a bond-centric rotating 3D view of the atomic structure.

        Bonds are the primary visual: crystalline (tetrahedral) bonds
        are drawn thick and coloured by depth; boundary / amorphous
        bonds are drawn faint.  Atoms are optional small dots.  The
        animation performs a full periodic 360-degree rotation.

        Classification follows the MATLAB ``plotAtoms02`` convention:
        an atom is *crystalline* if it has exactly K nearest neighbours
        within *bond_cutoff* **and** the mean displacement of those
        neighbours is less than *tetrahedral_thresh* (i.e. the local
        coordination is symmetric / tetrahedral).

        Parameters
        ----------
        shell_target
            First-shell targets.  Used to set *bond_cutoff* and the
            coordination number K automatically.
        output
            File path for a ``.mp4`` (recommended) or ``.gif``.
            ``None`` shows a static figure.
        width, height
            Frame size in pixels.
        fps
            Frames per second (GIF only).
        duration
            Total GIF length in seconds.  Rotation is always exactly
            360 degrees so the loop is seamless.
        elevation
            Camera elevation in degrees.
        atom_size
            Matplotlib scatter marker size.  Set to 0 to hide atoms.
        bond_cutoff
            NN bond length cutoff in Angstrom.
        show_cell
            Draw the periodic cell outline.
        show_atoms
            Draw atom dots.
        background
            Figure background colour.
        colormap
            Matplotlib colormap for crystalline bonds (coloured by
            depth / y-coordinate after rotation, like MATLAB ``bone``).
        tetrahedral_thresh
            Maximum norm of mean NN displacement vector for an atom to
            be classified as crystalline.  Smaller = stricter.
        show_progress
            Print frame counter during GIF rendering.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        pos = self.atoms.positions.copy()
        cell_mat = np.asarray(self.atoms.cell.array, dtype=np.float64)
        cell_inv = np.linalg.inv(cell_mat)
        num_atoms = len(self.atoms)

        # --- bond cutoff ---
        if bond_cutoff is None:
            if shell_target is not None:
                # Use a generous cutoff (pair_peak + 3*sigma or 1.2x)
                pair_peak_max = float(np.max(
                    np.asarray(shell_target.pair_peak, dtype=np.float64),
                ))
                bond_cutoff = pair_peak_max * 1.2
            else:
                bond_cutoff = 3.0

        if shell_target is not None:
            coord_target = np.asarray(shell_target.coordination_target, dtype=np.float64)
            species_idx = self._atom_species_index
            k_per_atom = np.array([
                int(np.round(coord_target[species_idx[a]].sum()))
                for a in range(num_atoms)
            ], dtype=np.intp)
        else:
            k_per_atom = np.full(num_atoms, 4, dtype=np.intp)

        # --- find bonds ---
        bi_all, bj_all, bd_all = neighbor_list("ijd", self.atoms, bond_cutoff)

        def _min_image(delta: np.ndarray) -> np.ndarray:
            frac = delta @ cell_inv
            frac -= np.rint(frac)
            return frac @ cell_mat

        # --- classify atoms as crystalline ---
        # Two criteria (either makes an atom crystalline):
        # 1) Tetrahedral check (MATLAB style): K NN within cutoff and
        #    symmetric coordination (mean displacement < thresh).
        #    Allow K-1 to K+1 neighbors for tolerance.
        # 2) Grain interior: deep inside a crystalline Voronoi cell.
        is_crystalline_atom = np.zeros(num_atoms, dtype=bool)

        # Criterion 1: tetrahedral / symmetric coordination
        for a in range(num_atoms):
            mask = bi_all == a
            nn_count = int(np.sum(mask))
            k_target = int(k_per_atom[a])
            if nn_count < max(k_target - 1, 1) or nn_count > k_target + 1:
                continue
            # Use K nearest for the displacement check
            dists_a = bd_all[mask]
            js_a = bj_all[mask]
            order = np.argsort(dists_a)[:k_target]
            dxyz = _min_image(pos[js_a[order]] - pos[a])
            mean_disp = np.linalg.norm(np.mean(dxyz, axis=0))
            if mean_disp < tetrahedral_thresh:
                is_crystalline_atom[a] = True

        # Criterion 2: grain interior atoms (always crystalline)
        grain_ids = self._grain_ids
        grain_seeds = self._grain_seeds
        if grain_ids is not None and grain_seeds is not None:
            pp_max = float(np.max(
                np.asarray(shell_target.pair_peak, dtype=np.float64),
            )) if shell_target is not None else 2.5
            bw = pp_max * 0.5
            delta_seeds = pos[:, None, :] - grain_seeds[None, :, :]
            frac_ds = delta_seeds @ cell_inv
            frac_ds -= np.rint(frac_ds)
            cart_ds = frac_ds @ cell_mat
            dist_to_seeds = np.sqrt(np.sum(cart_ds ** 2, axis=2))
            for ia in range(num_atoms):
                gid = grain_ids[ia]
                if gid < 0:
                    continue
                d_own = dist_to_seeds[ia, gid]
                dists_copy = dist_to_seeds[ia].copy()
                dists_copy[gid] = np.inf
                d_other = float(np.min(dists_copy))
                if (d_other - d_own) * 0.5 > bw:
                    is_crystalline_atom[ia] = True

        # Keep i < j for unique bonds
        mask_ij = bi_all < bj_all
        bi, bj = bi_all[mask_ij], bj_all[mask_ij]

        # Bond is crystalline if BOTH endpoints are crystalline
        bond_is_cryst = is_crystalline_atom[bi] & is_crystalline_atom[bj]

        # Bond segment endpoints (minimum-image)
        bond_vecs = _min_image(pos[bj] - pos[bi])
        bond_starts = pos[bi]
        bond_ends = bond_starts + bond_vecs

        # --- center everything ---
        a_vec, b_vec, c_vec = cell_mat[0], cell_mat[1], cell_mat[2]
        center = 0.5 * (a_vec + b_vec + c_vec)
        pos_c = pos - center
        bstart_c = bond_starts - center
        bend_c = bond_ends - center

        # Cell outline edges
        o = -center
        cell_corners = [
            o, o + a_vec, o + b_vec, o + c_vec,
            o + a_vec + b_vec, o + a_vec + c_vec, o + b_vec + c_vec,
            o + a_vec + b_vec + c_vec,
        ]
        cell_edge_pairs = [
            (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
            (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
        ]
        cell_segs = [(cell_corners[i], cell_corners[j]) for i, j in cell_edge_pairs]

        # --- colormap for crystalline bonds (depth-coloured) ---
        cmap = plt.get_cmap(colormap)

        cryst_mask = bond_is_cryst
        bnd_mask = ~bond_is_cryst
        extent = float(np.max(np.abs(pos_c))) * 1.15

        dpi = 100
        figsize = (width / dpi, height / dpi)

        def _rotate_2d(pts: np.ndarray, theta: float) -> np.ndarray:
            """Rotate x-y columns by theta radians (in-place friendly)."""
            c, s = np.cos(theta), np.sin(theta)
            x_new = pts[:, 0] * c - pts[:, 1] * s
            y_new = pts[:, 0] * s + pts[:, 1] * c
            out = pts.copy()
            out[:, 0] = x_new
            out[:, 1] = y_new
            return out

        def _draw_frame(theta_rad: float) -> "plt.Figure":
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor(background)
            fig.patch.set_facecolor(background)

            # Perspective projection (like MATLAB camproj('perspective'))
            try:
                ax.set_proj_type("persp", focal_length=0.25)
            except (TypeError, AttributeError):
                pass  # older matplotlib

            # Rotate bond endpoints in x-y plane (like MATLAB)
            bs_r = _rotate_2d(bstart_c, theta_rad)
            be_r = _rotate_2d(bend_c, theta_rad)

            # --- boundary bonds: very faint ---
            if np.any(bnd_mask):
                segs_b = list(zip(bs_r[bnd_mask], be_r[bnd_mask]))
                lc_b = Line3DCollection(
                    segs_b, linewidths=0.3,
                    colors=(0.0, 0.0, 0.0, 0.05),
                )
                ax.add_collection3d(lc_b)

            # --- crystalline bonds: depth-coloured + depth-width ---
            # Camera at +x (azim=0): larger rotated-x = closer.
            # Use squared depth for stronger contrast: close bonds pop,
            # far bonds fade to near-white via the Reds colormap.
            if np.any(cryst_mask):
                segs_cr = list(zip(bs_r[cryst_mask], be_r[cryst_mask]))
                mid_x_rot = 0.5 * (bs_r[cryst_mask, 0] + be_r[cryst_mask, 0])
                norm_depth = (mid_x_rot + extent) / max(2.0 * extent, _EPS)
                norm_depth = np.clip(norm_depth, 0, 1)
                # Square for stronger contrast: far → ~0 (white), close → 1 (bold)
                norm_sq = norm_depth ** 2
                # Colormap range 0.05–0.95 to avoid pure white and clipping
                cryst_colors = cmap(0.05 + 0.9 * norm_sq)
                # Linewidth: 0.2 at back, 2.5 at front
                cryst_lw = 0.2 + 2.3 * norm_sq
                lc_c = Line3DCollection(
                    segs_cr, linewidths=cryst_lw, colors=cryst_colors,
                )
                ax.add_collection3d(lc_c)

            # --- cell outline ---
            if show_cell:
                cell_segs_r = []
                for s, e in cell_segs:
                    s_r = _rotate_2d(s.reshape(1, 3), theta_rad)[0]
                    e_r = _rotate_2d(e.reshape(1, 3), theta_rad)[0]
                    cell_segs_r.append((s_r, e_r))
                lc_cell = Line3DCollection(
                    cell_segs_r, linewidths=1.5, colors="k", alpha=0.5,
                )
                ax.add_collection3d(lc_cell)

            # --- atoms (tiny dots) ---
            if show_atoms and atom_size > 0:
                pos_r = _rotate_2d(pos_c, theta_rad)
                ax.scatter(
                    pos_r[:, 0], pos_r[:, 1], pos_r[:, 2],
                    s=atom_size, c="k", alpha=0.15,
                    edgecolors="none", depthshade=False,
                )

            ax.set_xlim(-extent, extent)
            ax.set_ylim(-extent, extent)
            ax.set_zlim(-extent, extent)
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=elevation, azim=0)  # azim fixed; we rotate data
            ax.axis("off")
            fig.subplots_adjust(left=-0.05, right=1.05, bottom=-0.05, top=1.05)
            return fig

        # --- static display or animation ---
        if output is None:
            return _draw_frame(theta_rad=np.deg2rad(45.0))

        # Full 360-degree periodic rotation
        n_frames = int(fps * duration)
        thetas = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

        if show_progress:
            progress = _TextProgressBar(n_frames, label="Rendering", width=28)
        else:
            progress = None

        is_mp4 = str(output).lower().endswith(".mp4")

        if is_mp4:
            # MP4 via ffmpeg subprocess — true 60fps
            import subprocess
            import io

            # Frame dimensions from figure size (no bbox_inches="tight"
            # for the raw pipe — keeps pixel count deterministic).
            fw, fh = width, height

            ffmpeg_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "-s", f"{fw}x{fh}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "18",
                str(output),
            ]
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            for i, th in enumerate(thetas):
                fig_i = _draw_frame(th)
                buf_i = io.BytesIO()
                fig_i.savefig(buf_i, format="raw", dpi=dpi,
                              facecolor=background)
                plt.close(fig_i)
                proc.stdin.write(buf_i.getvalue())
                if progress is not None:
                    progress.update(i + 1)

            proc.stdin.close()
            proc.wait()
        else:
            # GIF fallback via Pillow
            from PIL import Image
            import io

            frames: list[Image.Image] = []
            for i, th in enumerate(thetas):
                fig = _draw_frame(th)
                buf = io.BytesIO()
                fig.savefig(
                    buf, format="png", dpi=dpi,
                    bbox_inches="tight", pad_inches=0,
                    facecolor=background,
                )
                plt.close(fig)
                buf.seek(0)
                frames.append(Image.open(buf).copy())
                buf.close()
                if progress is not None:
                    progress.update(i + 1)

            frame_duration_ms = int(1000 / fps)
            frames[0].save(
                output,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration_ms,
                loop=0,
                optimize=True,
            )

        if progress is not None:
            progress.update(n_frames)
        return output
