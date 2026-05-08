"""Render the relaxation *trajectory* using tricor's plot_structure graphics.

Each .npz contains a trajectory (multiple snapshots over the 200-step
relaxation).  We call Supercell.plot_structure(output=None) on every
snapshot to get one figure per frame — same bond-centric / depth-coloured
visuals used in notebooks/structure61test.ipynb — and pipe the frames to
ffmpeg for an MP4.  Camera angle stays fixed; the atoms themselves are
what moves.

Usage:
    python render_trajectory.py /path/to/trajectories_dir
    python render_trajectory.py /path/to/single_traj.npz

Requires ffmpeg on PATH.  Falls back to GIF via Pillow if not found.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.build import bulk

from tricor.shells import CoordinationShellTarget
from tricor.supercell import Supercell

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

FPS = 10                        # output playback speed (each snapshot = 1 frame)
WIDTH = 512
HEIGHT = 512
ATOM_SIZE = 10.0
COLORMAP = "Reds"               # matches structure61test.ipynb
BACKGROUND = "white"
HOLD_FINAL_SECONDS = 1.0        # repeat final snapshot this many seconds

# Si reference — used by plot_structure for bond cutoff / coordination count.
SI_LATTICE = 5.431

# ══════════════════════════════════════════════════════════════════════════════


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _build_supercell(
    positions0: np.ndarray, cell: np.ndarray, species: np.ndarray,
    rel_density: float, ref_atoms,
) -> Supercell:
    """Construct a Supercell and overwrite its atoms with the loaded state."""
    cell_edge = tuple(float(x) for x in np.diag(cell))
    sc = Supercell.from_atoms(
        ref_atoms, cell_dim_angstroms=cell_edge,
        rng_seed=0, relative_density=rel_density,
    )
    sc.atoms = Atoms(
        numbers=species, positions=positions0,
        cell=cell, pbc=ref_atoms.pbc,
    )
    sc._cell_matrix = np.asarray(sc.atoms.cell.array, dtype=np.float64)
    sc._cell_inverse = np.linalg.inv(sc._cell_matrix)
    return sc


def _frame_to_bytes(fig, dpi: int) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="raw", dpi=dpi, facecolor=BACKGROUND)
    plt.close(fig)
    return buf.getvalue()


def render_trajectory_mp4(
    npz_path: Path, out_path: Path, ref_atoms, shell_target,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    positions = np.asarray(data["positions"])    # (S, N, 3)
    cell = np.asarray(data["cell"], dtype=np.float64)
    species = np.asarray(data["species_numbers"], dtype=np.int64)
    rel_density = float(data["rel_density"])
    regime = str(data["regime"].item())
    num_snaps = positions.shape[0]
    num_atoms = positions.shape[1]

    sc = _build_supercell(positions[0], cell, species, rel_density, ref_atoms)

    # Frame dimensions — keep consistent with dpi=100 figsize.
    dpi = 100

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "18",
        str(out_path),
    ]
    proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    try:
        last_frame_bytes = b""
        for i in range(num_snaps):
            sc.atoms.positions = positions[i]
            fig = sc.plot_structure(
                shell_target, output=None,
                width=WIDTH, height=HEIGHT,
                atom_size=ATOM_SIZE, colormap=COLORMAP,
                background=BACKGROUND, show_progress=False,
            )
            last_frame_bytes = _frame_to_bytes(fig, dpi)
            proc.stdin.write(last_frame_bytes)
        # Hold the final state for a moment so the viewer can see the result.
        hold_frames = int(HOLD_FINAL_SECONDS * FPS)
        for _ in range(hold_frames):
            proc.stdin.write(last_frame_bytes)
    finally:
        proc.stdin.close()
        _, err = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (rc={proc.returncode}): {err.decode(errors='replace')}"
            )
    print(f"  -> {out_path.name}  [{regime}, {num_atoms} atoms, {num_snaps} frames]")


def render_trajectory_gif(
    npz_path: Path, out_path: Path, ref_atoms, shell_target,
) -> None:
    """Fallback path: write a GIF via PIL when ffmpeg is unavailable."""
    from PIL import Image
    data = np.load(npz_path, allow_pickle=True)
    positions = np.asarray(data["positions"])
    cell = np.asarray(data["cell"], dtype=np.float64)
    species = np.asarray(data["species_numbers"], dtype=np.int64)
    rel_density = float(data["rel_density"])
    regime = str(data["regime"].item())
    num_snaps = positions.shape[0]
    num_atoms = positions.shape[1]

    sc = _build_supercell(positions[0], cell, species, rel_density, ref_atoms)
    frames: list[Image.Image] = []
    for i in range(num_snaps):
        sc.atoms.positions = positions[i]
        fig = sc.plot_structure(
            shell_target, output=None,
            width=WIDTH, height=HEIGHT,
            atom_size=ATOM_SIZE, colormap=COLORMAP,
            background=BACKGROUND, show_progress=False,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=BACKGROUND,
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
    hold_frames = int(HOLD_FINAL_SECONDS * FPS)
    frames.extend([frames[-1].copy() for _ in range(hold_frames)])

    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / FPS), loop=0,
    )
    print(f"  -> {out_path.name}  [{regime}, {num_atoms} atoms, {len(frames)} frames]")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python render_trajectory.py <npz-file-or-dir>")
        sys.exit(1)

    target = Path(sys.argv[1]).resolve()
    use_gif = not _has_ffmpeg()
    ext = ".gif" if use_gif else ".mp4"
    if use_gif:
        print("ffmpeg not found — writing .gif instead of .mp4")

    if target.is_file() and target.suffix == ".npz":
        npz_files = [target]
        out_dir = target.parent
    elif target.is_dir():
        npz_files = sorted(target.glob("*.npz"))
        out_dir = target
    else:
        print(f"Not a .npz file or directory: {target}")
        sys.exit(1)

    render_dir = out_dir / "renders"
    render_dir.mkdir(exist_ok=True)

    ref_atoms = bulk("Si", crystalstructure="diamond", a=SI_LATTICE, cubic=True)
    shell_target = CoordinationShellTarget.from_atoms(ref_atoms)

    print(f"Rendering {len(npz_files)} trajectories -> {render_dir}")
    print(f"fps={FPS}  size={WIDTH}x{HEIGHT}  hold_final={HOLD_FINAL_SECONDS}s")

    renderer = render_trajectory_gif if use_gif else render_trajectory_mp4
    for npz_path in npz_files:
        out_path = render_dir / (npz_path.stem + ext)
        try:
            renderer(npz_path, out_path, ref_atoms, shell_target)
        except Exception as e:
            print(f"  {npz_path.name}: FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
