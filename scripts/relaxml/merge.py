"""Merge two trajectory manifests + symlink their .npz files into a new dir.

Self-creates the destination, validates sources before touching anything,
and prints sanity counts at the end.

Edit the CONFIG block below, then run:
    python merge_manifests.py
"""

from __future__ import annotations

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

SOURCES = [
    "./data/sio2_polymorphs_v1/SiO2/alpha_quartz_trajectories",
    "./data/sio2_polymorphs_v1/SiO2/alpha_cristobalite_trajectories",                                                                                                                                                                            
    "./data/sio2_polymorphs_v1/SiO2/beta_cristobalite_trajectories",   
    "./data/sio2_polymorphs_v1/SiO2/coesite_trajectories",                                                                                                                                                                                                                                                                                                                                                                                                                              
    "./data/multi_species_v1/Al2O3_trajectories",
    "./data/multi_species_v1/Ga2O3_mp-886_trajectories_150",
    "./data/multi_species_v1/TiO2_mp-390_trajectories_150",
#     "./data/multi_species_v1/Si3N4_trajectories",
    # "./data/multi_species_v1/SiC_trajectories",
    # "./data/multi_species_v1/SiO2_trajectories",
    # "./data/multi_species_v1/Si_trajectories",
    # "./data/multi_species_v1/BN_trajectories",
    # "./data/multi_species_v1/AlN_trajectories",
    ]
DESTINATION = "./data/sio2_polymorphs_v1/merged_train_for_stishovite" #"./data/si-n-trajectories" #

# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    src_dirs = [Path(s).resolve() for s in SOURCES]
    dst_dir = Path(DESTINATION).resolve()

    # Verify sources before touching anything.
    for sd in src_dirs:
        if not sd.is_dir():
            raise SystemExit(f"Source directory not found: {sd}")
        if not (sd / "manifest.csv").is_file():
            raise SystemExit(f"Source manifest not found: {sd}/manifest.csv")
        n = len(list(sd.glob("*.npz")))
        print(f"  {sd}: {n} .npz files, manifest OK")

    # Create destination.
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {dst_dir}")

    # Symlink .npz files.  Absolute paths so the symlinks resolve from any
    # cwd, and we replace any pre-existing symlink with the same name.
    n_linked = 0
    for sd in src_dirs:
        for src_npz in sorted(sd.glob("*.npz")):
            link = dst_dir / src_npz.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src_npz)
            n_linked += 1
    print(f"Linked {n_linked} .npz files into destination")

    # Merge manifests — header from first source, body rows from all sources.
    out_manifest = dst_dir / "manifest.csv"
    parts: list[str] = []
    for i, sd in enumerate(src_dirs):
        text = (sd / "manifest.csv").read_text().rstrip("\n").splitlines()
        if i == 0:
            parts.append(text[0])  # header
        parts.extend(text[1:])     # body
    out_manifest.write_text("\n".join(parts) + "\n")

    # Sanity counts.
    n_rows = sum(1 for _ in out_manifest.open()) - 1
    n_npz = len(list(dst_dir.glob("*.npz")))
    print()
    print(f"Merged manifest: {out_manifest}")
    print(f"  rows:       {n_rows}")
    print(f"  .npz files: {n_npz}")
    if n_rows != n_npz:
        print(f"  WARNING: row count != .npz count — possible duplicate filenames")


if __name__ == "__main__":
    main()
