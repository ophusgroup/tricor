"""Overlay train + val loss curves from a Lightning TensorBoard event file.

Reads scalar metrics from the events.out.tfevents file in a Lightning
version directory and writes a PNG with train_loss / val_loss (and the
relative-loss variants) on log-y axes.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RUN_DIR = "/home/ehrdt/tricor/scripts/macerelax/lightning_logs/composition_test_stride5_v3_no_st_dropout_wd1e-5/version_0"
OUT_PNG = "/home/ehrdt/tricor/scratch/train_val_losses_composition_test_stride5_v3.png"
INCLUDE_RELATIVE_LOSS = True   # second subplot for {train,val}_relative_loss if present

# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(run_dir: str) -> dict[str, tuple[list[float], list[float]]]:
    """Return {tag: (steps, values)} from a Lightning version dir's tfevents."""
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, tuple[list[float], list[float]]] = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        out[tag] = ([e.step for e in events], [e.value for e in events])
    return out


def main() -> None:
    run_dir = Path(RUN_DIR)
    if not run_dir.is_dir():
        raise SystemExit(f"[abort] not a dir: {run_dir}")
    scalars = load_scalars(str(run_dir))
    if not scalars:
        raise SystemExit(f"[abort] no scalars in {run_dir}")
    print(f"Found tags: {sorted(scalars)}")

    has_rel = INCLUDE_RELATIVE_LOSS and (
        "train_relative_loss" in scalars or "val_relative_loss" in scalars
    )
    n_plots = 2 if has_rel else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4.5), squeeze=False)
    axes = axes[0]

    def plot_pair(ax, train_tag: str, val_tag: str, title: str) -> None:
        if train_tag in scalars:
            xs, ys = scalars[train_tag]
            ax.plot(xs, ys, label=train_tag, color="#1f77b4", linewidth=1.6)
        if val_tag in scalars:
            xs, ys = scalars[val_tag]
            ax.plot(xs, ys, label=val_tag, color="#d62728", linewidth=1.6)
        ax.set_yscale("log")
        ax.set_xlabel("global step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")

    plot_pair(axes[0], "train_loss", "val_loss", "loss (MSE)")
    if has_rel:
        plot_pair(axes[1], "train_relative_loss", "val_relative_loss",
                  "relative loss (vs zero-displacement baseline)")

    fig.suptitle(run_dir.parent.name + " / " + run_dir.name, fontsize=11)
    fig.tight_layout()
    out = Path(OUT_PNG)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
