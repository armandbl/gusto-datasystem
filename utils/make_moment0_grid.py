#!/usr/bin/env python3
"""Assemble moment-0 PNGs into a 2×3 comparison grid matching the thesis layout.

Row 1: CII M8 (ref)  |  CII M5 (tgt)  |  (empty)
Row 2: NII M3 (ref)  |  NII M2 (tgt)  |  NII M6 (tgt)

Usage:
  python utils/make_moment0_grid.py --source G337 --moment0-dir <path/to/Compare/moment0> --output <path/to/output.png>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def build_grid(source: str, moment0_dir: Path, output_path: Path) -> None:
    """Load individual moment-0 PNGs and assemble into a 2×3 grid."""
    # Expected files
    panels: list[tuple[int, int, str | None]] = [
        (0, 0, f"moment0_{source}_CII_M8_reference.png"),   # top-left
        (0, 1, f"moment0_{source}_CII_M5_target.png"),       # top-centre
        (0, 2, None),                                         # top-right (empty)
        (1, 0, f"moment0_{source}_NII_M3_reference.png"),    # bottom-left
        (1, 1, f"moment0_{source}_NII_M2_target.png"),       # bottom-centre
        (1, 2, f"moment0_{source}_NII_M6_target.png"),       # bottom-right
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)

    for row, col, fname in panels:
        ax = axes[row][col]
        if fname is None:
            ax.axis("off")
            continue

        img_path = moment0_dir / fname
        if not img_path.exists():
            ax.text(0.5, 0.5, f"missing:\n{fname}", ha="center", va="center",
                    fontsize=8, color="red", transform=ax.transAxes)
            ax.axis("off")
            continue

        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.axis("off")

    fig.tight_layout(pad=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Moment-0 grid saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble moment-0 PNGs into a 2×3 comparison grid."
    )
    parser.add_argument("--source", required=True, help="Source name (e.g. G337, G348)")
    parser.add_argument("--moment0-dir", required=True,
                        help="Path to Compare/moment0 directory")
    parser.add_argument("--output", required=True,
                        help="Output PNG path")
    args = parser.parse_args()

    build_grid(args.source, Path(args.moment0_dir), Path(args.output))


if __name__ == "__main__":
    main()
