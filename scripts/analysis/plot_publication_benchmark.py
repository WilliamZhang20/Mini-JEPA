"""Render the frozen-controller literature comparison used for publication."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "Ours": "#0072B2",
    "BC": "#A7A9AC",
    "CQL": "#E69F00",
    "IQL": "#009E73",
    "ReBRAC": "#CC79A7",
    "VanTA": "#56B4E9",
    "HER": "#A7A9AC",
    "HER+EBP": "#E69F00",
    "FAHER": "#009E73",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def draw_grouped(
    ax,
    rows,
    panel,
    tasks,
    methods,
    title,
    ylabel,
    ylim,
    panel_letter,
):
    width = min(0.16, 0.76 / len(methods))
    x = np.arange(len(tasks), dtype=float)
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width

    for method, offset in zip(methods, offsets):
        selected = {
            row["task"]: row
            for row in rows
            if row["panel"] == panel and row["method"] == method
        }
        values = np.asarray([float(selected[task]["value"]) for task in tasks])
        low = np.asarray([float(selected[task]["err_low"]) for task in tasks])
        high = np.asarray([float(selected[task]["err_high"]) for task in tasks])
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.88,
            color=PALETTE[method],
            edgecolor="white",
            linewidth=0.65,
            label=method,
            zorder=2,
        )
        if np.any(low > 0) or np.any(high > 0):
            ax.errorbar(
                x + offset,
                values,
                yerr=np.vstack([low, high]),
                fmt="none",
                ecolor="#2B2B2B",
                elinewidth=0.75,
                capsize=1.8,
                capthick=0.75,
                zorder=3,
            )
        if method == "Ours":
            for bar, value in zip(bars, values):
                near_ceiling = value > ylim[0] + 0.86 * (ylim[1] - ylim[0])
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value - (ylim[1] - ylim[0]) * 0.035
                    if near_ceiling
                    else value + (ylim[1] - ylim[0]) * 0.026,
                    f"{value:.0f}" if abs(value - round(value)) < 0.05 else f"{value:.1f}",
                    ha="center",
                    va="top" if near_ceiling else "bottom",
                    fontsize=8.3,
                    fontweight="bold",
                    color="white" if near_ceiling else PALETTE["Ours"],
                    clip_on=False,
                )

    ax.set_xticks(x, tasks)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=10, fontsize=11.2, fontweight="bold")
    ax.text(
        -0.10,
        1.075,
        panel_letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.72, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#707070")
    ax.tick_params(axis="both", labelsize=8.8, length=3, color="#707070")
    ax.legend(
        frameon=False,
        fontsize=7.8,
        ncol=min(len(methods), 5),
        loc="upper left",
        handlelength=1.2,
        columnspacing=0.9,
        borderaxespad=0.15,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("runs/publication_benchmark/benchmark_scores.csv"),
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("runs/publication_benchmark/protocol_aligned_comparison"),
    )
    args = parser.parse_args()
    rows = load_rows(args.scores)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7.6), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.865, bottom=0.16, wspace=0.24, hspace=0.40)

    draw_grouped(
        axes[0, 0],
        rows,
        "AntMaze",
        ["UMaze-diverse", "Medium-diverse", "Large-diverse"],
        ["BC", "CQL", "IQL", "ReBRAC", "Ours"],
        "AntMaze diverse",
        "Success / normalized score",
        (0, 112),
        "a",
    )
    draw_grouped(
        axes[0, 1],
        rows,
        "Adroit expert",
        ["Door", "Relocate"],
        ["BC", "CQL", "IQL", "ReBRAC", "Ours"],
        "Adroit expert",
        "D4RL normalized return",
        (-4, 126),
        "b",
    )
    draw_grouped(
        axes[1, 0],
        rows,
        "Kitchen partial",
        ["Four-task score"],
        ["BC", "CQL", "IQL", "VanTA", "Ours"],
        "Franka Kitchen partial",
        "Normalized task completion",
        (0, 108),
        "c",
    )
    draw_grouped(
        axes[1, 1],
        rows,
        "Fetch 50-step",
        ["PickAndPlace", "Slide"],
        ["HER", "HER+EBP", "FAHER", "Ours"],
        "Fetch manipulation · 50-step horizon",
        "Episode success (%)",
        (0, 112),
        "d",
    )

    fig.suptitle(
        "Frozen JEPA controllers under literature-aligned evaluation",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.916,
        "Evaluation only—no RL fine-tuning. Blue bars are this work (100 episodes per task).",
        ha="left",
        fontsize=10.2,
        color="#4A4A4A",
    )
    fig.text(
        0.075,
        0.055,
        "Ours: 95% episode-level CI. Literature whiskers: variability reported by each source. "
        "AntMaze and Fetch use the Gymnasium-Robotics ports (v4); literature anchors use D4RL v2 "
        "and Fetch-v1, respectively. UMaze ours uses a map-distilled discrete router. "
        "See benchmark_protocol.md for supervision and version caveats.",
        ha="left",
        va="bottom",
        fontsize=7.7,
        color="#4A4A4A",
        wrap=True,
    )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(
            args.out_prefix.with_suffix(f".{suffix}"),
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
