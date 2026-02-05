import argparse
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUTS = [
    ("global_to_shared_async_constexpr.csv", "Global->Shared"),
    ("shared_to_global.csv", "Shared->Global"),
]


def ensure_out_dirs():
    Path("png").mkdir(exist_ok=True)
    Path("pdf").mkdir(exist_ok=True)
    Path("csv").mkdir(exist_ok=True)


def load_csv(path: Path, direction_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"input csv not found: {path}")
    df = pd.read_csv(path)
    required = {"SizeKB", "Type", "BandwidthGBps"}
    missing = required - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing columns: {missing_list}")
    df = df[["SizeKB", "Type", "BandwidthGBps"]].copy()
    df["Direction"] = direction_label
    return df


def save_legend(labels, save_opts, legend_cols=1, legend_fontsize=11, stem="output_legend"):
    fig, ax = plt.subplots(figsize=(6, 4))
    handles = []
    for label in labels:
        (line,) = ax.plot([], [], marker="o", label=label)
        handles.append(line)
    ax.legend(
        handles=handles,
        labels=labels,
        fontsize=legend_fontsize,
        loc="center",
        ncol=legend_cols,
        frameon=False,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"png/{stem}.png", bbox_inches="tight", **save_opts)
    fig.savefig(f"pdf/{stem}.pdf", bbox_inches="tight", **save_opts)
    plt.close(fig)


def plot_direction(df: pd.DataFrame, direction: str, save_opts, grid_opts):
    pivot = (
        df.pivot(index="SizeKB", columns="Type", values="BandwidthGBps")
        .sort_index()
    )
    sizes = list(pivot.index)
    x = range(len(sizes))

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in pivot.columns:
        ax.plot(x, pivot[col].values, marker="o", label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_title(f"Bandwidth: {direction}")
    ax.set_xlabel("Size (KB)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.grid(**grid_opts)
    ax.legend(loc="upper left")
    fig.tight_layout()
    stem = f"output_{direction.lower().replace('>', 'to').replace(' ', '_').replace('-', '_')}"
    fig.savefig(f"png/{stem}.png", **save_opts)
    fig.savefig(f"pdf/{stem}.pdf", **save_opts)
    plt.close(fig)
    return list(pivot.columns)


def build_inputs(args):
    if args.input:
        inputs = [Path(p) for p in args.input]
        if args.label and len(args.label) != len(inputs):
            raise ValueError("--label must match the number of --input entries")
        labels = []
        for idx, path in enumerate(inputs):
            if args.label:
                labels.append(args.label[idx])
            else:
                labels.append(path.stem)
        return list(zip(inputs, labels))
    return [(Path(p), label) for p, label in DEFAULT_INPUTS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="input CSV file (repeatable). default: known outputs",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="label for each --input (repeatable)",
    )
    parser.add_argument(
        "--combined-csv",
        default="csv/plot_data.csv",
        help="combined output CSV path (default: csv/plot_data.csv)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="plot graphs only (skip legend-only output)",
    )
    parser.add_argument(
        "--legend-cols",
        type=int,
        default=1,
        help="number of columns in legend-only output",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=int,
        default=11,
        help="legend font size (default: 11)",
    )
    args = parser.parse_args()

    ensure_out_dirs()
    inputs = build_inputs(args)

    frames = []
    for path, label in inputs:
        frames.append(load_csv(path, label))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["Direction", "Type", "SizeKB"])
    Path(args.combined_csv).parent.mkdir(exist_ok=True)
    df.to_csv(args.combined_csv, index=False)

    grid_opts = dict(which="both", linestyle="--", linewidth=0.5)
    save_opts = dict(dpi=300)

    all_labels = set()
    for direction in df["Direction"].unique():
        labels = plot_direction(df[df["Direction"] == direction], direction, save_opts, grid_opts)
        all_labels.update(labels)

    if not args.plot_only:
        save_legend(
            sorted(all_labels),
            save_opts,
            legend_cols=args.legend_cols,
            legend_fontsize=args.legend_fontsize,
        )


if __name__ == "__main__":
    main()
