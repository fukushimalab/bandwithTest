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


def ensure_out_dirs(output_dir: Path):
    (output_dir / "png").mkdir(parents=True, exist_ok=True)
    (output_dir / "pdf").mkdir(parents=True, exist_ok=True)
    (output_dir / "csv").mkdir(parents=True, exist_ok=True)


def resolve_input_path(path_text: str, input_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return input_dir / path


def resolve_output_path(path_text: str, output_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return output_dir / path


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


def save_legend(
    labels,
    output_dir: Path,
    save_opts,
    legend_cols=1,
    legend_fontsize=11,
    stem="output_legend",
):
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
    fig.savefig(output_dir / "png" / f"{stem}.png", bbox_inches="tight", **save_opts)
    fig.savefig(output_dir / "pdf" / f"{stem}.pdf", bbox_inches="tight", **save_opts)
    plt.close(fig)


def plot_direction(df: pd.DataFrame, direction: str, output_dir: Path, save_opts, grid_opts):
    pivot = (
        df.pivot_table(index="SizeKB", columns="Type", values="BandwidthGBps", aggfunc="first")
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
    fig.savefig(output_dir / "png" / f"{stem}.png", **save_opts)
    fig.savefig(output_dir / "pdf" / f"{stem}.pdf", **save_opts)
    plt.close(fig)
    return list(pivot.columns)


def plot_merged(df: pd.DataFrame, output_dir: Path, save_opts, grid_opts):
    pivot = (
        df.pivot_table(
            index="SizeKB",
            columns=["Direction", "Type"],
            values="BandwidthGBps",
            aggfunc="first",
        ).sort_index()
    )
    sizes = list(pivot.index)
    x = range(len(sizes))

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = []
    for direction, dtype in pivot.columns:
        label = f"{direction} | {dtype}"
        ax.plot(x, pivot[(direction, dtype)].values, marker="o", label=label)
        labels.append(label)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_title("Bandwidth: Global<->Shared (Merged)")
    ax.set_xlabel("Size (KB)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.grid(**grid_opts)
    ax.legend(loc="upper left")
    fig.tight_layout()
    stem = "output_global_shared_merged"
    fig.savefig(output_dir / "png" / f"{stem}.png", **save_opts)
    fig.savefig(output_dir / "pdf" / f"{stem}.pdf", **save_opts)
    plt.close(fig)
    return labels


def build_inputs(args, input_dir: Path):
    if args.input:
        inputs = [resolve_input_path(p, input_dir) for p in args.input]
        if args.label and len(args.label) != len(inputs):
            raise ValueError("--label must match the number of --input entries")
        labels = []
        for idx, path in enumerate(inputs):
            if args.label:
                labels.append(args.label[idx])
            else:
                labels.append(path.stem)
        return list(zip(inputs, labels))
    return [(input_dir / p, label) for p, label in DEFAULT_INPUTS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=".",
        help="base directory for default CSVs and relative --input paths",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="base directory for generated png/pdf/csv outputs",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="input CSV file (repeatable, relative to --input-dir unless absolute). default: known outputs",
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
        help="combined output CSV path (relative to --output-dir unless absolute)",
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

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    ensure_out_dirs(output_dir)
    inputs = build_inputs(args, input_dir)

    frames = []
    for path, label in inputs:
        frames.append(load_csv(path, label))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["Direction", "Type", "SizeKB"])
    combined_csv_path = resolve_output_path(args.combined_csv, output_dir)
    combined_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(combined_csv_path, index=False)

    grid_opts = dict(which="both", linestyle="--", linewidth=0.5)
    save_opts = dict(dpi=300)

    for direction in df["Direction"].unique():
        plot_direction(
            df[df["Direction"] == direction],
            direction,
            output_dir,
            save_opts,
            grid_opts,
        )

    merged_labels = plot_merged(df, output_dir, save_opts, grid_opts)

    if not args.plot_only:
        save_legend(
            sorted(merged_labels),
            output_dir,
            save_opts,
            legend_cols=args.legend_cols,
            legend_fontsize=args.legend_fontsize,
        )


if __name__ == "__main__":
    main()
