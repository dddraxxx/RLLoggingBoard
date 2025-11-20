#!/usr/bin/env python3
"""
Plot average tool counts per tool type over training steps.

This script inspects processed_images in RL logs to infer tool usage counts,
aggregates per step, writes a CSV, and produces a single figure with one curve
per tool type (step on X-axis, avg tool count on Y-axis).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fast_file_search import find_log_files  # noqa: E402
from fast_jsonl_reader import read_jsonl_parallel  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("rl_metric_plots_tooltype")
LOG_DIR_TOKEN = "rl_logging_board"


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot step vs tool counts per tool type using processed_images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-file",
        required=True,
        help="Path to rollout_data_rank*.jsonl file or directory containing such files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of worker processes for parallel JSONL parsing.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Minimum training step (inclusive).",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=-1,
        help="Maximum training step (inclusive). Use -1 for no limit.",
    )
    parser.add_argument(
        "--step-freq",
        type=int,
        default=1,
        help="Only include steps where step %% step_freq == 0.",
    )
    parser.add_argument(
        "--max-samples-each-step",
        type=int,
        default=1024,
        help="Global cap on samples to retain per step per file. Use <=0 for unlimited.",
    )
    parser.add_argument(
        "--per-chunk-cap",
        type=int,
        default=-1,
        help="Per-process cap when reading files. Defaults to max-samples-each-step.",
    )
    parser.add_argument(
        "--max-jsonl-num",
        type=int,
        default=None,
        help="If set, only process the first N JSONL files discovered in a directory.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars while reading JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save outputs. Defaults to rl_metric_plots_tooltype near the logs.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional explicit CSV path. Defaults to <output-dir>/tool_type_counts.csv.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit path for the combined plot. Defaults to "
            "<output-dir>/tool_type_counts_all.png. Task-specific plots will be "
            "saved alongside using datasource-specific filenames."
        ),
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=12.0,
        help="Matplotlib figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=6.0,
        help="Matplotlib figure height in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI when saving the plot.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra log messages during processing.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_output_dir(base_output_dir: Path, log_file: str) -> Path:
    if base_output_dir != DEFAULT_OUTPUT_DIR:
        return base_output_dir
    log_file_abs = os.path.abspath(log_file)
    lower_path = log_file_abs.lower()
    token_index = lower_path.find(LOG_DIR_TOKEN)
    if token_index != -1:
        replaced = (
            log_file_abs[:token_index]
            + DEFAULT_OUTPUT_DIR.name
            + log_file_abs[token_index + len(LOG_DIR_TOKEN) :]
        )
        return Path(replaced).parent
    log_path = Path(log_file_abs)
    parent = log_path if log_path.is_dir() else log_path.parent
    return parent / DEFAULT_OUTPUT_DIR.name


def sanitize_datasource_name(name: Any) -> str:
    text = str(name or "unknown")
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in text.lower())
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_") or "unknown"


def discover_log_files(log_path: str, max_jsonl_num: Optional[int], verbose: bool) -> List[str]:
    abs_path = os.path.abspath(log_path)
    if os.path.isdir(abs_path):
        if verbose:
            print(f"[info] Directory provided; scanning for rollout JSONL files under {abs_path}")
        discovered = find_log_files(abs_path)
        if discovered:
            log_files = [os.path.join(abs_path, rel) for rel in discovered]
        else:
            log_files = sorted(str(p) for p in Path(abs_path).glob("**/*.jsonl"))
            if verbose and not log_files:
                print("[warn] No JSONL files found.")
    elif os.path.isfile(abs_path):
        log_files = [abs_path]
    else:
        raise FileNotFoundError(f"Log path does not exist: {log_path}")

    log_files = sorted(log_files)
    if not log_files:
        raise FileNotFoundError(f"No JSONL files discovered under {log_path}")

    if max_jsonl_num is not None and max_jsonl_num > 0:
        if len(log_files) > max_jsonl_num and verbose:
            print(f"[info] Limiting to first {max_jsonl_num} file(s).")
        log_files = log_files[:max_jsonl_num]

    return log_files


def merge_step_payloads(
    accumulator: MutableMapping[int, Dict[str, List[Any]]],
    step_payloads: Optional[Dict[int, Dict[str, List[Any]]]],
) -> None:
    if not step_payloads:
        return
    for step, payload in step_payloads.items():
        bucket = accumulator.setdefault(step, {})
        for key, values in payload.items():
            if isinstance(values, list):
                bucket.setdefault(key, []).extend(values)
            else:
                bucket.setdefault(key, []).append(values)


def extract_tool_type_counts(processed_images: Any) -> Dict[str, int]:
    """
    Return a dict of {tool_type: count} for a single sample based on processed_images.
    The first two entries (input/origin) are ignored; the rest indicate tool usage.
    """
    counts: Dict[str, int] = {}
    if not isinstance(processed_images, list):
        return counts
    for entry in processed_images[2:]:
        if isinstance(entry, dict):
            tool_name = entry.get("tool")
            if tool_name:
                counts[tool_name] = counts.get(tool_name, 0) + 1
    return counts


def compute_step_tool_counts(step_payloads: Dict[int, Dict[str, List[Any]]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for step in sorted(step_payloads):
        payload = step_payloads[step]
        processed_entries = payload.get("processed_images") or []
        data_sources = payload.get("data_source") or []

        datasource_maps: Dict[str, List[Dict[str, int]]] = {}

        for idx, entry in enumerate(processed_entries):
            counts = extract_tool_type_counts(entry)
            if not counts:
                continue
            raw_ds = data_sources[idx] if idx < len(data_sources) else "unknown"
            ds_key = sanitize_datasource_name(raw_ds)
            datasource_maps.setdefault(ds_key, []).append(counts)

        for ds_key, count_maps in datasource_maps.items():
            if not count_maps:
                continue

            all_tools = set().union(*(mapping.keys() for mapping in count_maps))
            record = {
                "step": step,
                "datasource": ds_key,
                "sample_count": len(count_maps),
            }
            for tool in sorted(all_tools):
                values = [mapping.get(tool, 0) for mapping in count_maps]
                record[f"avg_{tool}"] = float(np.mean(values))
            records.append(record)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values(["datasource", "step"])


def plot_tool_counts_overall(df: pd.DataFrame, plot_path: Path, figsize: tuple[float, float], dpi: int) -> bool:
    numeric_cols = [col for col in df.columns if col.startswith("avg_")]
    if not numeric_cols:
        print("[warn] No tool count columns to plot.")
        return False

    grouped = df.groupby("step")
    if grouped.ngroups == 0:
        print("[warn] No data to plot for overall aggregation.")
        return False

    aggregated: List[Dict[str, float]] = []
    for step, group in grouped:
        entry = {"step": step}
        weights = group["sample_count"].fillna(0).to_numpy()
        for col in numeric_cols:
            values = group[col].fillna(0).to_numpy()
            if weights.sum() > 0:
                entry[col] = float(np.average(values, weights=weights))
            else:
                entry[col] = float(np.mean(values))
        aggregated.append(entry)

    if not aggregated:
        print("[warn] Aggregation produced no overall data.")
        return False

    agg_df = pd.DataFrame(aggregated).sort_values("step")

    plt.figure(figsize=figsize, dpi=dpi)
    for column in numeric_cols:
        plt.plot(agg_df["step"], agg_df[column], marker="o", linewidth=1.5, label=column.replace("avg_", ""))
    plt.xlabel("Step")
    plt.ylabel("Average tool count")
    plt.title("Average tool counts per type (all datasources)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[info] Saved combined tool-type plot -> {plot_path}")
    return True


def plot_tool_counts_per_datasource(
    df: pd.DataFrame,
    output_dir: Path,
    figsize: tuple[float, float],
    dpi: int,
) -> List[Path]:
    numeric_cols = [col for col in df.columns if col.startswith("avg_")]
    if not numeric_cols:
        return []

    plot_paths: List[Path] = []
    for ds in sorted(df["datasource"].unique()):
        ds_df = df[df["datasource"] == ds].sort_values("step")
        if ds_df.empty:
            continue
        plt.figure(figsize=figsize, dpi=dpi)
        for column in numeric_cols:
            plt.plot(ds_df["step"], ds_df[column], marker="o", linewidth=1.5, label=column.replace("avg_", ""))
        plt.xlabel("Step")
        plt.ylabel("Average tool count")
        plt.title(f"Average tool counts ({ds})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plot_path = output_dir / f"tool_type_counts_{ds}.png"
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        print(f"[info] Saved {ds} tool-type plot -> {plot_path}")

    return plot_paths


def main() -> None:
    args = parse_args()
    log_files = discover_log_files(args.log_file, args.max_jsonl_num, verbose=args.verbose)
    if args.verbose:
        print(f"[info] Found {len(log_files)} log file(s).")

    merged_steps: Dict[int, Dict[str, List[Any]]] = {}
    keys_to_collect = ["processed_images", "data_source"]

    for index, log_file in enumerate(log_files, start=1):
        if args.verbose:
            print(f"[info] Loading log ({index}/{len(log_files)}): {log_file}")
        payloads = read_jsonl_parallel(
            file_path=log_file,
            workers=args.workers,
            start_step=args.start_step,
            end_step=args.end_step,
            step_freq=args.step_freq,
            max_samples_each_step=args.max_samples_each_step,
            keys_to_collect=keys_to_collect,
            per_chunk_cap=args.per_chunk_cap,
            verbose=args.verbose,
            show_progress=not args.no_progress,
            key_defaults=None,
        )
        merge_step_payloads(merged_steps, payloads)

    if not merged_steps:
        raise RuntimeError("No processed_images payloads were loaded.")

    df = compute_step_tool_counts(merged_steps)
    if df.empty:
        raise RuntimeError("No tool counts could be computed from processed_images.")

    output_dir = resolve_output_dir(args.output_dir, args.log_file)
    ensure_output_dir(output_dir)

    csv_path = args.csv_path or (output_dir / "tool_type_counts.csv")
    df.to_csv(csv_path, index=False)
    print(f"[info] Wrote tool-type counts CSV -> {csv_path}")

    plot_path = args.plot_path or (output_dir / "tool_type_counts_all.png")
    plot_tool_counts_overall(df, plot_path, (args.fig_width, args.fig_height), args.dpi)
    plot_tool_counts_per_datasource(df, output_dir, (args.fig_width, args.fig_height), args.dpi)


if __name__ == "__main__":
    main()
