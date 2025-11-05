#!/usr/bin/env python3
"""
Standalone CLI to compute and plot RL logging metrics using lambda examples.

Reads large JSONL rollout logs with the parallel reader, applies selected
metrics from ``lambda_examples.py``, aggregates their outputs, and produces
curves (PNG/HTML) plus optional CSV exports.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Prefer non-interactive backend for matplotlib in CLI environments
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - plotly is optional
    go = None

from fast_jsonl_reader import read_jsonl_parallel
from fast_file_search import find_log_files
from lambda_examples_v2 import LAMBDA_EXAMPLES

MetricFunc = Callable[[Dict[str, List]], Dict[str, List]]

VALIDATION_PATH_KEYWORD = "validation_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot RL logging metrics using lambda examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log-file",
        required=True,
        help="Path to rollout_data_rank*.jsonl file or directory containing such files",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["all"],
        help="Metric names from lambda_examples. Use 'all' to include every metric.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of worker processes for parallel JSONL reading.",
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
        help="Global cap on samples kept per step. Use <=0 for unlimited.",
    )
    parser.add_argument(
        "--per-chunk-cap",
        type=int,
        default=-1,
        help="Per-process cap for each step. Defaults to max-samples-each-step.",
    )
    parser.add_argument(
        "--aggregator",
        choices=["mean", "median", "sum", "max", "min", "count"],
        default="mean",
        help="Reduction to apply when metric lambda returns a list of values.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Rolling window size for smoothing curves (>=1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rl_metric_plots"),
        help="Directory to store generated figures.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional path to export aggregated metrics as CSV.",
    )
    parser.add_argument(
        "--plot-backend",
        choices=["matplotlib", "plotly"],
        default="matplotlib",
        help="Which plotting backend to use. Plotly requires the package to be installed.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during JSONL loading.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=None,
        help="Optional limit on the number of steps to evaluate (useful for quick sanity checks).",
    )

    return parser.parse_args()


def get_selected_metrics(metric_names: Sequence[str]) -> Dict[str, MetricFunc]:
    registry = {name: func for name, func in LAMBDA_EXAMPLES}
    if not metric_names or "all" in metric_names:
        return registry

    missing = [name for name in metric_names if name not in registry]
    if missing:
        raise ValueError(f"Unknown metric(s): {', '.join(missing)}")

    return {name: registry[name] for name in metric_names}


AGGREGATORS: Dict[str, Callable[[pd.Series], float]] = {
    "mean": lambda s: float(s.mean()),
    "median": lambda s: float(s.median()),
    "sum": lambda s: float(s.sum()),
    "max": lambda s: float(s.max()),
    "min": lambda s: float(s.min()),
    "count": lambda s: float(s.count()),
}


def flatten_numeric(value: object) -> List[float]:
    """Recursively collect numeric leaves from nested structures."""
    result: List[float] = []
    if value is None:
        return result

    if isinstance(value, (float, int, np.number, bool)):
        val = float(value)
        if math.isfinite(val):
            result.append(val)
        return result

    if isinstance(value, (list, tuple, set)):
        for item in value:
            result.extend(flatten_numeric(item))
        return result

    if isinstance(value, dict):
        # Ignore dicts here; handled by caller to preserve keys as series.
        return result

    if isinstance(value, np.ndarray):
        return flatten_numeric(value.tolist())

    if isinstance(value, pd.Series):
        return flatten_numeric(value.tolist())

    return result


def extract_metric_entries(
    metric_name: str,
    data: object,
    aggregator: str,
    series_prefix: Optional[str] = None,
) -> List[Tuple[str, str, float]]:
    """Flatten lambda outputs into (metric, series, value) triples."""
    entries: List[Tuple[str, str, float]] = []
    agg_fn = AGGREGATORS[aggregator]

    if isinstance(data, dict):
        for key, value in data.items():
            sub_series = f"{series_prefix}/{key}" if series_prefix else str(key)
            entries.extend(extract_metric_entries(metric_name, value, aggregator, sub_series))
        return entries

    numeric_values = flatten_numeric(data)
    if not numeric_values:
        return entries

    series = series_prefix if series_prefix else "value"
    series_pd = pd.Series(numeric_values, dtype=float)
    aggregated = agg_fn(series_pd)
    entries.append((metric_name, series, aggregated))
    return entries


def evaluate_metrics(
    step_data: Dict[int, Dict[str, List]],
    metrics: Dict[str, MetricFunc],
    aggregator: str,
    sample_steps: Optional[int] = None,
) -> pd.DataFrame:
    """Run lambda metrics step-by-step and collect aggregated results."""
    rows: List[Dict[str, object]] = []
    steps = sorted(step_data.keys())

    if sample_steps is not None:
        steps = steps[:sample_steps]

    for step in steps:
        payload = step_data[step]
        for metric_name, metric_fn in metrics.items():
            try:
                result = metric_fn(payload)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] Metric '{metric_name}' failed on step {step}: {exc}")
                continue

            for name, series, value in extract_metric_entries(metric_name, result, aggregator):
                rows.append({"step": step, "metric": name, "series": series, "value": value})

    df = pd.DataFrame(rows, columns=["step", "metric", "series", "value"])
    return df


def apply_smoothing(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if df.empty or window <= 1:
        df["value_smoothed"] = df["value"] if "value" in df else df
        return df

    df = df.copy()
    df["value_smoothed"] = (
        df.sort_values("step")
        .groupby(["metric", "series"])["value"]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return df


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def is_validation_summary(path: str) -> bool:
    """Return True when the provided path looks like a validation summary file."""
    return VALIDATION_PATH_KEYWORD in Path(path).name.lower()


def load_validation_summary(summary_path: Path) -> pd.DataFrame:
    """
    Parse validation summary JSONL into a tidy DataFrame containing the metrics
    required for plotting.
    """
    records: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)

    with summary_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            step = int(payload.get("step", 0))
            metrics = payload.get("metrics", {})
            for key, value in metrics.items():
                parts = key.split("/")
                if len(parts) < 3:
                    continue
                _scope, dataset = parts[0], parts[1]
                metric_path = "/".join(parts[2:])

                dataset_entry = records[dataset].setdefault(step, {})
                if metric_path == "acc_reward/mean@1":
                    dataset_entry["acc_reward_mean"] = float(value)
                elif metric_path == "count_vision/mean@1":
                    dataset_entry["tool_count_mean"] = float(value)
                elif metric_path == "num_samples":
                    dataset_entry["num_samples"] = float(value)

    if not records:
        raise ValueError(f"No usable metrics located in validation summary: {summary_path}")

    rows: List[Dict[str, float]] = []
    for dataset, steps in records.items():
        for step, values in steps.items():
            rows.append(
                {
                    "dataset": dataset,
                    "step": step,
                    "acc_reward_mean": values.get("acc_reward_mean", float("nan")),
                    "tool_count_mean": values.get("tool_count_mean", float("nan")),
                    "num_samples": values.get("num_samples", 0.0),
                }
            )

    if not rows:
        raise ValueError(f"No rows could be constructed from validation summary: {summary_path}")

    df = pd.DataFrame(rows)
    df.sort_values(["dataset", "step"], inplace=True)
    return df


def plot_validation_bars(
    df: pd.DataFrame,
    output_dir: Path,
    column: str,
    title: str,
    filename: str,
    ylabel: str,
) -> None:
    """Render a simple bar chart for validation summary metrics."""
    ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["dataset"], df[column].astype(float))
    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig_path = output_dir / filename
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[info] Saved validation plot: {fig_path}")


def plot_validation_lines(
    df: pd.DataFrame,
    output_dir: Path,
    column: str,
    title: str,
    filename: str,
    ylabel: str,
) -> None:
    """Render a line chart of metric vs step for each dataset."""
    ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(12, 6))

    plotted = False
    for dataset, dataset_df in df.sort_values("step").groupby("dataset"):
        series = dataset_df.dropna(subset=[column])
        if series.empty:
            continue
        ax.plot(series["step"], series[column].astype(float), marker="o", label=dataset)
        plotted = True

    if not plotted:
        print(f"[warn] No data available to plot '{column}' for validation summary.")
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = output_dir / filename
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[info] Saved validation plot: {fig_path}")


def plot_with_matplotlib(df: pd.DataFrame, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    unique_metrics = df["metric"].unique()

    for metric_name in unique_metrics:
        metric_df = df[df["metric"] == metric_name].sort_values("step")

        # Check if we need to split by datasource (look for __ separator in series)
        datasource_groups = {}
        for series_name in metric_df["series"].unique():
            if "__" in str(series_name):
                # Extract datasource from series like "sealvqa__max_tool_count"
                datasource = str(series_name).split("__")[0]
                if datasource not in datasource_groups:
                    datasource_groups[datasource] = []
                datasource_groups[datasource].append(series_name)
            else:
                # No datasource grouping, plot all together
                datasource_groups["all"] = list(metric_df["series"].unique())
                break

        # Create separate plot for each datasource group
        for datasource, series_list in datasource_groups.items():
            datasource_df = metric_df[metric_df["series"].isin(series_list)]
            fig, ax = plt.subplots(figsize=(12, 6))

            for series_name, series_df in datasource_df.groupby("series"):
                y_values = (
                    series_df["value_smoothed"]
                    if "value_smoothed" in series_df
                    else series_df["value"]
                )
                # Remove datasource prefix from legend label
                label = str(series_name).split("__")[1] if "__" in str(series_name) else series_name
                ax.plot(series_df["step"], y_values, label=label)

            title = f"{metric_name} - {datasource}" if datasource != "all" else metric_name
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.3)

            safe_name = metric_name.replace(' ', '_').replace('/', '_')
            if datasource != "all":
                safe_name = f"{datasource}_{safe_name}"
            filename = f"{safe_name}.png"
            fig_path = output_dir / filename
            fig.tight_layout()
            fig.savefig(fig_path)
            plt.close(fig)
            print(f"[info] Saved matplotlib plot: {fig_path}")


def plot_with_plotly(df: pd.DataFrame, output_dir: Path) -> None:
    if go is None:
        raise RuntimeError("plotly is not installed. Install plotly or select matplotlib backend.")

    ensure_output_dir(output_dir)
    unique_metrics = df["metric"].unique()

    for metric_name in unique_metrics:
        metric_df = df[df["metric"] == metric_name].sort_values("step")

        # Check if we need to split by datasource (look for __ separator in series)
        datasource_groups = {}
        for series_name in metric_df["series"].unique():
            if "__" in str(series_name):
                # Extract datasource from series like "sealvqa__max_tool_count"
                datasource = str(series_name).split("__")[0]
                if datasource not in datasource_groups:
                    datasource_groups[datasource] = []
                datasource_groups[datasource].append(series_name)
            else:
                # No datasource grouping, plot all together
                datasource_groups["all"] = list(metric_df["series"].unique())
                break

        # Create separate plot for each datasource group
        for datasource, series_list in datasource_groups.items():
            datasource_df = metric_df[metric_df["series"].isin(series_list)]
            fig = go.Figure()

            for series_name, series_df in datasource_df.groupby("series"):
                y_values = (
                    series_df["value_smoothed"]
                    if "value_smoothed" in series_df
                    else series_df["value"]
                )
                # Remove datasource prefix from legend label
                label = str(series_name).split("__")[1] if "__" in str(series_name) else series_name
                fig.add_trace(
                    go.Scatter(
                        x=series_df["step"],
                        y=y_values,
                        mode="lines+markers",
                        name=label,
                    )
                )

            title = f"{metric_name} - {datasource}" if datasource != "all" else metric_name
            fig.update_layout(
                title=title,
                xaxis_title="Step",
                yaxis_title="Value",
                template="plotly_white",
            )

            safe_name = metric_name.replace(' ', '_').replace('/', '_')
            if datasource != "all":
                safe_name = f"{datasource}_{safe_name}"
            filename = f"{safe_name}.html"
            fig_path = output_dir / filename
            fig.write_html(fig_path, include_plotlyjs="cdn")
            print(f"[info] Saved plotly figure: {fig_path}")


def main() -> None:
    args = parse_args()
    log_path = os.path.abspath(args.log_file)

    # Auto-search for log files if directory is provided
    if os.path.isdir(log_path):
        print(f"[info] Directory provided, searching for rollout_data_rank*.jsonl files...")
        log_files = find_log_files(log_path)
        if not log_files:
            raise FileNotFoundError(f"No rollout_data_rank*.jsonl files found in {log_path}")
        # Use first found file by default (or combine all files)
        log_file = os.path.join(log_path, log_files[0])
        print(f"[info] Found {len(log_files)} log file(s), using: {log_file}")
        if len(log_files) > 1:
            print(f"[info] Other files found: {log_files[1:]}")
    else:
        log_file = log_path

    validation_mode = is_validation_summary(log_file)

    # Determine output directory based on log file path if not specified
    if args.output_dir == Path("rl_metric_plots"):  # User didn't specify custom output
        # Try to place output alongside rl_logging_board using the actual log_file path
        if "rl_logging_board" in log_file:
            # Replace rl_logging_board with rl_metric_plots in the log file path
            output_dir = Path(log_file.replace("rl_logging_board", "rl_metric_plots")).parent
            print(f"[info] Output directory: {output_dir}")
        else:
            # Fallback: create rl_metric_plots in the same parent directory as log file
            output_dir = Path(log_file).parent / "rl_metric_plots"
            print(f"[info] Output directory: {output_dir}")

        if validation_mode:
            output_dir = output_dir.with_name(f"{output_dir.name}_val")
            print(f"[info] Validation run detected, adjusted output directory: {output_dir}")
    else:
        output_dir = args.output_dir

    if validation_mode:
        summary_df = load_validation_summary(Path(log_file))
        plot_validation_lines(
            summary_df,
            output_dir,
            column="acc_reward_mean",
            title="Validation acc_reward mean@1 over steps",
            filename="validation_acc_reward_mean.png",
            ylabel="acc_reward (mean@1)",
        )
        plot_validation_lines(
            summary_df,
            output_dir,
            column="tool_count_mean",
            title="Validation tool count (mean@1) over steps",
            filename="validation_tool_count_mean.png",
            ylabel="tool count (mean@1)",
        )
        latest_df = (
            summary_df.sort_values("step")
            .groupby("dataset", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        plot_validation_bars(
            latest_df,
            output_dir,
            column="num_samples",
            title="Validation samples per dataset",
            filename="validation_num_samples.png",
            ylabel="num samples",
        )
        print("[info] Done.")
        return

    metrics = get_selected_metrics(args.metrics)
    if not metrics:
        raise ValueError("No metrics selected. Check --metrics argument.")

    # Reduce workers to 1 if sample_steps is small
    workers = args.workers
    if args.sample_steps is not None and args.sample_steps < 10:
        workers = 1
        print(f"[info] Reducing workers to 1 due to small sample_steps ({args.sample_steps})")

    print(f"[info] Loading log file: {log_file}")
    step_payloads = read_jsonl_parallel(
        file_path=log_file,
        workers=workers,
        start_step=args.start_step,
        end_step=args.end_step,
        step_freq=args.step_freq,
        max_samples_each_step=args.max_samples_each_step,
        keys_to_collect=None,  # Collect dynamically to satisfy lambda requirements
        per_chunk_cap=args.per_chunk_cap,
        verbose=True,
        show_progress=True,  # Always show progress
        key_defaults=None,
    )

    if not step_payloads:
        raise RuntimeError("No data loaded from the JSONL file with the provided filters.")

    print(f"[info] Evaluating {len(metrics)} metric(s) across {len(step_payloads)} steps.")
    metric_df = evaluate_metrics(
        step_data=step_payloads,
        metrics=metrics,
        aggregator=args.aggregator,
        sample_steps=args.sample_steps,
    )

    if metric_df.empty:
        raise RuntimeError("No metric values produced; check metric selection or aggregator.")

    metric_df = apply_smoothing(metric_df, max(args.smooth_window, 1))

    if args.csv_path:
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        metric_df.to_csv(args.csv_path, index=False)
        print(f"[info] Wrote aggregated metrics CSV: {args.csv_path}")

    if args.plot_backend == "matplotlib":
        plot_with_matplotlib(metric_df, output_dir)
    else:
        plot_with_plotly(metric_df, output_dir)

    print("[info] Done.")


if __name__ == "__main__":
    main()
