#!/usr/bin/env python3
"""
Standalone CLI to compute per-step average tool counts and rewards from RL logs
and visualize the relationship on a scatter plot.

This script mirrors the "read JSONL rollout logs and aggregate per step" flow
from plot_lambda_metrics.py but focuses on a single plot:

- For every training step, compute the average number of tool calls observed in
  assistant responses plus the average reward (configurable key, default
  "reward").
- Export the resulting table to CSV for downstream analysis.
- Generate a scatter plot on the tool-count-vs-reward plane, colouring points
  by the training step number for easy visual inspection.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional

import matplotlib

# Prefer a non-interactive backend for CLI usage
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fast_file_search import find_log_files  # noqa: E402
from fast_jsonl_reader import read_jsonl_parallel  # noqa: E402

# Regex patterns reused from lambda_examples to count completed tool exchanges.
TOOL_RESPONSE_PATTERN = re.compile(
    r"</tool_response><\|im_end\|>\s*<\|im_start\|>assistant", re.IGNORECASE
)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.IGNORECASE)
DEFAULT_OUTPUT_DIR = Path("rl_metric_plots_toolreward")
LOG_DIR_TOKEN = "rl_logging_board"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average tool count vs average reward per training step.",
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
        "--reward-key",
        type=str,
        default="reward",
        help="Name of the reward field to average (e.g., reward, acc_reward).",
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
        help=(
            "Directory where outputs are saved. When left at the default "
            "rl_metric_plots_toolreward, the directory is created next to the logs "
            "similar to plot_lambda_metrics.py."
        ),
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional explicit CSV path. Defaults to <output-dir>/tool_vs_reward.csv.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional explicit plot path. Defaults to <output-dir>/tool_vs_reward.png.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="If set, skip scatter plot generation and only export the CSV.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=10.0,
        help="Matplotlib figure width in inches for the scatter plot.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=6.0,
        help="Matplotlib figure height in inches for the scatter plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Matplotlib figure DPI when saving the scatter plot.",
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
    """
    Reuse the plot_lambda_metrics.py scheme: when --output-dir is left at the default,
    place artifacts next to the log file under an *_toolreward directory, otherwise
    respect the custom path verbatim.
    """
    if base_output_dir != DEFAULT_OUTPUT_DIR:
        return base_output_dir

    log_file_abs = os.path.abspath(log_file)
    log_path = Path(log_file_abs)

    lower_path = log_file_abs.lower()
    token_index = lower_path.find(LOG_DIR_TOKEN)
    if token_index != -1:
        replaced = (
            log_file_abs[:token_index]
            + DEFAULT_OUTPUT_DIR.name
            + log_file_abs[token_index + len(LOG_DIR_TOKEN) :]
        )
        return Path(replaced).parent

    parent = log_path if log_path.is_dir() else log_path.parent
    return parent / DEFAULT_OUTPUT_DIR.name


def discover_log_files(log_path: str, max_jsonl_num: Optional[int], verbose: bool) -> List[str]:
    """Resolve the log argument into a sorted list of absolute file paths."""
    abs_path = os.path.abspath(log_path)

    if os.path.isdir(abs_path):
        if verbose:
            print(f"[info] Directory provided. Searching for rollout JSONL files under {abs_path}")
        discovered = find_log_files(abs_path)
        if discovered:
            log_files = [os.path.join(abs_path, rel) for rel in discovered]
        else:
            if verbose:
                print("[warn] No rollout_data_rank*.jsonl files found. Falling back to *.jsonl scan.")
            log_files = sorted(str(p) for p in Path(abs_path).glob("**/*.jsonl"))
    elif os.path.isfile(abs_path):
        log_files = [abs_path]
    else:
        raise FileNotFoundError(f"Log path does not exist: {log_path}")

    log_files = sorted(log_files)
    if not log_files:
        raise FileNotFoundError(f"No JSONL files discovered under {log_path}")

    if max_jsonl_num is not None and max_jsonl_num > 0:
        if len(log_files) > max_jsonl_num and verbose:
            print(f"[info] Limiting to first {max_jsonl_num} files per --max-jsonl-num")
        log_files = log_files[:max_jsonl_num]

    return log_files


def merge_step_payloads(
    accumulator: MutableMapping[int, Dict[str, List[Any]]],
    step_payloads: Optional[Dict[int, Dict[str, List[Any]]]],
) -> None:
    """Merge per-step payloads from a file into the accumulator."""
    if not step_payloads:
        return

    for step, payload in step_payloads.items():
        bucket = accumulator.setdefault(step, {})
        for key, values in payload.items():
            if isinstance(values, list):
                bucket.setdefault(key, []).extend(values)
            else:  # Defensive fallback
                bucket.setdefault(key, []).append(values)


def count_tool_calls(response_text: Optional[str]) -> int:
    """Count tool calls in a single response string."""
    if not response_text or not isinstance(response_text, str):
        return 0
    count = len(TOOL_RESPONSE_PATTERN.findall(response_text))
    if count == 0:
        count = len(TOOL_CALL_PATTERN.findall(response_text))
    return count


def _safe_float(value: Any) -> Optional[float]:
    """Convert reward values to floats, ignoring invalid entries."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_reward_values(raw_values: Optional[List[Any]]) -> List[float]:
    """Flatten nested reward entries (e.g., [ [0.8], [0.7] ]) into plain floats."""
    if not raw_values:
        return []

    extracted: List[float] = []
    for value in raw_values:
        if isinstance(value, (list, tuple)):
            for item in value:
                converted = _safe_float(item)
                if converted is not None:
                    extracted.append(converted)
        else:
            converted = _safe_float(value)
            if converted is not None:
                extracted.append(converted)
    return extracted


def tool_count_from_processed_images(processed_entry: Any) -> Optional[int]:
    """Derive the tool count from processed_images (len - 2 for input & origin)."""
    if not isinstance(processed_entry, list):
        return None
    return max(len(processed_entry) - 2, 0)


def compute_step_statistics(
    step_payloads: Dict[int, Dict[str, List[Any]]],
    reward_key: str,
) -> pd.DataFrame:
    """Convert merged payloads into a tidy DataFrame with per-step averages."""
    rows: List[Dict[str, Any]] = []

    for step in sorted(step_payloads):
        payload = step_payloads[step]
        tool_counts: List[float] = []
        processed_entries = payload.get("processed_images") or []
        if processed_entries:
            for entry in processed_entries:
                count = tool_count_from_processed_images(entry)
                if count is not None:
                    tool_counts.append(count)

        response_source = "processed_images" if tool_counts else "response"
        responses = payload.get("response") or []
        if not tool_counts and responses:
            tool_counts = [count_tool_calls(resp) for resp in responses if isinstance(resp, str)]

        rewards = payload.get(reward_key)
        reward_values = extract_reward_values(rewards)
        if not reward_values and reward_key != "reward":
            reward_values = extract_reward_values(payload.get("reward"))

        avg_tool = float(np.mean(tool_counts)) if tool_counts else math.nan
        avg_reward = float(np.mean(reward_values)) if reward_values else math.nan

        rows.append(
            {
                "step": step,
                "response_count": len(processed_entries) if response_source == "processed_images" else len(responses),
                "reward_count": len(reward_values),
                "avg_tool_count": avg_tool,
                "avg_reward": avg_reward,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("step").reset_index(drop=True)


def plot_tool_vs_reward(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple[float, float],
    dpi: int,
) -> bool:
    """Render and save the scatter plot. Returns True if a plot was saved."""
    valid = df.dropna(subset=["avg_tool_count", "avg_reward"], how="any")
    if valid.empty:
        print("[warn] No valid points with both avg_tool_count and avg_reward. Skipping plot.")
        return False

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    scatter = ax.scatter(
        valid["avg_tool_count"],
        valid["avg_reward"],
        c=valid["step"],
        cmap="viridis",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.85,
    )
    ax.set_xlabel("Average tool count per step")
    ax.set_ylabel("Average reward per step")
    ax.set_title("Tool count vs reward (per step)")
    ax.grid(True, linestyle="--", alpha=0.4)
    cbar = fig.colorbar(scatter, ax=ax, label="Step")

    # Only show integer ticks on the colorbar if the range is short.
    if len(valid["step"].unique()) <= 20:
        cbar.set_ticks(sorted(valid["step"].unique()))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[info] Saved scatter plot -> {output_path}")
    return True


def main() -> None:
    args = parse_args()
    log_files = discover_log_files(args.log_file, args.max_jsonl_num, verbose=args.verbose)
    if args.verbose:
        print(f"[info] Found {len(log_files)} log file(s) to process.")

    merged_steps: Dict[int, Dict[str, List[Any]]] = {}
    keys_to_collect = ["response", "processed_images"]
    if args.reward_key not in keys_to_collect:
        keys_to_collect.append(args.reward_key)
    if args.reward_key != "reward" and "reward" not in keys_to_collect:
        keys_to_collect.append("reward")

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
        raise RuntimeError("No step payloads were loaded. Check your filters or log path.")

    df = compute_step_statistics(merged_steps, args.reward_key)
    if df.empty:
        raise RuntimeError("No step statistics computed; responses or rewards may be missing.")

    output_dir = resolve_output_dir(args.output_dir, args.log_file)
    ensure_output_dir(output_dir)
    csv_path = args.csv_path or (output_dir / "tool_vs_reward.csv")
    df.to_csv(csv_path, index=False)
    print(f"[info] Wrote step summary CSV -> {csv_path}")

    if not args.skip_plot:
        plot_path = args.plot_path or (output_dir / "tool_vs_reward.png")
        plot_tool_vs_reward(df, plot_path, (args.fig_width, args.fig_height), args.dpi)
    else:
        print("[info] Plotting skipped per --skip-plot")


if __name__ == "__main__":
    main()
