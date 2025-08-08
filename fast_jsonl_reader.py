#!/usr/bin/env python3
"""
A fast, parallel JSONL reader for RL logs to test chunked multi-process parsing.

Features:
- Splits a JSONL file into newline-aligned chunks and parses in parallel.
- Applies step constraints early: start_step, end_step, step_freq.
- Enforces a per-chunk cap per step to bound memory and work.
- Merges results and enforces a global max_samples_each_step cap.
- Defaults to a light set of keys to avoid heavy payloads; customizable via CLI.

Usage example (CLI):
  python3 RLLoggingBoard/fast_jsonl_reader.py \
    --file /path/to/rollout_data_rank0.jsonl \
    --workers 8 --max-samples-each-step 256 --start-step 0 --end-step -1 --step-freq 1

Usage example (Python):
  from fast_jsonl_reader import read_jsonl_parallel

  # Simple usage - just pass file path
  data = read_jsonl_parallel("/path/to/rollout_data.jsonl")

  # With custom parameters
  data = read_jsonl_parallel(
      file_path="/path/to/rollout_data.jsonl",
      workers=8,
      start_step=0,
      end_step=100,
      max_samples_each_step=256,
      keys_to_collect=["prompt", "response", "reward"],
      verbose=True
  )

  # Access the data
  for step, step_data in data.items():
      prompts = step_data["prompt"]
      responses = step_data["response"]
      rewards = step_data["reward"]

This script does NOT change Streamlit code; it's for performance testing the reader.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Prefer orjson if available for speed; fallback to stdlib json
try:
    import orjson  # type: ignore

    def json_loads(raw: bytes) -> Any:
        return orjson.loads(raw)

except Exception:  # pragma: no cover
    import json

    def json_loads(raw: bytes) -> Any:
        return json.loads(raw.decode("utf-8"))


# A light default set of keys to collect to avoid heavy memory usage by default.
DEFAULT_KEYS: Tuple[str, ...] = (
    "prompt",
    "response",
    "ref_response",
    "reward",
    "ref_reward",
    "image_path",
)


def compute_newline_aligned_chunks(file_path: str, num_chunks: int) -> List[Tuple[int, int]]:
    """Compute roughly equal-sized byte ranges aligned to newline boundaries.

    Args:
        file_path: Absolute path to a JSONL file.
        num_chunks: Desired number of chunks.

    Returns:
        List of (start_offset, end_offset) tuples covering the entire file.
    """
    assert num_chunks >= 1
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return []

    if num_chunks == 1:
        return [(0, file_size)]

    offsets: List[int] = [0]
    with open(file_path, "rb") as f:
        for i in range(1, num_chunks):
            tentative = (file_size * i) // num_chunks
            f.seek(tentative)
            # Advance to the next newline to avoid splitting a line
            f.readline()
            offsets.append(f.tell())
    offsets.append(file_size)

    chunks: List[Tuple[int, int]] = []
    for i in range(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        if start < end:
            chunks.append((start, end))
    return chunks


def parse_chunk(
    file_path: str,
    start_offset: int,
    end_offset: int,
    start_step: int,
    end_step: int,
    step_freq: int,
    keys_to_collect: Sequence[str],
    per_chunk_step_cap: int,
) -> Tuple[Dict[int, Dict[str, List[Any]]], int, int]:
    """Parse a chunk of the file and return per-step aggregated data.

    Applies filters early and enforces a per-chunk cap per step to reduce work.

    Args:
        file_path: Path to the JSONL file.
        start_offset: Byte offset to start reading.
        end_offset: Byte offset to stop reading (inclusive of lines starting before end).
        start_step: Minimum step (inclusive).
        end_step: Maximum step (inclusive); use -1 for no limit.
        step_freq: Only accept steps where step % step_freq == 0.
        keys_to_collect: Keys to extract from each JSON object.
        per_chunk_step_cap: Max items per step to retain within this chunk. Use <=0 for unlimited.

    Returns:
        (results, lines_read, lines_kept)
        - results: { step: { key: [values...] } }
        - lines_read: total lines scanned in this chunk
        - lines_kept: lines that passed filters and were retained
    """
    results: Dict[int, Dict[str, List[Any]]] = {}
    lines_read = 0
    lines_kept = 0

    with open(file_path, "rb") as f:
        f.seek(start_offset)

        while True:
            pos = f.tell()
            if pos >= end_offset:
                break

            line = f.readline()
            if not line:
                break
            lines_read += 1

            try:
                data = json_loads(line)
            except Exception:
                continue

            try:
                step_value = int(data.get("step", -1))
            except Exception:
                continue

            if step_value < start_step:
                continue
            if end_step != -1 and step_value > end_step:
                break
            if step_value % step_freq != 0:
                continue

            # Initialize per-step buckets lazily
            bucket = results.get(step_value)
            if bucket is None:
                bucket = {k: [] for k in keys_to_collect}
                results[step_value] = bucket

            # Enforce per-chunk cap per step to reduce memory
            if per_chunk_step_cap > 0:
                # Use first key's length as canonical length; buckets are kept in sync
                first_key = keys_to_collect[0]
                if len(bucket[first_key]) >= per_chunk_step_cap:
                    continue

            for key in keys_to_collect:
                value = data.get(key)
                if value is not None:
                    bucket[key].append(value)

            lines_kept += 1

    return results, lines_read, lines_kept


def merge_results(
    partials: Iterable[Dict[int, Dict[str, List[Any]]]],
    keys_to_collect: Sequence[str],
    global_per_step_cap: int,
) -> Dict[int, Dict[str, List[Any]]]:
    """Merge per-chunk results into a single dict and enforce global caps.

    Args:
        partials: Iterable of per-chunk results.
        keys_to_collect: Keys collected in each partial.
        global_per_step_cap: Max items per step globally; <=0 for unlimited.

    Returns:
        Merged results as { step: { key: [values...] } } respecting global caps.
    """
    merged: Dict[int, Dict[str, List[Any]]] = {}
    for part in partials:
        for step, step_dict in part.items():
            dest = merged.get(step)
            if dest is None:
                dest = {k: [] for k in keys_to_collect}
                merged[step] = dest

            if global_per_step_cap > 0:
                # Determine how many more we can add for this step
                current_len = len(dest[keys_to_collect[0]])
                remaining = max(0, global_per_step_cap - current_len)
                if remaining <= 0:
                    continue
                # Append up to remaining items
                for k in keys_to_collect:
                    dest[k].extend(step_dict[k][:remaining])
            else:
                for k in keys_to_collect:
                    dest[k].extend(step_dict[k])

    return merged


def read_jsonl_parallel(
    file_path: str,
    workers: Optional[int] = None,
    start_step: int = 0,
    end_step: int = -1,
    step_freq: int = 1,
    max_samples_each_step: int = 256,
    keys_to_collect: Optional[List[str]] = None,
    per_chunk_cap: int = -1,
    verbose: bool = False,
    show_progress: bool = False,
) -> Dict[int, Dict[str, List[Any]]]:
    """Easy-to-use wrapper for parallel JSONL reading.

    Args:
        file_path: Path to the JSONL file to read
        workers: Number of worker processes (defaults to CPU count / 2)
        start_step: Minimum step to include (inclusive)
        end_step: Maximum step to include (inclusive, -1 for no limit)
        step_freq: Only include steps where step % step_freq == 0
        max_samples_each_step: Global cap per step across entire file
        keys_to_collect: List of keys to extract from each JSON object (defaults to DEFAULT_KEYS)
        per_chunk_cap: Per-chunk cap per step (defaults to max_samples_each_step if <= 0)
        verbose: If True, print timing and summary information
        show_progress: If True, show a tqdm progress bar (requires tqdm)

    Returns:
        Dictionary mapping step numbers to collected data:
        { step: { key: [values...] } }
    """
    # Handle defaults
    if workers is None:
        workers = max(1, (os.cpu_count() or 4) // 2)
    else:
        workers = max(1, workers)

    if keys_to_collect is None:
        keys_to_collect = list(DEFAULT_KEYS)

    if per_chunk_cap <= 0:
        per_chunk_cap = max_samples_each_step

    # Ensure absolute path
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get file size for progress display
    file_size = os.path.getsize(file_path)
    size_unit = 'MB' if file_size > 1_000_000 else 'KB' if file_size > 1_000 else 'bytes'
    size_factor = 1_000_000 if size_unit == 'MB' else 1_000 if size_unit == 'KB' else 1
    total_size = file_size / size_factor

    # Compute chunks
    start_time = time.time()
    chunks = compute_newline_aligned_chunks(file_path, workers)
    chunk_compute_ms = (time.time() - start_time) * 1000.0

    if not chunks:
        if verbose:
            print("Empty file or no chunks to process.")
        return {}

    if verbose:
        print(f"File: {file_path}")
        print(f"Size: {file_size:,} bytes; Workers: {workers}; Chunks: {len(chunks)}")
        print(f"Offsets computed in {chunk_compute_ms:.1f} ms")

    # Parse chunks in parallel
    parse_start = time.time()
    partial_results: List[Optional[Dict[int, Dict[str, List[Any]]]]] = [None] * len(chunks)
    total_lines_read = 0
    total_lines_kept = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_index = {}
        for idx, (start_off, end_off) in enumerate(chunks):
            fut = executor.submit(
                parse_chunk,
                file_path,
                start_off,
                end_off,
                start_step,
                end_step,
                step_freq,
                keys_to_collect,
                per_chunk_cap,
            )
            future_to_index[fut] = idx

        # Setup progress tracking
        futures_list = list(future_to_index.keys())
        completed_count = 0

        # Use tqdm if requested and available
        if show_progress and TQDM_AVAILABLE:
            progress_bar = tqdm(
                total=len(chunks),
                desc=f"Processing {total_size:,.1f} {size_unit}",
                unit="chunks",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )
        else:
            progress_bar = None

        for fut in as_completed(futures_list):
            idx = future_to_index[fut]
            result, lines_read, lines_kept = fut.result()
            partial_results[idx] = result
            total_lines_read += lines_read
            total_lines_kept += lines_kept

            completed_count += 1

            if progress_bar:
                # Show chunk-level progress instead of overall progress
                # Display current chunk being processed and lines within it
                progress_bar.set_description(f"Chunk {idx+1}/{len(chunks)}: {lines_read:,} lines")
                progress_bar.set_postfix({
                    'total_read': f'{total_lines_read:,}',
                    'total_kept': f'{total_lines_kept:,}',
                    'keep_rate': f'{total_lines_kept/max(1,total_lines_read)*100:.1f}%'
                })
                progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

    # Filter out None results
    partial_results = [r for r in partial_results if r is not None]
    parse_ms = (time.time() - parse_start) * 1000.0

    # Merge results
    merge_start = time.time()

    # Show merge progress if progress is enabled
    if show_progress and TQDM_AVAILABLE:
        print("\nMerging results from all chunks...")

    merged = merge_results(
        partials=partial_results,
        keys_to_collect=keys_to_collect,
        global_per_step_cap=max_samples_each_step
    )
    if show_progress and TQDM_AVAILABLE:
        print(f"Merged {len(merged)} steps in {merge_ms:.1f}ms")
    merge_ms = (time.time() - merge_start) * 1000.0
    total_ms = (time.time() - start_time) * 1000.0

    if verbose:
        steps = list(merged.keys())
        total_steps = len(steps)
        print(f"\n=== Parallel Reader Summary ===")
        print(f"Lines scanned: {total_lines_read:,}; Lines kept: {total_lines_kept:,}")
        print(f"Unique steps: {total_steps}")
        if steps:
            print(f"Step range: {min(steps)} .. {max(steps)}")
            first_key = keys_to_collect[0] if keys_to_collect else None
            if first_key:
                total_items = sum(len(merged[s][first_key]) for s in steps)
                print(f"Total items: {total_items:,}")
        print(f"Timing: parse={parse_ms:.1f}ms, merge={merge_ms:.1f}ms, total={total_ms:.1f}ms")

    return merged


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel JSONL reader test for RL logs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--file", required=True, help="Absolute path to a JSONL file to parse")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Number of worker processes")
    parser.add_argument("--start-step", type=int, default=0, help="Minimum step (inclusive)")
    parser.add_argument("--end-step", type=int, default=-1, help="Maximum step (inclusive); -1 for no limit")
    parser.add_argument("--step-freq", type=int, default=1, help="Only keep steps where step % step_freq == 0")
    parser.add_argument(
        "--max-samples-each-step",
        type=int,
        default=256,
        help="Global cap per step across entire file; <=0 for unlimited",
    )
    parser.add_argument(
        "--keys",
        type=str,
        default=",".join(DEFAULT_KEYS),
        help="Comma-separated keys to collect from each JSON object",
    )
    parser.add_argument(
        "--per-chunk-cap",
        type=int,
        default=-1,
        help="Per-chunk cap per step; if <=0, defaults to max-samples-each-step",
    )
    parser.add_argument(
        "--show-summary-steps",
        type=int,
        default=10,
        help="Print up to this many step counts in summary",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show a progress bar while processing (requires tqdm)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    # Parse keys to collect
    keys_to_collect: List[str] = [k.strip() for k in args.keys.split(",") if k.strip()]
    if not keys_to_collect:
        keys_to_collect = list(DEFAULT_KEYS)

    # Use the modular function
    print(f"Reading file: {args.file}")
    print(f"Parameters: workers={args.workers}, start_step={args.start_step}, end_step={args.end_step}, "
          f"step_freq={args.step_freq}, max_samples_each_step={args.max_samples_each_step}")
    print(f"Collecting keys: {keys_to_collect}")

    if not TQDM_AVAILABLE and args.show_progress:
        print("Note: Install tqdm for better progress bars: pip install tqdm")

    start_time = time.time()

    # Call the modular function with verbose=True to get detailed output
    # Use show_progress flag from command line arguments
    data = read_jsonl_parallel(
        file_path=args.file,
        workers=args.workers,
        start_step=args.start_step,
        end_step=args.end_step,
        step_freq=args.step_freq,
        max_samples_each_step=args.max_samples_each_step,
        keys_to_collect=keys_to_collect,
        per_chunk_cap=args.per_chunk_cap,
        verbose=True,
        show_progress=args.show_progress  # Respect the CLI flag
    )

    total_time = (time.time() - start_time) * 1000.0

    # Print additional summary if requested
    if args.show_summary_steps > 0 and data:
        steps = sorted(data.keys())
        display_steps = steps[:min(len(steps), args.show_summary_steps)]
        print("\nSample step counts:")
        for step in display_steps:
            first_key = keys_to_collect[0] if keys_to_collect else None
            if first_key and first_key in data[step]:
                count = len(data[step][first_key])
                print(f"  step={step}: {count} items")

    print(f"\nTotal execution time: {total_time:.1f} ms")


if __name__ == "__main__":
    main()

"""
----------------------------------------
Quick Command Reference (examples)

Test time for reading (parallel only)
Prints offsets time and total parallel time

python3 /scratch/doqihu/RLLoggingBoard/fast_jsonl_reader.py \
  --file /scratch/doqihu/work/verl_logs/agent_vlagent/dpev2_7b_zoom_lowres/20250807_091729/logs/rl_logging_board/agent_vlagent/dpev2_7b_zoom_lowres/rollout_data_rank0.jsonl \
  --workers 8 \
  --max-samples-each-step 256000 \
  --start-step 2 --end-step 2 --step-freq 1
"""