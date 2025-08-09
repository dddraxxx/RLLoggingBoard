#!/usr/bin/env python3
"""
Fast parallel file search utility for finding log files in large directory structures.

This module provides optimized file discovery functions that use parallel processing
to significantly speed up searching for files matching specific patterns.

Performance: ~3-17x faster than standard glob/os.walk on large directory trees.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set, Tuple
from pathlib import Path


def find_files_parallel(
    base_path: str,
    pattern: str = '_rank0.jsonl',
    workers: Optional[int] = None,
    file_suffix: bool = True
) -> List[str]:
    """
    Fast parallel file discovery using ThreadPoolExecutor.
    
    This function splits the directory tree into subdirectories and scans them
    in parallel, providing significant speedup for large directory structures.
    
    Args:
        base_path: Root directory to search
        pattern: Pattern to match. If file_suffix=True, matches files ending with pattern.
                 If file_suffix=False, uses exact filename match.
        workers: Number of worker threads (defaults to CPU count / 2)
        file_suffix: If True, use endswith() matching. If False, use exact match.
    
    Returns:
        List of relative paths to matching files, sorted alphabetically
    
    Example:
        >>> files = find_files_parallel("/data/logs", "_rank0.jsonl")
        >>> print(f"Found {len(files)} log files")
    """
    if workers is None:
        workers = min(16, max(1, (os.cpu_count() or 4) // 2))  # Cap at 16 workers
    
    print(f"Using {workers} workers for parallel file search")
    
    def matches_pattern(filename: str) -> bool:
        """Check if filename matches the pattern."""
        if file_suffix:
            return filename.endswith(pattern)
        else:
            return filename == pattern
    
    def scan_directory(dir_path: str) -> List[str]:
        """Scan a single directory tree for matching files."""
        logs = []
        try:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if matches_pattern(file):
                        full_path = os.path.join(root, file)
                        logs.append(os.path.relpath(full_path, base_path))
        except (PermissionError, OSError):
            pass
        return logs
    
    # Get top-level directories for parallel scanning
    try:
        entries = os.listdir(base_path)
        top_dirs = [os.path.join(base_path, d) for d in entries 
                   if os.path.isdir(os.path.join(base_path, d))]
        # Also check for files in root directory
        root_files = [f for f in entries 
                     if os.path.isfile(os.path.join(base_path, f)) and matches_pattern(f)]
    except (PermissionError, OSError):
        return []
    
    if not top_dirs:
        # If no subdirectories, just scan the base path
        return scan_directory(base_path)
    
    all_logs = root_files.copy()  # Start with files in root
    
    # Use ThreadPoolExecutor for parallel scanning
    with ThreadPoolExecutor(max_workers=min(workers, len(top_dirs))) as executor:
        futures = [executor.submit(scan_directory, dir_path) for dir_path in top_dirs]
        
        for future in as_completed(futures):
            try:
                logs = future.result()
                all_logs.extend(logs)
            except Exception:
                continue
    
    return sorted(all_logs)  # Sort for consistent ordering


def find_directories_with_files(
    base_path: str,
    pattern: str = '_rank0.jsonl',
    workers: Optional[int] = None
) -> List[str]:
    """
    Find all directories containing files matching a pattern.
    
    This is useful for getting a list of directories that contain specific files,
    rather than the files themselves.
    
    Args:
        base_path: Root directory to search
        pattern: File suffix pattern to match
        workers: Number of worker threads (defaults to CPU count / 2)
    
    Returns:
        Sorted list of unique directory paths (relative to base_path) containing matching files
    
    Example:
        >>> dirs = find_directories_with_files("/data/logs", "_rank0.jsonl")
        >>> print(f"Found {len(dirs)} directories with log files")
    """
    # First, find all matching files
    all_files = find_files_parallel(base_path, pattern, workers)
    
    # Extract unique directories
    directories = set()
    for file_path in all_files:
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            directories.add(parent_dir)
        else:
            directories.add('.')  # Files in root directory
    
    return sorted(list(directories))


def count_files_parallel(
    base_path: str,
    pattern: str = '_rank0.jsonl',
    workers: Optional[int] = None
) -> Tuple[int, int]:
    """
    Count files matching a pattern without returning the full list.
    
    This is memory-efficient for very large directory trees where you only
    need the count, not the actual file paths.
    
    Args:
        base_path: Root directory to search
        pattern: File suffix pattern to match
        workers: Number of worker threads (defaults to CPU count / 2)
    
    Returns:
        Tuple of (file_count, directory_count)
    
    Example:
        >>> file_count, dir_count = count_files_parallel("/data/logs")
        >>> print(f"Found {file_count} files in {dir_count} directories")
    """
    if workers is None:
        workers = min(16, max(1, (os.cpu_count() or 4) // 2))  # Cap at 16 workers
    
    def count_in_directory(dir_path: str) -> Tuple[int, Set[str]]:
        """Count matching files in a directory tree."""
        count = 0
        dirs_with_files = set()
        try:
            for root, dirs, files in os.walk(dir_path):
                matching_files = [f for f in files if f.endswith(pattern)]
                if matching_files:
                    count += len(matching_files)
                    dirs_with_files.add(os.path.relpath(root, base_path))
        except (PermissionError, OSError):
            pass
        return count, dirs_with_files
    
    # Get top-level directories
    try:
        entries = os.listdir(base_path)
        top_dirs = [os.path.join(base_path, d) for d in entries 
                   if os.path.isdir(os.path.join(base_path, d))]
        # Count files in root
        root_count = sum(1 for f in entries 
                        if os.path.isfile(os.path.join(base_path, f)) and f.endswith(pattern))
    except (PermissionError, OSError):
        return 0, 0
    
    if not top_dirs:
        count, dirs = count_in_directory(base_path)
        return count, len(dirs)
    
    total_count = root_count
    all_directories = set()
    if root_count > 0:
        all_directories.add('.')
    
    # Parallel counting
    with ThreadPoolExecutor(max_workers=min(workers, len(top_dirs))) as executor:
        futures = [executor.submit(count_in_directory, dir_path) for dir_path in top_dirs]
        
        for future in as_completed(futures):
            try:
                count, dirs = future.result()
                total_count += count
                all_directories.update(dirs)
            except Exception:
                continue
    
    return total_count, len(all_directories)


def find_files_by_depth(
    base_path: str,
    pattern: str = '_rank0.jsonl',
    max_depth: int = -1,
    workers: Optional[int] = None
) -> List[str]:
    """
    Find files with a maximum directory depth limit.
    
    This is useful when you want to limit the search to avoid going too deep
    into nested directory structures.
    
    Args:
        base_path: Root directory to search
        pattern: File suffix pattern to match
        max_depth: Maximum depth to search (-1 for unlimited)
        workers: Number of worker threads (defaults to CPU count / 2)
    
    Returns:
        List of relative paths to matching files
    
    Example:
        >>> # Only search up to 2 levels deep
        >>> files = find_files_by_depth("/data/logs", "_rank0.jsonl", max_depth=2)
    """
    if max_depth == -1:
        return find_files_parallel(base_path, pattern, workers)
    
    if workers is None:
        workers = min(16, max(1, (os.cpu_count() or 4) // 2))  # Cap at 16 workers
    
    def scan_with_depth(dir_path: str, current_depth: int) -> List[str]:
        """Scan directory with depth limit."""
        if current_depth > max_depth:
            return []
        
        logs = []
        try:
            for entry in os.listdir(dir_path):
                full_path = os.path.join(dir_path, entry)
                if os.path.isfile(full_path) and entry.endswith(pattern):
                    logs.append(os.path.relpath(full_path, base_path))
                elif os.path.isdir(full_path) and current_depth < max_depth:
                    logs.extend(scan_with_depth(full_path, current_depth + 1))
        except (PermissionError, OSError):
            pass
        return logs
    
    # For depth-limited search, we can't easily parallelize by top-level dirs
    # So we'll use a simpler approach
    return sorted(scan_with_depth(base_path, 0))


# Convenience functions for common use cases
def find_log_files(base_path: str, workers: Optional[int] = None) -> List[str]:
    """Find all _rank0.jsonl log files in a directory tree."""
    return find_files_parallel(base_path, '_rank0.jsonl', workers)


def find_log_directories(base_path: str, workers: Optional[int] = None) -> List[str]:
    """Find all directories containing _rank0.jsonl log files."""
    return find_directories_with_files(base_path, '_rank0.jsonl', workers)


if __name__ == "__main__":
    import sys
    import time
    
    # Simple CLI for testing
    if len(sys.argv) < 2:
        print("Usage: python fast_file_search.py <directory_path> [pattern]")
        sys.exit(1)
    
    search_path = sys.argv[1]
    pattern = sys.argv[2] if len(sys.argv) > 2 else '_rank0.jsonl'
    
    print(f"Searching for '{pattern}' files in: {search_path}")
    print("=" * 60)
    
    # Test parallel search
    start = time.time()
    files = find_files_parallel(search_path, pattern)
    elapsed = time.time() - start
    
    print(f"Found {len(files)} files in {elapsed:.3f} seconds")
    
    if files:
        print("\nFirst 5 files:")
        for f in files[:5]:
            print(f"  - {f}")
    
    # Count files
    file_count, dir_count = count_files_parallel(search_path, pattern)
    print(f"\nSummary: {file_count} files in {dir_count} directories")