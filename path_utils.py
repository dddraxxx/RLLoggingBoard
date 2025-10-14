import os
from typing import List, Optional, Sequence, Tuple

DEFAULT_IMAGE_SEARCH_SUBDIRS = [
    'images',
    os.path.join('..', '..', 'processed_images'),
]


def resolve_image_path(
    image_path: str,
    log_dir: Optional[str] = None,
    search_subdirs: Optional[Sequence[str]] = None,
    extra_search_paths: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], List[str]]:
    """
    Attempt to resolve an image path by probing a set of fallback directories.

    Args:
        image_path: Raw image path from the log.
        log_dir: Directory containing the current log files.
        search_subdirs: Relative directories (from `log_dir`) to probe, joined with the
            image basename.
        extra_search_paths: Additional directories to probe; relative entries are
            considered relative to `log_dir`.

    Returns:
        Tuple of (resolved absolute path or None, ordered list of attempted paths).
    """
    if not image_path:
        return None, []

    attempted: List[str] = []
    seen = set()

    def add_candidate(candidate: str) -> None:
        candidate = os.path.normpath(candidate)
        if candidate not in seen:
            seen.add(candidate)
            attempted.append(candidate)

    add_candidate(image_path)

    if log_dir and not os.path.isabs(image_path):
        add_candidate(os.path.join(log_dir, image_path))

    subdirs = search_subdirs if search_subdirs is not None else DEFAULT_IMAGE_SEARCH_SUBDIRS
    if log_dir:
        for subdir in subdirs:
            if not subdir:
                continue
            candidate_dir = subdir if os.path.isabs(subdir) else os.path.join(log_dir, subdir)
            add_candidate(os.path.join(candidate_dir, os.path.basename(image_path)))

    if extra_search_paths:
        for root in extra_search_paths:
            if not root:
                continue
            candidate_dir = root if os.path.isabs(root) else os.path.join(log_dir or '', root)
            add_candidate(os.path.join(candidate_dir, os.path.basename(image_path)))

    for candidate in attempted:
        if os.path.exists(candidate):
            return os.path.abspath(candidate), attempted

    return None, attempted

