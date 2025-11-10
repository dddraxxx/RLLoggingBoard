#!/usr/bin/env python3
"""Utilities to export rollout samples into structured JSON conversations."""

from __future__ import annotations

import json
import math
import re
import random
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pillow is optional but preferred for resizing
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

CONVERSATION_MAX_PIXELS = 600 * 600
IMAGE_PAD_TOKEN = "<|image_pad|>"
IMAGE_PAD_PATTERN = re.compile(
    rf"{re.escape(IMAGE_PAD_TOKEN)}(?:\s*\*\s*\d+)?"
)


@dataclass
class SampleRecord:
    step: int
    data_source: str
    prompt: str
    response: str
    images: List[Dict[str, Any]]
    tool_history: Optional[List[str]]
    ability: Optional[str]
    acc_reward: Optional[float]
    ground_truth: Optional[str]


@dataclass
class ImageAsset:
    label: str
    display_path: str
    original_local_path: Optional[str]
    source_path: Optional[str]
    uses_preview: bool
    kind: str


def _iter_step_records(step_data: Dict[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    if not step_data:
        return

    lengths = [
        len(values)
        for values in step_data.values()
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, dict))
    ]
    if not lengths:
        return
    total = max(lengths)

    for idx in range(total):
        record: Dict[str, Any] = {}
        for key, values in step_data.items():
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                record[key] = values[idx] if idx < len(values) else None
            else:
                record[key] = values
        yield record


def _collect_images(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []
    seen_assets: set[Tuple[str, str, str]] = set()

    def add_image(
        path: Optional[str],
        label: str,
        kind: str,
        source: Optional[str] = None,
    ) -> None:
        if not isinstance(path, str) or not path:
            return
        key = (path, kind, label)
        if key in seen_assets:
            return
        seen_assets.add(key)
        images.append(
            {
                "label": label,
                "path": path,
                "kind": kind,
                "source": source or path,
            }
        )

    primary = entry.get("image_path") or entry.get("processed_image_path")
    add_image(primary, "original", "original")

    processed = entry.get("processed_images")
    if isinstance(processed, Sequence):
        input_items: List[Tuple[Optional[str], str, str, Optional[str]]] = []
        other_items: List[Tuple[Optional[str], str, str, Optional[str]]] = []
        for item in processed:
            if not isinstance(item, dict):
                continue
            path = item.get("path") or item.get("image_path")
            label = item.get("tool") or item.get("label") or "processed"
            label_str = str(label) if label is not None else "processed"
            kind = "processed"
            if label_str == "input":
                kind = "input"
            elif label_str == "origin":
                kind = "original"
            target = input_items if kind == "input" else other_items
            target.append((path, label_str, kind, item.get("source")))

        for path, label_str, kind, source in input_items + other_items:
            add_image(path, label_str, kind, source)

    history_images = entry.get("tool_execution_history")
    if isinstance(history_images, Sequence):
        for idx, item in enumerate(history_images, start=1):
            if not isinstance(item, dict):
                continue
            path = item.get("image_path")
            if not isinstance(path, str):
                continue
            reason = item.get("reason")
            label = f"tool_call_{idx}"
            if isinstance(reason, str) and reason.strip():
                label = f"{label} ({reason.strip()})"
            add_image(path, label, "tool_output")

    return images


def _count_image_pad_markers(text: str) -> int:
    if not text:
        return 0

    return sum(1 for _ in IMAGE_PAD_PATTERN.finditer(text))


def _format_tool_history(entry: Dict[str, Any]) -> Optional[List[str]]:
    history = entry.get("tool_execution_history")
    if not history:
        return None

    lines: List[str] = []
    for idx, item in enumerate(history, start=1):
        if not isinstance(item, dict):
            continue
        name = item.get("name") or "image_zoom_in_tool"
        arguments = item.get("arguments")
        reason = item.get("reason")
        summary_parts = [f"{name} #{idx}"]
        if isinstance(arguments, dict):
            arg_summary = ", ".join(f"{k}={v}" for k, v in arguments.items())
            if arg_summary:
                summary_parts.append(f"({arg_summary})")
        if isinstance(reason, str) and reason.strip():
            summary_parts.append(f"- {reason.strip()}")
        lines.append(" ".join(summary_parts))

    return lines or None


def _compress_image_pad_sequences(text: str) -> str:
    if IMAGE_PAD_TOKEN not in text:
        return text

    pattern = re.compile(rf"({re.escape(IMAGE_PAD_TOKEN)})(?:\s*\1)+")

    def repl(match: re.Match[str]) -> str:
        segment = match.group(0)
        count = segment.count(IMAGE_PAD_TOKEN)
        return f"{IMAGE_PAD_TOKEN}*{count}"

    return pattern.sub(repl, text)


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    cleaned = textwrap.dedent(text).strip()
    return _compress_image_pad_sequences(cleaned)


def _slugify(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in ("-", "_"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug or "dataset"


def _extract_im_blocks(text: str) -> List[Tuple[str, str, int, int]]:
    blocks: List[Tuple[str, str, int, int]] = []
    if not text:
        return blocks

    search_pos = 0
    start_token = "<|im_start|>"
    end_token = "<|im_end|>"
    start_len = len(start_token)
    end_len = len(end_token)

    while True:
        start_idx = text.find(start_token, search_pos)
        if start_idx == -1:
            break
        role_start = start_idx + start_len
        newline_idx = text.find("\n", role_start)
        if newline_idx == -1:
            break
        role = text[role_start:newline_idx].strip()
        end_idx = text.find(end_token, newline_idx)
        if end_idx == -1:
            break
        content = text[newline_idx + 1 : end_idx]
        block_end = end_idx + end_len
        blocks.append((role, content, start_idx, block_end))
        search_pos = block_end

    return blocks


def _parse_prompt_messages(prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for role, content, _, _ in _extract_im_blocks(prompt):
        cleaned = _clean_text(content)
        if role == "assistant" and not cleaned:
            continue
        if not cleaned:
            continue
        messages.append({"role": role, "content": cleaned})
    return messages


def _parse_response_messages(response: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if not response:
        return messages

    blocks = _extract_im_blocks(response)
    if blocks:
        prefix = response[: blocks[0][2]].strip()
        if prefix:
            messages.append({"role": "assistant", "content": _clean_text(prefix)})

        for role, content, _, block_end in blocks:
            cleaned = _clean_text(content)
            if role == "assistant" and not cleaned:
                continue
            if not cleaned:
                continue
            messages.append({"role": role, "content": cleaned})
        suffix = response[blocks[-1][3] :].strip()
        if suffix:
            messages.append({"role": "assistant", "content": _clean_text(suffix)})
    else:
        cleaned = _clean_text(response)
        if cleaned:
            messages.append({"role": "assistant", "content": cleaned})

    return messages


def _unique_destination(dest_dir: Path, base_name: str) -> Path:
    candidate = dest_dir / base_name
    if not candidate.exists():
        return candidate

    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    counter = 1
    while True:
        candidate = dest_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _maybe_create_preview(image_path: Path, dest_dir: Path, max_pixels: int = CONVERSATION_MAX_PIXELS) -> Optional[Path]:
    if Image is None:
        return None

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width * height <= max_pixels:
                return None

            scale = math.sqrt(max_pixels / float(width * height))
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            preview_name = f"{image_path.stem}_preview{image_path.suffix}"
            preview_path = _unique_destination(dest_dir, preview_name)

            img_copy = img.copy()
            resample_attr = getattr(Image, "Resampling", None)
            if resample_attr:
                resample = getattr(resample_attr, "LANCZOS", getattr(resample_attr, "BICUBIC", None))
            else:
                resample = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", getattr(Image, "ANTIALIAS", None)))
            if resample is None:
                resample = 1
            img_copy.thumbnail((new_width, new_height), resample=resample)
            img_copy.save(preview_path)
            return preview_path
    except Exception:
        return None


def _prepare_image_assets(
    images: List[Dict[str, Any]],
    assets_dir: Path,
) -> Tuple[Optional[ImageAsset], List[ImageAsset]]:
    assets_dir.mkdir(parents=True, exist_ok=True)

    original_asset: Optional[ImageAsset] = None
    conversation_assets: List[ImageAsset] = []

    for image in images:
        raw_path = image.get("path")
        if not isinstance(raw_path, str):
            continue

        label = str(image.get("label") or "image")
        kind = str(image.get("kind") or "processed")

        src_path = Path(raw_path)
        display_path = raw_path
        original_local: Optional[str] = None
        uses_preview = False

        if src_path.exists():
            try:
                dest_path = _unique_destination(assets_dir, src_path.name)
                shutil.copy2(src_path, dest_path)
                rel_path = Path("images") / dest_path.name
                original_local = rel_path.as_posix()
                display_path = original_local

                if kind == "original":
                    preview = _maybe_create_preview(dest_path, assets_dir)
                    if preview is not None:
                        display_path = (Path("images") / preview.name).as_posix()
                        uses_preview = True
            except Exception:
                display_path = raw_path
        else:
            display_path = raw_path

        asset = ImageAsset(
            label=label,
            display_path=display_path,
            original_local_path=original_local,
            source_path=raw_path,
            uses_preview=uses_preview,
            kind=kind,
        )

        if kind == "original":
            if original_asset is None:
                original_asset = asset
            continue

        conversation_assets.append(asset)

    return original_asset, conversation_assets


def _image_asset_to_entry(asset: ImageAsset) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "type": "image",
        "label": asset.label,
        "kind": asset.kind,
        "path": asset.display_path,
        "uses_preview": asset.uses_preview,
    }
    if asset.original_local_path:
        entry["full_resolution_path"] = asset.original_local_path
    if asset.source_path:
        entry["source_path"] = asset.source_path
    return entry


def _md_path(path: Optional[str]) -> Optional[str]:
    if not isinstance(path, str) or not path:
        return path
    if path.startswith(("http://", "https://", "/", "../")):
        return path
    return (Path("..") / Path(path)).as_posix()


def export_step_samples(
    step_data: Dict[str, Sequence[Any]],
    step: int,
    cases_per_dataset: int,
    export_root: Path,
    log_file: str,
    seed: Optional[int] = None,
) -> List[Path]:
    if cases_per_dataset <= 0 or not step_data:
        return []

    exports: List[Path] = []

    dataset_samples: Dict[str, List[SampleRecord]] = {}
    for entry in _iter_step_records(step_data):
        data_source = entry.get("data_source") or entry.get("datasource")
        if not isinstance(data_source, str):
            continue
        prompt = entry.get("prompt") or ""
        response = entry.get("response") or ""
        images = _collect_images(entry)
        tool_hist = _format_tool_history(entry)
        ability = entry.get("ability")
        acc_reward = entry.get("acc_reward")
        try:
            acc_reward_val = float(acc_reward) if acc_reward is not None else None
        except (TypeError, ValueError):
            acc_reward_val = None
        ground_truth = entry.get("ground_truth")
        if not isinstance(ground_truth, str):
            ground_truth_str: Optional[str] = None
        else:
            ground_truth_str = ground_truth
        dataset_samples.setdefault(data_source, []).append(
            SampleRecord(
                step=step,
                data_source=data_source,
                prompt=prompt,
                response=response,
                images=images,
                tool_history=tool_hist,
                ability=ability if isinstance(ability, str) else None,
                acc_reward=acc_reward_val,
                ground_truth=ground_truth_str,
            )
        )

    if not dataset_samples:
        return []

    step_dir = export_root / "rl_sample_vis" / str(step)
    step_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = step_dir / "images"

    for dataset, samples in dataset_samples.items():
        rng_seed = f"{seed}:{step}:{dataset}" if seed is not None else f"{step}:{dataset}"
        rng = random.Random(rng_seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        subset = shuffled[: cases_per_dataset]
        if not subset:
            continue

        slug = _slugify(dataset)
        file_path = step_dir / f"{slug}.json"
        md_dir = step_dir / slug
        md_dir.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "dataset": dataset,
            "step": step,
            "log_file": log_file,
            "samples": [],
        }

        for idx, sample in enumerate(subset, start=1):
            prompt_messages = _parse_prompt_messages(sample.prompt)
            response_messages = _parse_response_messages(sample.response)
            messages = prompt_messages + response_messages

            original_asset: Optional[ImageAsset] = None
            conversation_assets: List[ImageAsset] = []
            if sample.images:
                original_asset, conversation_assets = _prepare_image_assets(sample.images, assets_dir)

            conversation: List[Dict[str, Any]] = []
            image_queue: List[ImageAsset] = list(conversation_assets)

            for message in messages:
                role = message["role"]
                content = _clean_text(message["content"])
                pad_count = _count_image_pad_markers(content)
                if not content and pad_count <= 0:
                    continue
                if content:
                    conversation.append(
                        {
                            "type": "text",
                            "role": role,
                            "content": content,
                        }
                    )
                if pad_count > 0 and image_queue:
                    for _ in range(min(pad_count, len(image_queue))):
                        conversation.append(_image_asset_to_entry(image_queue.pop(0)))

            sample_entry: Dict[str, Any] = {
                "index": idx,
                "step": sample.step,
                "data_source": sample.data_source,
                "conversation": conversation,
            }

            if original_asset is not None:
                sample_entry["original_image"] = _image_asset_to_entry(original_asset)

            if sample.ability:
                sample_entry["ability"] = sample.ability

            if sample.tool_history:
                sample_entry["tool_history"] = sample.tool_history
            if sample.acc_reward is not None:
                sample_entry["acc_reward"] = sample.acc_reward
            if sample.ground_truth is not None:
                sample_entry["ground_truth"] = sample.ground_truth

            payload["samples"].append(sample_entry)

            # Markdown export for this sample
            md_lines: List[str] = []
            md_lines.append(f"# {dataset} – Step {step} – Sample {idx}")
            metadata_bits: List[str] = []
            if sample_entry.get("ability"):
                metadata_bits.append(f"Ability: `{sample_entry['ability']}`")
            if sample_entry.get("acc_reward") is not None:
                metadata_bits.append(f"acc_reward: {sample_entry['acc_reward']}")
            if sample_entry.get("ground_truth"):
                metadata_bits.append(f"GT: `{sample_entry['ground_truth']}`")
            if metadata_bits:
                md_lines.append("_" + " · ".join(metadata_bits) + "_")
            md_lines.append("")

            original_image = sample_entry.get("original_image")
            if isinstance(original_image, dict):
                orig_path = original_image.get("full_resolution_path") or original_image.get("path")
                if orig_path:
                    orig_link = _md_path(orig_path) or orig_path
                    md_lines.append(f"**Original image:** [{original_image.get('label', 'original')}]({orig_link})")
                    md_lines.append("")

            md_lines.append("## Conversation")
            md_lines.append("")

            for entry in conversation:
                if entry.get("type") == "image":
                    label = entry.get("label", "image")
                    path = _md_path(entry.get("path"))
                    full_res = _md_path(entry.get("full_resolution_path"))
                    uses_preview = bool(entry.get("uses_preview"))
                    if path:
                        md_lines.append(f"![{label}]({path})")
                        if uses_preview and full_res and full_res != path:
                            md_lines.append(f"[Full resolution]({full_res})")
                    md_lines.append("")
                else:
                    role = entry.get("role", "assistant").title()
                    content = entry.get("content", "")
                    md_lines.append(f"**{role}:**")
                    md_lines.append("```text")
                    md_lines.append(content)
                    md_lines.append("```")
                    md_lines.append("")

            if sample_entry.get("tool_history"):
                md_lines.append("## Tool History")
                md_lines.append("")
                for line in sample_entry["tool_history"]:
                    md_lines.append(f"- {line}")
                md_lines.append("")

            md_path = md_dir / f"sample_{idx}.md"
            with md_path.open("w", encoding="utf-8") as md_handle:
                md_handle.write("\n".join(md_lines).strip() + "\n")

        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

        exports.append(file_path)

    return exports
