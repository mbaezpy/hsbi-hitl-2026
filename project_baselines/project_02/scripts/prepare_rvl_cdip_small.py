#!/usr/bin/env python3
"""Create a reproducible, OCR-enriched RVL-CDIP course sample.

The script reads an existing Hugging Face DatasetDict from local disk. It does
not download data. Selected images are copied to a separate output directory,
so the source dataset remains unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Fail instead of contacting the Hub if the local source path is incorrect.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

try:
    import pytesseract
    import rvl_cdip_ocr as adaptive_ocr
    from datasets import DatasetDict, load_from_disk
except ImportError as exc:
    raise SystemExit(
        "Install the Python dependencies first: "
        "python -m pip install datasets pillow pytesseract"
    ) from exc

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPOSITORY_DIR / "data" / "rvl_cdip_small"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "prepared" / "rvl_cdip_small"

COURSE_LABELS = (
    "advertisement",
    "email",
    "form",
    "invoice",
    "letter",
    "news article",
    "presentation",
    "resume",
)
SPLITS = ("train", "validation", "test")
CSV_COLUMNS = (
    "sample_id",
    "source_dataset",
    "source_split",
    "source_row_index",
    "source_row_id",
    "source_label_id",
    "source_label",
    "label",
    "image_path",
    "image_sha256",
    "width",
    "height",
    "ocr_text",
    "ocr_chars",
    "ocr_status",
    "ocr_error",
)
OCR_PIPELINE_ID = adaptive_ocr.OCR_PIPELINE_ID
OCR_POLICY = adaptive_ocr.DEFAULT_POLICY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Local Hugging Face dataset (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Prepared data directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=20,
        help="Total selected examples per class across the original splits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tesseract-lang", default="eng")
    parser.add_argument("--tesseract-config", default="--psm 3")
    parser.add_argument("--ocr-timeout", type=float, default=60.0)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Shared OCR cache (default: OUTPUT/.ocr_cache).",
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_hash(image) -> str:
    """Hash decoded pixels, mode, and dimensions to find exact duplicates."""
    image = image.convert("L")
    header = f"L:{image.width}x{image.height}:".encode("ascii")
    return sha256_bytes(header + image.tobytes())


def deterministic_rank(seed: int, split: str, label: str, index: int) -> bytes:
    """Stable pseudo-random order independent of Python's hash seed."""
    value = f"{seed}\0{split}\0{label}\0{index}".encode()
    return hashlib.sha256(value).digest()


def allocate_across_splits(total: int, available: dict[str, int]) -> dict[str, int]:
    """Allocate a class total using source proportions and largest remainder."""
    available_total = sum(available.values())
    if total > available_total:
        raise ValueError(f"requested {total}, but only {available_total} are available")

    ideal = {split: total * available[split] / available_total for split in SPLITS}
    allocation = {split: int(ideal[split]) for split in SPLITS}
    remaining = total - sum(allocation.values())
    priority = sorted(
        SPLITS,
        key=lambda split: (-(ideal[split] - allocation[split]), SPLITS.index(split)),
    )
    for split in priority[:remaining]:
        allocation[split] += 1
    return allocation


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_cached_ocr(path: Path, identity: dict[str, Any]) -> dict | None:
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if all(cached.get(key) == value for key, value in identity.items()):
        return cached
    return None


def cache_path(cache_dir: Path, namespace: str, identity: dict[str, Any]) -> Path:
    cache_key = sha256_bytes(json.dumps(identity, sort_keys=True).encode("utf-8"))
    path = cache_dir / OCR_PIPELINE_ID / namespace / f"{cache_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def candidate_identity(
    content_hash: str,
    tesseract_version: str,
    tesseract_lang: str,
    base_options: tuple[str, ...],
    rotation: int,
    psm: int,
    timeout: float,
) -> dict[str, Any]:
    return {
        "ocr_pipeline": OCR_PIPELINE_ID,
        "ocr_engine_version": adaptive_ocr.OCR_ENGINE_VERSION,
        "image_sha256": content_hash,
        "tesseract_version": tesseract_version,
        "tesseract_lang": tesseract_lang,
        "tesseract_base_options": list(base_options),
        "psm": psm,
        "rotation": rotation,
        "ocr_timeout": timeout,
        "preprocessing": {
            "grayscale": True,
            "upscale_min_dimension": OCR_POLICY.upscale_min_dimension,
            "upscale_max_factor": OCR_POLICY.upscale_max_factor,
            "resampling": "PIL.Image.Resampling.LANCZOS",
        },
    }


def run_adaptive_ocr(
    image,
    content_hash: str,
    tesseract_version: str,
    tesseract_lang: str,
    base_options: tuple[str, ...],
    timeout: float,
    cache_dir: Path,
) -> tuple[dict[str, Any], bool]:
    """Run configuration G followed by the tested J2 policy when triggered."""
    final_identity = {
        "ocr_pipeline": OCR_PIPELINE_ID,
        "ocr_engine_version": adaptive_ocr.OCR_ENGINE_VERSION,
        "policy_version": adaptive_ocr.POLICY_VERSION,
        "policy_parameters": OCR_POLICY.as_dict(),
        "image_sha256": content_hash,
        "tesseract_version": tesseract_version,
        "tesseract_lang": tesseract_lang,
        "tesseract_base_options": list(base_options),
        "ocr_timeout": timeout,
        "timeout_behavior": "per_tesseract_call",
    }
    final_path = cache_path(cache_dir, "final", final_identity)
    cached_final = read_cached_ocr(final_path, final_identity)
    if cached_final is not None:
        return cached_final["result"], True

    upscaled, scale_factor = adaptive_ocr.upscale_image(image, OCR_POLICY)
    candidates: dict[tuple[int, int], dict[str, Any]] = {}

    def get_candidate(rotation: int, psm: int) -> dict[str, Any]:
        key = (rotation, psm)
        if key in candidates:
            return candidates[key]
        identity = candidate_identity(
            content_hash,
            tesseract_version,
            tesseract_lang,
            base_options,
            rotation,
            psm,
            timeout,
        )
        path = cache_path(cache_dir, "candidates", identity)
        cached = read_cached_ocr(path, identity)
        if cached is not None:
            candidate = cached["result"]
        else:
            processed = adaptive_ocr.rotate_clockwise(upscaled, rotation)
            candidate = adaptive_ocr.run_ocr_candidate(
                processed,
                lang=tesseract_lang,
                config=adaptive_ocr.config_for_psm(base_options, psm),
                timeout=timeout,
            )
            candidate.update(
                {
                    "candidate_id": f"rotation_{rotation}_psm_{psm}",
                    "rotation": rotation,
                    "psm": psm,
                    "scale_factor": scale_factor,
                    "processed_width": processed.width,
                    "processed_height": processed.height,
                }
            )
            write_json(path, {**identity, "result": candidate})
        candidates[key] = candidate
        return candidate

    primary = get_candidate(0, 3)
    selected, decision = adaptive_ocr.apply_j2_policy(
        primary, get_candidate, OCR_POLICY
    )
    selected_text = str(selected.get("ocr_text") or "")
    if selected_text:
        status = "ok"
        error = ""
    elif selected.get("text_succeeded"):
        status = "empty"
        error = ""
    else:
        status = "error"
        attempted = {
            candidate_id for candidate_id in decision["attempted_candidate_ids"]
        }
        errors = [
            str(candidate.get("ocr_error") or "")
            for candidate in candidates.values()
            if candidate["candidate_id"] in attempted and candidate.get("ocr_error")
        ]
        error = " | ".join(dict.fromkeys(errors)) or "OCR produced no usable text."
    final_result = {
        "ocr_text": selected_text,
        "ocr_status": status,
        "ocr_error": error,
        "selected_candidate_id": selected["candidate_id"],
        "triggered_recovery": decision["triggered_recovery"],
    }
    write_json(final_path, {**final_identity, "result": final_result})
    return final_result, False


def main() -> None:
    args = parse_args()
    if args.examples_per_class <= 0:
        raise SystemExit("--examples-per-class must be positive")
    if args.ocr_timeout <= 0:
        raise SystemExit("--ocr-timeout must be positive")
    try:
        tesseract_base_options = adaptive_ocr.split_primary_config(
            args.tesseract_config
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if shutil.which("tesseract") is None:
        raise SystemExit(
            "Tesseract was not found. On Debian/Ubuntu, install "
            "tesseract-ocr and tesseract-ocr-eng."
        )

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Local dataset not found: {source}")
    if output == source or source in output.parents or output in source.parents:
        raise SystemExit("Source and output directories must be separate.")

    cache_dir = (args.cache_dir or output / ".ocr_cache").expanduser().resolve()
    if cache_dir == source or source in cache_dir.parents:
        raise SystemExit("The OCR cache must not be placed inside the source dataset.")
    output.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(source))
    if not isinstance(dataset, DatasetDict) or any(
        split not in dataset for split in SPLITS
    ):
        raise SystemExit(
            "Expected a local DatasetDict with train/validation/test splits."
        )

    label_feature = dataset["train"].features.get("label")
    label_names = getattr(label_feature, "names", None)
    if not label_names:
        raise SystemExit("Expected a ClassLabel feature named 'label'.")
    if any(label not in label_names for label in COURSE_LABELS):
        raise SystemExit("The local dataset does not contain all eight course labels.")
    for split in SPLITS:
        if "image" not in dataset[split].features:
            raise SystemExit(f"Split {split!r} has no image feature.")
        split_names = getattr(dataset[split].features.get("label"), "names", None)
        if split_names != label_names:
            raise SystemExit("ClassLabel mappings differ between source splits.")

    label_ids = {label: label_names.index(label) for label in COURSE_LABELS}
    indices: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    availability: dict[str, dict[str, int]] = defaultdict(dict)
    retained_ids = set(label_ids.values())
    for split in SPLITS:
        for index, label_id in enumerate(dataset[split]["label"]):
            if label_id in retained_ids:
                indices[label_names[label_id]][split].append(index)
        for label in COURSE_LABELS:
            availability[label][split] = len(indices[label][split])

    try:
        allocations = {
            label: allocate_across_splits(args.examples_per_class, availability[label])
            for label in COURSE_LABELS
        }
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    tesseract_version = str(pytesseract.get_tesseract_version()).splitlines()[0]
    configuration = {
        "source": str(source),
        "examples_per_class": args.examples_per_class,
        "seed": args.seed,
        "labels": list(COURSE_LABELS),
        "tesseract_version": tesseract_version,
        "tesseract_lang": args.tesseract_lang,
        "tesseract_config": args.tesseract_config,
        "ocr_pipeline": OCR_PIPELINE_ID,
        "ocr_engine_version": adaptive_ocr.OCR_ENGINE_VERSION,
        "ocr_policy_version": adaptive_ocr.POLICY_VERSION,
        "ocr_policy_parameters": OCR_POLICY.as_dict(),
        "tesseract_base_options": list(tesseract_base_options),
    }
    summary_path = output / "summary.json"
    existing_manifest = output / "dataset.csv"
    if existing_manifest.exists() and not summary_path.exists():
        raise SystemExit(
            "The output already contains dataset.csv but no summary.json, so its "
            "OCR pipeline cannot be verified. Choose a fresh --output directory."
        )
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("configuration") != configuration:
            existing_pipeline = existing.get("configuration", {}).get(
                "ocr_pipeline", "legacy_grayscale_psm3"
            )
            raise SystemExit(
                f"This output uses OCR pipeline {existing_pipeline!r}, not "
                f"{OCR_PIPELINE_ID!r}. Choose a fresh --output directory; "
                "existing prepared data will not be overwritten. Use --cache-dir "
                "only to share versioned OCR cache entries."
            )
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_hashes = set()
    duplicate_candidates = 0
    cache_hits = 0
    ocr_executions = 0
    expected_rows = args.examples_per_class * len(COURSE_LABELS)

    for label in COURSE_LABELS:
        for split in SPLITS:
            candidates = sorted(
                indices[label][split],
                key=lambda index: deterministic_rank(args.seed, split, label, index),
            )
            selected_in_split = 0
            for index in candidates:
                if selected_in_split == allocations[label][split]:
                    break
                source_row = dataset[split][index]
                image = source_row["image"].convert("L")
                content_hash = image_hash(image)
                if content_hash in seen_hashes:
                    duplicate_candidates += 1
                    continue

                sample_id = f"{split}-{index:06d}-{content_hash[:12]}"
                label_dir = label.replace(" ", "_")
                relative_image = Path("images") / split / label_dir / f"{sample_id}.png"
                output_image = output / relative_image
                output_image.parent.mkdir(parents=True, exist_ok=True)
                if not output_image.exists():
                    image.save(output_image, format="PNG", optimize=True)

                ocr_result, cache_hit = run_adaptive_ocr(
                    image,
                    content_hash,
                    tesseract_version,
                    args.tesseract_lang,
                    tesseract_base_options,
                    args.ocr_timeout,
                    cache_dir,
                )
                if cache_hit:
                    cache_hits += 1
                else:
                    ocr_executions += 1
                ocr_text = ocr_result["ocr_text"]
                ocr_status = ocr_result["ocr_status"]
                ocr_error = ocr_result["ocr_error"]

                seen_hashes.add(content_hash)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "source_dataset": "abarsky/rvl_cdip_small",
                        "source_split": split,
                        "source_row_index": index,
                        "source_row_id": f"{split}:{index}",
                        "source_label_id": source_row["label"],
                        "source_label": label,
                        "label": label,
                        "image_path": relative_image.as_posix(),
                        "image_sha256": content_hash,
                        "width": image.width,
                        "height": image.height,
                        "ocr_text": ocr_text,
                        "ocr_chars": len(ocr_text),
                        "ocr_status": ocr_status,
                        "ocr_error": ocr_error,
                    }
                )
                selected_in_split += 1
                print(
                    f"[{len(rows):>{len(str(expected_rows))}}/{expected_rows}] "
                    f"{split:<10} {label:<14} {ocr_status}",
                    flush=True,
                )
            if selected_in_split != allocations[label][split]:
                raise RuntimeError(
                    f"Only found {selected_in_split} unique {label!r} pages in {split!r}; "
                    f"needed {allocations[label][split]}."
                )

    rows.sort(
        key=lambda row: (
            SPLITS.index(row["source_split"]),
            row["label"],
            row["sample_id"],
        )
    )
    manifest_path = output / "dataset.csv"
    temporary_manifest = output / "dataset.csv.tmp"
    with temporary_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_manifest.replace(manifest_path)

    by_class = Counter(row["label"] for row in rows)
    by_split = Counter(row["source_split"] for row in rows)
    by_ocr_status = Counter(row["ocr_status"] for row in rows)
    summary = {
        "configuration": configuration,
        "source": {
            "split_sizes": {split: len(dataset[split]) for split in SPLITS},
            "features": list(dataset["train"].features),
            "available_course_examples": availability,
            "limitation": (
                "This mirror has no original filename or document-family ID. "
                "source_row_id is a local split:index reference; related pages "
                "cannot be grouped reliably."
            ),
        },
        "generated": {
            "rows": len(rows),
            "class_distribution": dict(sorted(by_class.items())),
            "split_distribution": {split: by_split[split] for split in SPLITS},
            "split_class_allocation": allocations,
            "unique_image_hashes": len(seen_hashes),
            "exact_duplicate_candidates_skipped": duplicate_candidates,
            "ocr_status": dict(sorted(by_ocr_status.items())),
            "ocr_executions": ocr_executions,
            "ocr_cache_hits": cache_hits,
            "ocr_characters": sum(row["ocr_chars"] for row in rows),
        },
    }
    write_json(summary_path, summary)
    print(f"\nPrepared {len(rows)} examples in {output}")
    print(json.dumps(summary["generated"], indent=2))


if __name__ == "__main__":
    main()
