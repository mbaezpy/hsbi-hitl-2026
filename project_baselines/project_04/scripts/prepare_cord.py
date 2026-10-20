#!/usr/bin/env python3
"""Prepare a reproducible CORD v2 subset for Project 04.

By default the script uses the official Hugging Face dataset-viewer API. This
downloads only selected receipt images, not the multi-gigabyte Parquet files.
The viewer currently exposes 26 train preview rows and all 100 validation/test
rows. For larger train samples, pass a local DatasetDict saved with
``save_to_disk`` and use ``--source-mode local``.

CORD already supplies OCR, boxes, and semantic annotations. No OCR is run here.
The introductory table maps lines to four receipt-summary roles plus OTHER;
the complete source annotations remain in ``documents.jsonl``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_ID = "naver-clova-ix/cord-v2"
SOURCE_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
SPLITS = ("train", "validation", "test")
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_SOURCE = REPOSITORY_DIR / "data" / "cord_v2"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "prepared" / "cord_v2"

SOURCE_TO_COURSE = {
    "sub_total.subtotal_price": "subtotal",
    "sub_total.service_price": "service_charge",
    "sub_total.tax_price": "tax",
    "total.total_price": "total_amount",
}
COURSE_ROLES = (*SOURCE_TO_COURSE.values(), "OTHER")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-mode",
        choices=("auto", "viewer", "local"),
        default="auto",
        help="Use local saved data when present, otherwise the lightweight viewer API.",
    )
    parser.add_argument(
        "--local-source",
        type=Path,
        default=DEFAULT_LOCAL_SOURCE,
        help=f"DatasetDict saved with save_to_disk (default: {DEFAULT_LOCAL_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Prepared output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--train-size", type=int, default=26)
    parser.add_argument("--validation-size", type=int, default=35)
    parser.add_argument("--test-size", type=int, default=34)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--refresh-source",
        action="store_true",
        help="Re-read the source even when this exact prepared sample already exists.",
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_rank(seed: int, split: str, row_index: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{split}\0{row_index}".encode()).digest()


def request_bytes(url: str, timeout: float, attempts: int = 5) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "HSBI-HITL-course-data/1.0"})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                retry_after = getattr(exc, "headers", {}).get("Retry-After") if getattr(exc, "headers", None) else None
                time.sleep(float(retry_after) if retry_after else 2**attempt)
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from last_error


def request_json(url: str, timeout: float) -> dict[str, Any]:
    return json.loads(request_bytes(url, timeout).decode("utf-8"))


def api_url(endpoint: str, **parameters: Any) -> str:
    return "https://datasets-server.huggingface.co/" + endpoint + "?" + urllib.parse.urlencode(parameters)


def verify_revision(revision: str, timeout: float) -> None:
    revision_url = f"https://huggingface.co/api/datasets/{DATASET_ID}/revision/{revision}"
    revision_info = request_json(revision_url, timeout)
    if revision_info.get("sha") != revision:
        raise RuntimeError(f"Could not verify requested revision {revision!r}.")

    # The viewer API serves the current repository revision and has no revision
    # query parameter. Refuse viewer mode if main moves, rather than silently
    # sampling different data under the old revision label.
    main_info = request_json(f"https://huggingface.co/api/datasets/{DATASET_ID}", timeout)
    if main_info.get("sha") != revision:
        raise RuntimeError(
            "The viewer currently serves a newer repository revision. Use a pinned "
            "local DatasetDict or deliberately update --revision after auditing it."
        )


def viewer_rows(split: str, timeout: float) -> list[dict[str, Any]]:
    if split == "train":
        payload = request_json(
            api_url("first-rows", dataset=DATASET_ID, config="default", split=split),
            timeout,
        )
        rows = payload.get("rows", [])
    else:
        # Four modest requests are more reliable than one large JSON response,
        # while still exposing the complete 100-row validation/test split.
        rows = []
        for offset in range(0, 100, 25):
            payload = request_json(
                api_url(
                    "rows",
                    dataset=DATASET_ID,
                    config="default",
                    split=split,
                    offset=offset,
                    length=25,
                ),
                timeout,
            )
            rows.extend(payload.get("rows", []))
    if not rows:
        raise RuntimeError(f"Viewer returned no {split} rows: {payload.get('error', payload)}")
    return rows


def local_rows(local_source: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        from datasets import DatasetDict, Image, load_from_disk
    except ImportError as exc:
        raise SystemExit(
            "Local mode requires Hugging Face datasets: python -m pip install datasets"
        ) from exc

    dataset = load_from_disk(str(local_source))
    if not isinstance(dataset, DatasetDict) or any(split not in dataset for split in SPLITS):
        raise RuntimeError("Local source must be a DatasetDict with train/validation/test.")

    result: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        if not {"image", "ground_truth"}.issubset(dataset[split].column_names):
            raise RuntimeError(f"Local split {split!r} lacks image or ground_truth.")
        undecoded = dataset[split].cast_column("image", Image(decode=False))
        # Keep only lightweight row references. Selected rows are decoded later;
        # loading every embedded image here would consume gigabytes of memory.
        result[split] = [
            {"row_idx": index, "_dataset": undecoded} for index in range(len(undecoded))
        ]
    return result


def quad_bbox(quad: dict[str, Any]) -> list[int]:
    xs = [quad[f"x{i}"] for i in range(1, 5)]
    ys = [quad[f"y{i}"] for i in range(1, 5)]
    return [min(xs), min(ys), max(xs), max(ys)]


def union_bbox(boxes: list[list[int]]) -> list[int]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def simplify_word(word: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": word.get("text", ""),
        "bbox": quad_bbox(word["quad"]),
        "quad": word["quad"],
        "row_id": word.get("row_id"),
        "is_key": word.get("is_key"),
    }


def prepare_lines(ground_truth: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep all released valid lines and textual repeating-symbol distractors."""
    lines: list[dict[str, Any]] = []
    for source_index, source_line in enumerate(ground_truth.get("valid_line", [])):
        words = [simplify_word(word) for word in source_line.get("words", [])]
        if not words:
            continue
        source_category = source_line.get("category", "")
        lines.append(
            {
                "line_id": f"valid_{source_index:04d}",
                "text": " ".join(word["text"] for word in words).strip(),
                "bbox": union_bbox([word["bbox"] for word in words]),
                "words": words,
                "source_category": source_category,
                "course_role": SOURCE_TO_COURSE.get(source_category, "OTHER"),
                "group_id": source_line.get("group_id"),
                "sub_group_id": source_line.get("sub_group_id"),
                "annotation_source": "valid_line",
            }
        )

    # repeating_symbol is a list of word groups. These are genuine released
    # distractors and are retained as OTHER; dontcare has boxes but no OCR text.
    for source_index, source_group in enumerate(ground_truth.get("repeating_symbol", [])):
        words = [simplify_word(word) for word in source_group if "quad" in word]
        if not words:
            continue
        lines.append(
            {
                "line_id": f"repeating_{source_index:04d}",
                "text": " ".join(word["text"] for word in words).strip(),
                "bbox": union_bbox([word["bbox"] for word in words]),
                "words": words,
                "source_category": "repeating_symbol",
                "course_role": "OTHER",
                "group_id": None,
                "sub_group_id": None,
                "annotation_source": "repeating_symbol",
            }
        )
    return lines


def simplify_document_labels(documents: list[dict[str, Any]]) -> Counter[str]:
    """Apply the five-class mapping while retaining every source category."""
    counts: Counter[str] = Counter()
    for document in documents:
        for line in document["lines"]:
            label = SOURCE_TO_COURSE.get(line.get("source_category", ""), "OTHER")
            line["course_role"] = label
            counts[label] += 1
    return counts


def write_documents(documents: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_flat_table(documents: list[dict[str, Any]], path: Path) -> int:
    """Write the notebook-ready table without layout or gold metadata features."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    row_count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("document_id", "text", "label", "split")
        )
        writer.writeheader()
        for document in documents:
            for line in document["lines"]:
                writer.writerow(
                    {
                        "document_id": document["document_id"],
                        "text": line.get("text", ""),
                        "label": line["course_role"],
                        "split": document["source_split"],
                    }
                )
                row_count += 1
    temporary.replace(path)
    return row_count


def image_bytes_from_row(row: dict[str, Any], timeout: float) -> bytes:
    image = row["image"]
    if isinstance(image, dict) and image.get("src"):
        return request_bytes(image["src"], timeout)
    if isinstance(image, dict) and image.get("bytes") is not None:
        return image["bytes"]
    if isinstance(image, dict) and image.get("path"):
        return Path(image["path"]).read_bytes()
    raise RuntimeError("Unsupported image representation in source row.")


def main() -> None:
    args = parse_args()
    requested = {
        "train": args.train_size,
        "validation": args.validation_size,
        "test": args.test_size,
    }
    if any(size <= 0 for size in requested.values()):
        raise SystemExit("All split sizes must be positive.")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be positive.")

    output = args.output.expanduser().resolve()
    local_source = args.local_source.expanduser().resolve()
    if output == local_source or local_source in output.parents or output in local_source.parents:
        raise SystemExit("Source and output directories must be separate.")

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    jsonl_path = output / "documents.jsonl"
    flat_path = output / "ocr_lines.csv"

    # Rebuild the lightweight table from an identical prepared document sample.
    # This also upgrades samples made by the earlier seven-role script without
    # touching their images or contacting Hugging Face again.
    if summary_path.exists() and jsonl_path.exists() and not args.refresh_source:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        previous = existing.get("configuration", {})
        same_sample = (
            previous.get("dataset") == DATASET_ID
            and previous.get("revision") == args.revision
            and previous.get("requested_split_sizes") == requested
            and previous.get("seed") == args.seed
        )
        if not same_sample:
            raise SystemExit(
                "Output contains another sample configuration. Choose a different "
                "--output directory, or pass the sizes and seed recorded in summary.json."
            )
        documents = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        actual_splits = Counter(doc["source_split"] for doc in documents)
        document_ids = [doc["document_id"] for doc in documents]
        if dict(actual_splits) != requested or len(document_ids) != len(set(document_ids)):
            raise SystemExit("Existing documents do not match the recorded sample.")
        if any(doc.get("source_revision") != args.revision for doc in documents):
            raise SystemExit("Existing documents do not match the requested source revision.")
        role_counts = simplify_document_labels(documents)
        write_documents(documents, jsonl_path)
        line_count = write_flat_table(documents, flat_path)
        previous["course_schema"] = list(COURSE_ROLES)
        generated = existing.setdefault("generated", {})
        generated["ocr_lines"] = line_count
        generated["course_role_distribution"] = dict(sorted(role_counts.items()))
        generated["documents_sha256"] = sha256_bytes(jsonl_path.read_bytes())
        generated["ocr_lines_sha256"] = sha256_bytes(flat_path.read_bytes())
        write_target = summary_path.with_suffix(".json.tmp")
        write_target.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_target.replace(summary_path)
        print(f"Reused {len(documents)} prepared documents; wrote {flat_path}")
        print(json.dumps(generated, indent=2))
        return

    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise SystemExit("Install Pillow first: python -m pip install pillow") from exc

    source_mode = args.source_mode
    if source_mode == "auto":
        source_mode = "local" if local_source.exists() else "viewer"
    if source_mode == "local" and not local_source.is_dir():
        raise SystemExit(f"Local CORD DatasetDict not found: {local_source}")

    started = time.perf_counter()
    if source_mode == "viewer":
        verify_revision(args.revision, args.request_timeout)
        source_rows = {split: viewer_rows(split, args.request_timeout) for split in SPLITS}
    else:
        source_rows = local_rows(local_source)

    available = {split: len(source_rows[split]) for split in SPLITS}
    for split in SPLITS:
        if requested[split] > available[split]:
            detail = ""
            if source_mode == "viewer" and split == "train":
                detail = (
                    " The official viewer exposes only 26 train preview rows; use "
                    "--source-mode local with an official DatasetDict for a larger train sample."
                )
            raise SystemExit(
                f"Requested {requested[split]} {split} rows, but {available[split]} are "
                f"available in {source_mode} mode.{detail}"
            )

    configuration = {
        "dataset": DATASET_ID,
        "revision": args.revision,
        "source_mode": source_mode,
        "local_source": str(local_source) if source_mode == "local" else None,
        "requested_split_sizes": requested,
        "seed": args.seed,
        "course_schema": list(COURSE_ROLES),
    }
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        previous_configuration = existing.get("configuration", {})
        comparable_previous = {
            key: value
            for key, value in previous_configuration.items()
            if key != "course_schema"
        }
        comparable_current = {
            key: value for key, value in configuration.items() if key != "course_schema"
        }
        if comparable_previous != comparable_current:
            raise SystemExit(
                "Output contains another configuration. Choose a different --output directory."
            )

    selected: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        selected[split] = sorted(
            source_rows[split],
            key=lambda wrapped: stable_rank(args.seed, split, wrapped["row_idx"]),
        )[: requested[split]]

    documents: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    seen_images: dict[str, tuple[str, int]] = {}
    role_counts: Counter[str] = Counter()
    source_category_counts: Counter[str] = Counter()

    expected = sum(requested.values())
    for split in SPLITS:
        for wrapped in selected[split]:
            row_index = int(wrapped["row_idx"])
            row = (
                wrapped["row"]
                if "row" in wrapped
                else wrapped["_dataset"][row_index]
            )
            raw_ground_truth = row["ground_truth"]
            ground_truth = (
                json.loads(raw_ground_truth)
                if isinstance(raw_ground_truth, str)
                else raw_ground_truth
            )
            raw_image = image_bytes_from_row(row, args.request_timeout)
            content_hash = sha256_bytes(raw_image)
            if content_hash in seen_images:
                previous = seen_images[content_hash]
                raise RuntimeError(
                    f"Exact image duplicate crosses selected rows: {previous} and "
                    f"{(split, row_index)}. Choose another seed/sample."
                )
            seen_images[content_hash] = (split, row_index)

            with PILImage.open(io.BytesIO(raw_image)) as image:
                image.verify()
            with PILImage.open(io.BytesIO(raw_image)) as image:
                width, height = image.size
                extension = ".png" if image.format == "PNG" else ".jpg"

            document_id = f"cord_{split}_{row_index:04d}"
            relative_image = Path("images") / split / f"{document_id}{extension}"
            output_image = output / relative_image
            output_image.parent.mkdir(parents=True, exist_ok=True)
            if not output_image.exists() or sha256_bytes(output_image.read_bytes()) != content_hash:
                output_image.write_bytes(raw_image)

            lines = prepare_lines(ground_truth)
            for line in lines:
                role_counts[line["course_role"]] += 1
                source_category_counts[line["source_category"]] += 1

            meta = ground_truth.get("meta", {})
            document = {
                "document_id": document_id,
                "source_dataset": DATASET_ID,
                "source_revision": args.revision,
                "source_split": split,
                "source_row_index": row_index,
                "source_image_id": meta.get("image_id"),
                "image": relative_image.as_posix(),
                "image_sha256": content_hash,
                "image_width": width,
                "image_height": height,
                "ocr_text": "\n".join(line["text"] for line in lines if line["text"]),
                "lines": lines,
                "gt_parse": ground_truth.get("gt_parse", {}),
                "source_meta": meta,
                "roi": ground_truth.get("roi", {}),
                "dontcare": ground_truth.get("dontcare", []),
            }
            documents.append(document)
            manifest_rows.append(
                {
                    "document_id": document_id,
                    "source_split": split,
                    "source_row_index": row_index,
                    "source_image_id": meta.get("image_id"),
                    "image": relative_image.as_posix(),
                    "image_sha256": content_hash,
                    "ocr_lines": len(lines),
                    "ocr_characters": len(document["ocr_text"]),
                }
            )
            print(
                f"[{len(documents):>{len(str(expected))}}/{expected}] "
                f"{split:<10} row {row_index:<3} lines={len(lines)}",
                flush=True,
            )

    documents.sort(key=lambda doc: (SPLITS.index(doc["source_split"]), doc["source_row_index"]))
    manifest_rows.sort(
        key=lambda row: (SPLITS.index(row["source_split"]), row["source_row_index"])
    )

    write_documents(documents, jsonl_path)
    flat_line_count = write_flat_table(documents, flat_path)

    manifest_path = output / "manifest.csv"
    temporary_manifest = output / "manifest.csv.tmp"
    with temporary_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    temporary_manifest.replace(manifest_path)

    summary = {
        "configuration": configuration,
        "source_access": {
            "available_rows_in_mode": available,
            "revision_verification": (
                "Pinned repository SHA verified before viewer access."
                if source_mode == "viewer"
                else "Local DatasetDict; its originating Hub revision cannot be inferred from save_to_disk metadata."
            ),
            "viewer_limitation": (
                "At the pinned revision, the dataset viewer exposes 26 train preview "
                "rows because train Parquet shards exceed its scan limit. Validation "
                "and test expose all 100 rows. Use local mode for more train rows."
                if source_mode == "viewer"
                else None
            ),
            "candidate_limitation": (
                "CORD v2 supplies curated valid_line OCR rather than every visible "
                "receipt line. Excluded semantic categories and repeating symbols are "
                "retained as OTHER, but ordinary header/store OCR is unavailable."
            ),
        },
        "generated": {
            "documents": len(documents),
            "split_distribution": dict(Counter(doc["source_split"] for doc in documents)),
            "ocr_lines": flat_line_count,
            "ocr_characters": sum(len(doc["ocr_text"]) for doc in documents),
            "course_role_distribution": dict(sorted(role_counts.items())),
            "source_category_distribution": dict(sorted(source_category_counts.items())),
            "unique_image_hashes": len(seen_images),
            "processing_seconds": round(time.perf_counter() - started, 3),
            "documents_sha256": sha256_bytes(jsonl_path.read_bytes()),
            "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
            "ocr_lines_sha256": sha256_bytes(flat_path.read_bytes()),
        },
        "license": {
            "name": "Creative Commons Attribution 4.0 International",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": (
                "CORD v2, NAVER CLOVA AI Research; Park et al., CORD: A "
                "Consolidated Receipt Dataset for Post-OCR Parsing (2019)."
            ),
        },
    }
    write_target = summary_path.with_suffix(".json.tmp")
    write_target.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_target.replace(summary_path)
    print(f"\nPrepared {len(documents)} documents in {output}")
    print(json.dumps(summary["generated"], indent=2))


if __name__ == "__main__":
    main()
