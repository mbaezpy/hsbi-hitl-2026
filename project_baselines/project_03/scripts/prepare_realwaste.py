#!/usr/bin/env python3
"""Download or index RealWaste and create reproducible document-level splits.

The original images are never modified or copied. The output is a CSV manifest
whose paths point back to the source tree, plus a JSON provenance summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DATASET_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
DATASET_SHA256 = "1ede08b32358ee62065bcc1c8cb47a2ece04e8dff5b1fb53342bb8750895b2f3"
CLASSES = (
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "validation", "test")

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPOSITORY_DIR / "data" / "realwaste"
DEFAULT_ARCHIVE = REPOSITORY_DIR / "data" / "downloads" / "realwaste.zip"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "prepared" / "realwaste"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Extracted RealWaste directory (default: repository data/realwaste).",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Existing or downloaded UCI ZIP archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for manifest.csv and summary.json.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the official archive when neither source nor archive exists.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per class for a smaller stratified teaching sample.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=int,
        default=4,
        help="Maximum 64-bit difference-hash distance reported as a near duplicate.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace prepared metadata for a different configuration; source is untouched.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, *parts: object) -> bytes:
    value = "\0".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(value.encode("utf-8")).digest()


def download_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {DATASET_URL}")
    with urllib.request.urlopen(DATASET_URL, timeout=120) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out)
    actual_hash = sha256_file(temporary)
    if actual_hash != DATASET_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded archive SHA-256 is {actual_hash}, expected {DATASET_SHA256}."
        )
    temporary.replace(destination)


def extract_archive(archive: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite existing source directory: {destination}")
    if sha256_file(archive) != DATASET_SHA256:
        raise RuntimeError(
            "Archive does not match the verified UCI RealWaste artifact. "
            "Download a fresh copy or inspect the source before continuing."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="realwaste-extract-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as bundle:
            root = temporary.resolve()
            for member in bundle.infolist():
                target = (temporary / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"Unsafe path in archive: {member.filename}")
            bundle.extractall(temporary)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def find_class_directories(source: Path) -> dict[str, Path]:
    directories: dict[str, Path] = {}
    for label in CLASSES:
        candidates = [
            path
            for path in source.rglob(label)
            if path.is_dir()
            and any(
                child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
                for child in path.iterdir()
            )
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one image directory for {label!r}, found {len(candidates)}."
            )
        directories[label] = candidates[0]
    return directories


def difference_hash(image: Any) -> str:
    # Pillow is imported lazily so --help and existing-manifest reuse stay cheap.
    grayscale = image.convert("L").resize((9, 8))
    pixel_data = (
        grayscale.get_flattened_data()
        if hasattr(grayscale, "get_flattened_data")
        else grayscale.getdata()
    )
    pixels = list(pixel_data)
    bits = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            bits = (bits << 1) | int(left > right)
    return f"{bits:016x}"


def split_targets(size: int, validation_fraction: float, test_fraction: float) -> dict[str, int]:
    validation = round(size * validation_fraction)
    test = round(size * test_fraction)
    if size >= 3:
        validation = max(1, validation)
        test = max(1, test)
    train = size - validation - test
    if train < 1:
        raise RuntimeError(
            "The selected sample is too small for non-empty train/validation/test splits."
        )
    return {"train": train, "validation": validation, "test": test}


def near_duplicate_groups(
    records: list[dict[str, Any]], threshold: int
) -> list[list[dict[str, Any]]]:
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    hash_values = [int(record["perceptual_hash"], 16) for record in records]
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if (hash_values[left] ^ hash_values[right]).bit_count() <= threshold:
                left_root, right_root = find(left), find(right)
                if left_root != right_root:
                    parents[right_root] = left_root

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[find(index)].append(record)
    return list(grouped.values())


def assign_splits(
    records: list[dict[str, Any]],
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    near_duplicate_threshold: int,
) -> None:
    """Stratify by class while keeping same-label near-duplicate candidates together."""
    for label in CLASSES:
        class_records = [record for record in records if record["label"] == label]
        targets = split_targets(len(class_records), validation_fraction, test_fraction)
        ordered_groups = sorted(
            near_duplicate_groups(class_records, near_duplicate_threshold),
            key=lambda group: stable_rank(seed, label, group[0]["image_id"]),
        )
        remaining = targets.copy()
        for group in ordered_groups:
            split = max(
                SPLITS,
                key=lambda candidate: (
                    remaining[candidate],
                    stable_rank(seed, label, group[0]["image_id"], candidate),
                ),
            )
            for record in group:
                record["split"] = split
            remaining[split] -= len(group)


def audit_duplicates(
    records: list[dict[str, Any]], threshold: int
) -> dict[str, Any]:
    exact_cross_split = 0
    near_cross_split = 0
    same_label_near_cross_split = 0
    cross_label_hash_collisions = 0
    near_examples: list[dict[str, Any]] = []
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            if left["split"] == right["split"]:
                continue
            if left["sha256"] == right["sha256"]:
                exact_cross_split += 1
            distance = (
                int(left["perceptual_hash"], 16)
                ^ int(right["perceptual_hash"], 16)
            ).bit_count()
            if distance <= threshold:
                near_cross_split += 1
                if left["label"] == right["label"]:
                    same_label_near_cross_split += 1
                else:
                    cross_label_hash_collisions += 1
                if len(near_examples) < 20:
                    near_examples.append(
                        {
                            "left": left["image_id"],
                            "right": right["image_id"],
                            "distance": distance,
                        }
                    )
    return {
        "method": "SHA-256 exact match and 64-bit difference hash",
        "near_duplicate_hamming_threshold": threshold,
        "exact_cross_split_pairs": exact_cross_split,
        "near_cross_split_pairs": near_cross_split,
        "same_label_near_cross_split_pairs": same_label_near_cross_split,
        "cross_label_hash_collision_pairs": cross_label_hash_collisions,
        "near_cross_split_examples": near_examples,
        "note": (
            "Same-label near-duplicate candidates are grouped before splitting. "
            "Cross-label hash matches are reported as likely background collisions. "
            "Difference hashes are a warning signal, not a semantic duplicate detector."
        ),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = (
        "image_id",
        "image_path",
        "label",
        "split",
        "sha256",
        "perceptual_hash",
        "width",
        "height",
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.max_per_class is not None and args.max_per_class < 3:
        raise SystemExit("--max-per-class must be at least 3.")
    if not 0 < args.validation_fraction < 1 or not 0 < args.test_fraction < 1:
        raise SystemExit("Validation and test fractions must be between 0 and 1.")
    if args.validation_fraction + args.test_fraction >= 1:
        raise SystemExit("Validation and test fractions must sum to less than 1.")
    if not 0 <= args.near_duplicate_threshold <= 64:
        raise SystemExit("--near-duplicate-threshold must be between 0 and 64.")

    source = args.source.expanduser().resolve()
    archive = args.archive.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == source or source in output.parents or output in source.parents:
        raise SystemExit("Source and prepared-output directories must be separate.")

    if not source.exists():
        if not archive.exists():
            if not args.download:
                raise SystemExit(
                    f"RealWaste source not found at {source}. Pass --download, "
                    "--archive, or --source with an existing extracted copy."
                )
            download_archive(archive)
        print(f"Extracting {archive} to {source}")
        extract_archive(archive, source)
    if not source.is_dir():
        raise SystemExit(f"Source is not a directory: {source}")

    configuration = {
        "source": str(source),
        "source_archive_sha256": DATASET_SHA256,
        "max_per_class": args.max_per_class,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "near_duplicate_threshold": args.near_duplicate_threshold,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.csv"
    summary_path = output / "summary.json"
    if manifest_path.exists() and summary_path.exists() and not args.force:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("configuration") == configuration:
            print(f"Prepared manifest already matches this configuration: {manifest_path}")
            return
        raise SystemExit(
            "Output contains another configuration. Choose a different --output or use --force."
        )

    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Install Pillow first: python -m pip install pillow") from exc

    class_directories = find_class_directories(source)
    source_counts: Counter[str] = Counter()
    selected_paths: list[tuple[str, Path]] = []
    for label, directory in class_directories.items():
        paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        source_counts[label] = len(paths)
        if args.max_per_class is not None and args.max_per_class > len(paths):
            raise SystemExit(
                f"Requested {args.max_per_class} images from {label}, but only "
                f"{len(paths)} are available."
            )
        ranked = sorted(paths, key=lambda path: stable_rank(args.seed, label, path.name))
        selected_paths.extend(
            (label, path)
            for path in (
                ranked if args.max_per_class is None else ranked[: args.max_per_class]
            )
        )

    records: list[dict[str, Any]] = []
    for number, (label, path) in enumerate(selected_paths, start=1):
        with Image.open(path) as image:
            width, height = image.size
            perceptual_hash = difference_hash(image)
        records.append(
            {
                "image_id": path.relative_to(source).as_posix(),
                "image_path": Path(os.path.relpath(path, output)).as_posix(),
                "label": label,
                "sha256": sha256_file(path),
                "perceptual_hash": perceptual_hash,
                "width": width,
                "height": height,
            }
        )
        if number % 250 == 0 or number == len(selected_paths):
            print(f"Inspected {number}/{len(selected_paths)} images", flush=True)

    assign_splits(
        records,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    split_order = {name: index for index, name in enumerate(SPLITS)}
    records.sort(key=lambda row: (split_order[row["split"]], row["label"], row["image_id"]))
    duplicate_audit = audit_duplicates(records, args.near_duplicate_threshold)
    if duplicate_audit["exact_cross_split_pairs"]:
        raise RuntimeError("Exact duplicate images crossed prepared partitions.")

    write_csv(records, manifest_path)
    split_distribution = Counter(record["split"] for record in records)
    class_distribution = Counter(record["label"] for record in records)
    split_class_distribution: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts = Counter(record["label"] for record in records if record["split"] == split)
        split_class_distribution[split] = {label: counts[label] for label in CLASSES}

    summary = {
        "configuration": configuration,
        "source": {
            "dataset": "RealWaste",
            "uci_id": 908,
            "url": "https://archive.ics.uci.edu/dataset/908/realwaste",
            "download_url": DATASET_URL,
            "classes": list(CLASSES),
            "source_class_distribution": {label: source_counts[label] for label in CLASSES},
        },
        "generated": {
            "images": len(records),
            "split_distribution": dict(split_distribution),
            "class_distribution": {label: class_distribution[label] for label in CLASSES},
            "split_class_distribution": split_class_distribution,
            "manifest_sha256": sha256_file(manifest_path),
            "duplicate_audit": duplicate_audit,
        },
        "license": {
            "name": "Creative Commons Attribution 4.0 International",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": (
                "RealWaste, Single, Iranmanesh, and Raad (2023), UCI Machine "
                "Learning Repository, DOI 10.24432/C5SS4G."
            ),
        },
    }
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_summary.replace(summary_path)

    print(f"Prepared {len(records)} images in {manifest_path}")
    print(json.dumps(summary["generated"], indent=2))


if __name__ == "__main__":
    main()
