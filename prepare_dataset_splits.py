"""Validate the canonical train/validation split used by the original project."""

import argparse
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get("SEGFORMER_RDA_DATASET_ROOT", PROJECT_ROOT / "data" / "liefeng"))
VOC_ROOT = DATASET_ROOT / "VOC2007"
SPLIT_DIR = VOC_ROOT / "ImageSets" / "Segmentation"

EXPECTED_COUNTS = {"train": 1147, "val": 203}


def read_split(name):
    path = SPLIT_DIR / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def copy_source_splits(source_dir):
    source_dir = Path(source_dir).expanduser().resolve()
    for name in EXPECTED_COUNTS:
        source = source_dir / f"{name}.txt"
        if not source.is_file():
            raise FileNotFoundError(f"Missing source split file: {source}")
        shutil.copyfile(source, SPLIT_DIR / source.name)


def remove_unused_splits():
    for name in ("test.txt", "trainval.txt"):
        path = SPLIT_DIR / name
        if path.exists():
            path.unlink()


def validate_splits():
    splits = {name: read_split(name) for name in EXPECTED_COUNTS}
    for name, expected in EXPECTED_COUNTS.items():
        actual = len(splits[name])
        if actual != expected:
            raise RuntimeError(f"Expected {expected} entries in {name}.txt, found {actual}")

    overlap = set(splits["train"]) & set(splits["val"])
    if overlap:
        raise RuntimeError(f"Train/validation overlap detected: {sorted(overlap)[:5]}")

    total = len(set(splits["train"]) | set(splits["val"]))
    if total != 1350:
        raise RuntimeError(f"Expected 1350 unique patches, found {total}")

    print(f"Validated patch-level split: train={len(splits['train'])}, val={len(splits['val'])}, total={total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-split-dir",
        help="Optional directory containing the canonical train.txt and val.txt files.",
    )
    args = parser.parse_args()
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    if args.source_split_dir:
        copy_source_splits(args.source_split_dir)
    remove_unused_splits()
    validate_splits()


if __name__ == "__main__":
    main()
