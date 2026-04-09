#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "scrabble-player-rating.zip"
EXTRACT_DIR = RAW_DIR / "scrabble-player-rating"
REQUIRED_FILES = {
    "games.csv",
    "sample_submission.csv",
    "test.csv",
    "train.csv",
    "turns.csv",
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_extracted_files() -> list[str]:
    missing = []
    for name in sorted(REQUIRED_FILES):
        if not (EXTRACT_DIR / name).exists():
            missing.append(name)
    return missing


def extract_dataset(zf: zipfile.ZipFile) -> None:
    members = [member for member in zf.infolist() if not member.is_dir()]

    prefixes = {Path(member.filename).parts[0] for member in members if len(Path(member.filename).parts) > 1}
    strip_root = len(prefixes) == 1

    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    for member in members:
        parts = Path(member.filename).parts
        relative_parts = parts[1:] if strip_root and len(parts) > 1 else parts
        if not relative_parts:
            continue

        destination = EXTRACT_DIR.joinpath(*relative_parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print(f"Missing dataset archive: {ZIP_PATH}")
        print("Download it from the shared Drive folder and place it in data/raw/.")
        return 1

    print(f"Dataset archive found: {ZIP_PATH}")
    print(f"SHA256: {sha256sum(ZIP_PATH)}")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        extract_dataset(zf)

    missing = validate_extracted_files()
    if missing:
        print("Extraction completed, but some expected files are missing:")
        for name in missing:
            print(f" - {name}")
        return 1

    print(f"Dataset extracted successfully to: {EXTRACT_DIR}")
    print("Available files:")
    for name in sorted(REQUIRED_FILES):
        print(f" - {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
