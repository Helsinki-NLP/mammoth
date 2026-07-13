#!/usr/bin/env python3
# common_files.py

from __future__ import annotations

import argparse
from pathlib import Path


def collect_files(root: Path, recursive: bool) -> set[str]:
    if recursive:
        return {
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file()
        }
    return {
        p.name
        for p in root.iterdir()
        if p.is_file()
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Show files common to several directories."
    )
    ap.add_argument("dirs", nargs="+", help="Directories to compare")
    ap.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Compare relative paths recursively, not just basenames"
    )
    ap.add_argument(
        "--only-common",
        action="store_true",
        help="Print only files present in all directories"
    )
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dirs]

    for d in dirs:
        if not d.is_dir():
            raise SystemExit(f"Not a directory: {d}")

    file_sets = {d: collect_files(d, args.recursive) for d in dirs}

    common = set.intersection(*file_sets.values()) if file_sets else set()

    print(f"Directories: {len(dirs)}")
    for d in dirs:
        print(f"{d}: {len(file_sets[d])} files")
    print(f"Common to all: {len(common)}")
    print()

    if args.only_common:
        for f in sorted(common):
            print(f)
        return 0

    print("Files common to all directories:")
    for f in sorted(common):
        print(f"  {f}")

    print()
    print("Files missing from some directories:")
    all_files = set.union(*file_sets.values()) if file_sets else set()

    for f in sorted(all_files - common):
        present = [str(d) for d in dirs if f in file_sets[d]]
        missing = [str(d) for d in dirs if f not in file_sets[d]]
        print(f"{f}")
        print(f"  present: {', '.join(present)}")
        print(f"  missing: {', '.join(missing)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

