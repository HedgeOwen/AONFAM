from __future__ import annotations

import argparse
from pathlib import Path

from note_model import read_table, write_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare clean-only note tables for training")
    p.add_argument("--clean_root", type=Path, required=True, help="Root with train/validation/test clean note tables")
    p.add_argument("--output_root", type=Path, required=True, help="Where to write labeled clean-only files")
    p.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    p.add_argument("--merge_into", type=Path, default=None, help="Optional target root to merge outputs into")
    p.add_argument("--suffix", type=str, default="_clean_labeled", help="Suffix appended to output stem")
    p.add_argument("--max_files_per_split", type=int, default=0)
    return p.parse_args()


def iter_split_files(root: Path, split: str, file_format: str):
    # List all tables in a given split directory.
    split_dir = root / split
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob(f"*.{file_format}"))


def main() -> None:
    args = parse_args()
    splits = ["train", "validation", "test"]

    total = 0
    for split in splits:
        files = iter_split_files(args.clean_root, split, args.format)
        if args.max_files_per_split > 0:
            files = files[: args.max_files_per_split]

        out_split = args.output_root / split
        out_split.mkdir(parents=True, exist_ok=True)

        if args.merge_into is not None:
            merge_split = args.merge_into / split
            merge_split.mkdir(parents=True, exist_ok=True)
        else:
            merge_split = None

        for fp in files:
            df = read_table(fp, args.format)
            # Clean-only tables are labeled as KEEP with no correction pitch.
            df["label"] = 0
            df["correct_pitch"] = -1
            if "case" not in df.columns:
                df["case"] = "NONE"
            if "error_case_color" not in df.columns:
                df["error_case_color"] = "WW"

            out_name = f"{fp.stem}{args.suffix}.{args.format}"
            out_path = out_split / out_name
            write_table(df, out_path, args.format)

            if merge_split is not None:
                # Optionally merge into a shared target root.
                write_table(df, merge_split / out_name, args.format)

            total += 1

        print(f"prepared split={split}, files={len(files)}")

    print(f"done: total_prepared_files={total}")


if __name__ == "__main__":
    main()
