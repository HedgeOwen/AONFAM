from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from note_model import read_table, write_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inject substitution/extra-note errors into clean note tables and write labeled outputs"
    )
    p.add_argument("--input_root", type=Path, required=True, help="Root containing train/validation/test clean tables")
    p.add_argument("--output_root", type=Path, required=True, help="Output root for *_labeled files")
    p.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sub_rate", type=float, default=0.05, help="Substitution rate among original notes")
    p.add_argument("--extra_rate", type=float, default=0.02, help="Extra-note rate among original notes")
    p.add_argument("--min_pitch", type=int, default=0)
    p.add_argument("--max_pitch", type=int, default=127)
    p.add_argument("--max_shift", type=int, default=6, help="Max absolute semitone shift for substitution/extra")
    p.add_argument("--max_files_per_split", type=int, default=0, help="0 means all files")
    return p.parse_args()


# Required columns for the clean note tables (shared schema across scripts).
REQUIRED = [
    "piece_id",
    "note_id",
    "inst_id",
    "program",
    "is_drum",
    "onset",
    "offset",
    "duration",
    "pitch",
    "velocity",
    "onset_bin",
    "beat_id",
    "bar_id",
]


def _validate(df: pd.DataFrame, path: Path) -> None:
    # Fail fast if input tables are missing the expected schema.
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def _random_shift(rng: np.random.Generator, max_shift: int) -> int:
    # Sample a non-zero semitone shift within [-max_shift, max_shift].
    if max_shift <= 0:
        return 1
    choices = np.arange(-max_shift, max_shift + 1)
    choices = choices[choices != 0]
    return int(rng.choice(choices))


def inject_piece_errors(
    df: pd.DataFrame,
    rng: np.random.Generator,
    sub_rate: float,
    extra_rate: float,
    min_pitch: int,
    max_pitch: int,
    max_shift: int,
) -> pd.DataFrame:
    _validate(df, Path("<in-memory>"))
    out = df.copy()
    n = len(out)
    if n == 0:
        # Keep schema-compatible empty tables.
        out["label"] = pd.Series(dtype=np.int64)
        out["correct_pitch"] = pd.Series(dtype=np.int64)
        out["case"] = pd.Series(dtype=str)
        out["error_case_color"] = pd.Series(dtype=str)
        return out

    # Initialize label columns:
    # label=0 KEEP, label=1 REPLACE, label=2 DELETE (extra note).
    out["label"] = 0
    out["correct_pitch"] = -1
    out["case"] = "NONE"
    out["error_case_color"] = "WW"

    indices = np.arange(n)
    # Pick distinct indices for substitution and extra-note injection.
    sub_n = min(n, int(round(n * max(sub_rate, 0.0))))
    sub_idx = rng.choice(indices, size=sub_n, replace=False) if sub_n > 0 else np.array([], dtype=np.int64)

    remain = np.setdiff1d(indices, sub_idx, assume_unique=False)
    extra_n = min(len(remain), int(round(n * max(extra_rate, 0.0))))
    extra_base_idx = rng.choice(remain, size=extra_n, replace=False) if extra_n > 0 else np.array([], dtype=np.int64)

    # Substitution: shift pitch of existing notes and record the original pitch.
    for i in sub_idx:
        old_pitch = int(out.at[i, "pitch"])
        new_pitch = int(np.clip(old_pitch + _random_shift(rng, max_shift), min_pitch, max_pitch))
        if new_pitch == old_pitch:
            new_pitch = min(max_pitch, old_pitch + 1) if old_pitch < max_pitch else max(min_pitch, old_pitch - 1)
        out.at[i, "pitch"] = new_pitch
        out.at[i, "label"] = 1
        out.at[i, "correct_pitch"] = old_pitch

    extra_rows: List[pd.Series] = []
    next_note_id = int(pd.to_numeric(out["note_id"], errors="coerce").max()) + 1 if n > 0 else 0
    # Extra-note injection: duplicate timing fields, but shift pitch and mark as label=2.
    for i in extra_base_idx:
        base = out.loc[i].copy()
        base_pitch = int(base["pitch"])
        extra_pitch = int(np.clip(base_pitch + _random_shift(rng, max_shift), min_pitch, max_pitch))
        if extra_pitch == base_pitch:
            extra_pitch = min(max_pitch, base_pitch + 1) if base_pitch < max_pitch else max(min_pitch, base_pitch - 1)
        base["note_id"] = next_note_id
        next_note_id += 1
        base["pitch"] = extra_pitch
        base["label"] = 2
        base["correct_pitch"] = -1
        extra_rows.append(base)

    if extra_rows:
        out = pd.concat([out, pd.DataFrame(extra_rows)], ignore_index=True)

    # Piece-level case tag for downstream summary/visualization.
    has_sub = len(sub_idx) > 0
    has_extra = len(extra_base_idx) > 0
    if has_sub and has_extra:
        piece_case = "BOTH"
    elif has_sub:
        piece_case = "SUB_ONLY"
    elif has_extra:
        piece_case = "EXTRA_ONLY"
    else:
        piece_case = "NONE"
    out["case"] = piece_case
    out["error_case_color"] = "WW"

    # Deterministic ordering for training and inspection.
    out = out.sort_values(["onset", "pitch", "note_id"]).reset_index(drop=True)
    return out


def process_split(split_dir: Path, out_dir: Path, file_format: str, rng: np.random.Generator, args: argparse.Namespace) -> int:
    files = sorted(split_dir.glob(f"*.{file_format}"))
    if args.max_files_per_split > 0:
        files = files[: args.max_files_per_split]
    count = 0
    for fp in files:
        df = read_table(fp, file_format)
        _validate(df, fp)
        # Apply error injection per piece file.
        labeled = inject_piece_errors(
            df,
            rng=rng,
            sub_rate=args.sub_rate,
            extra_rate=args.extra_rate,
            min_pitch=args.min_pitch,
            max_pitch=args.max_pitch,
            max_shift=args.max_shift,
        )
        out_name = f"{fp.stem}_labeled.{file_format}"
        out_path = out_dir / out_name
        write_table(labeled, out_path, file_format)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    total = 0
    for split in ["train", "validation", "test"]:
        in_dir = args.input_root / split
        out_dir = args.output_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        if not in_dir.exists():
            print(f"skip split={split}: missing {in_dir}")
            continue
        # Process each split independently so missing splits do not abort the run.
        c = process_split(in_dir, out_dir, args.format, rng, args)
        total += c
        print(f"processed split={split}, files={c}")
    print(f"done: total_files={total}")


if __name__ == "__main__":
    main()
