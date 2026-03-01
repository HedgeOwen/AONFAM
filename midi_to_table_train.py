from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pretty_midi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert MAESTRO MIDI files to note-event tables")
    p.add_argument("--maestro_root", type=Path, required=True, help="Path to maestro-v3.0.0 root")
    p.add_argument("--metadata_csv", type=Path, default=None, help="Path to maestro metadata csv (default: <maestro_root>/maestro-v3.0.0.csv)")
    p.add_argument("--output_root", type=Path, required=True, help="Output root with train/validation/test subfolders")
    p.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    p.add_argument("--onset_bin_size", type=float, default=0.01, help="Seconds per onset bin")
    p.add_argument("--max_files_per_split", type=int, default=0)
    return p.parse_args()


def write_table(df: pd.DataFrame, out_path: Path, file_format: str) -> None:
    # Keep IO format handling consistent across scripts.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)


def pick_program_and_drum(inst: pretty_midi.Instrument) -> Tuple[int, int]:
    # Helper to normalize pretty_midi fields into ints.
    return int(inst.program), int(inst.is_drum)


def build_note_rows(pm: pretty_midi.PrettyMIDI, piece_id: str, onset_bin_size: float) -> List[dict]:
    rows: List[dict] = []
    note_id = 0

    # Beat/bar placeholders if timing map is unavailable.
    # MAESTRO has expressive timing; robust beat/bar extraction is optional in this baseline.
    for inst_id, inst in enumerate(pm.instruments):
        program, is_drum = pick_program_and_drum(inst)
        for n in inst.notes:
            onset = float(n.start)
            offset = float(n.end)
            duration = max(0.0, offset - onset)
            onset_bin = int(round(onset / onset_bin_size)) if onset_bin_size > 0 else int(round(onset * 100))
            # One row per note event.
            rows.append(
                {
                    "piece_id": piece_id,
                    "note_id": note_id,
                    "inst_id": inst_id,
                    "program": program,
                    "is_drum": is_drum,
                    "onset": onset,
                    "offset": offset,
                    "duration": duration,
                    "pitch": int(n.pitch),
                    "velocity": int(max(1, min(127, n.velocity))),
                    "onset_bin": onset_bin,
                    "beat_id": -1,
                    "bar_id": -1,
                }
            )
            note_id += 1
    return rows


def convert_one(midi_path: Path, out_path: Path, piece_id: str, file_format: str, onset_bin_size: float) -> None:
    # Parse MIDI and write a schema-compatible table.
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    rows = build_note_rows(pm, piece_id=piece_id, onset_bin_size=onset_bin_size)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        # keep empty schema-compatible table
        df = pd.DataFrame(
            columns=[
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
        )
    write_table(df, out_path, file_format)


def main() -> None:
    args = parse_args()
    meta_path = args.metadata_csv or (args.maestro_root / "maestro-v3.0.0.csv")
    meta = pd.read_csv(meta_path)

    required = {"split", "midi_filename"}
    missing = required - set(meta.columns)
    if missing:
        raise RuntimeError(f"metadata csv missing columns: {sorted(missing)}")

    total = 0
    for split in ["train", "validation", "test"]:
        split_df = meta[meta["split"] == split].copy()
        if args.max_files_per_split > 0:
            split_df = split_df.iloc[: args.max_files_per_split]

        out_split = args.output_root / split
        out_split.mkdir(parents=True, exist_ok=True)

        for _, row in split_df.iterrows():
            rel_midi = Path(str(row["midi_filename"]))
            midi_path = args.maestro_root / rel_midi
            if not midi_path.exists():
                print(f"skip missing midi: {midi_path}")
                continue

            # Derive a stable piece id from the MAESTRO relative path.
            piece_id = rel_midi.with_suffix("").as_posix().replace("/", "__")
            out_path = out_split / f"{rel_midi.stem}.{args.format}"
            convert_one(midi_path, out_path, piece_id, args.format, args.onset_bin_size)
            total += 1

        print(f"converted split={split}, files={len(split_df)}")

    print(f"done: total_converted={total}")


if __name__ == "__main__":
    main()
