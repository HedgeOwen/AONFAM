from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pretty_midi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert one MIDI file to a note-event table")
    p.add_argument("--midi", type=Path, required=True, help="Path to a single MIDI file")
    p.add_argument("--out", type=Path, required=True, help="Output table path (.csv or .parquet)")
    p.add_argument("--format", choices=["csv", "parquet"], default="csv")
    p.add_argument("--onset_bin_size", type=float, default=0.01)
    p.add_argument("--drop_drums", action="store_true")
    return p.parse_args()


def write_table(df: pd.DataFrame, out_path: Path, file_format: str) -> None:
    # Keep IO format handling consistent across scripts.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)


def midi_to_note_table(
    midi_path: Path,
    onset_bin_size: float = 0.01,
    drop_drums: bool = False,
) -> pd.DataFrame:
    """
    Convert one MIDI to the same schema used by midi_diagnose.py.
    """
    # Parse MIDI into a flat note table.
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    rows: List[dict] = []
    piece_id = midi_path.stem

    for inst_id, inst in enumerate(pm.instruments):
        if drop_drums and inst.is_drum:
            continue

        for n in inst.notes:
            onset = float(n.start)
            offset = float(n.end)
            duration = max(0.0, offset - onset)
            onset_bin = int(round(onset / onset_bin_size)) if onset_bin_size > 0 else int(round(onset * 100))

            # One row per note; beat/bar placeholders are -1 for this simple converter.
            rows.append(
                {
                    "piece_id": piece_id,
                    "inst_id": int(inst_id),
                    "program": int(inst.program),
                    "is_drum": int(inst.is_drum),
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

    if not rows:
        return pd.DataFrame(
            columns=[
                "piece_id", "note_id", "inst_id", "program", "is_drum",
                "onset", "offset", "duration", "pitch", "velocity",
                "onset_bin", "beat_id", "bar_id",
            ]
        )

    df = pd.DataFrame(rows)
    # Stable ordering and explicit note_id assignment.
    df = df.sort_values(["onset", "offset", "inst_id", "pitch", "velocity"]).reset_index(drop=True)
    df.insert(1, "note_id", np.arange(len(df), dtype=np.int64))
    return df


def main() -> None:
    args = parse_args()
    df = midi_to_note_table(
        args.midi,
        onset_bin_size=args.onset_bin_size,
        drop_drums=args.drop_drums,
    )
    # Write output and report basic stats.
    write_table(df, args.out, args.format)
    print(f"converted: {args.midi}")
    print(f"rows={len(df)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
