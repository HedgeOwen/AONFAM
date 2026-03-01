from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pretty_midi
import torch
from torch.utils.data import DataLoader

from note_model import ACTION_NAMES, NoteCorrectionModel, NoteWindowDataset, build_piece, write_table


# =========================
# Shared MIDI -> Table Logic
# =========================
def midi_to_note_table(
    midi_path: Path,
    onset_bin_size: float = 0.01,
    drop_drums: bool = False,
) -> pd.DataFrame:
    """
    Convert one MIDI file to the note-event table schema expected by the model.
    This is intentionally aligned with midi_to_table.py.

    Output columns:
      piece_id, note_id, inst_id, program, is_drum,
      onset, offset, duration, pitch, velocity,
      onset_bin, beat_id, bar_id
    """
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
                    "beat_id": -1,  # baseline placeholder
                    "bar_id": -1,   # baseline placeholder
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

    # Global stable sort (important for deterministic note_id and model input order)
    df = df.sort_values(["onset", "offset", "inst_id", "pitch", "velocity"]).reset_index(drop=True)
    df.insert(1, "note_id", np.arange(len(df), dtype=np.int64))
    return df


# =========================
# Diagnose
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose one MIDI file and print note-level error suggestions")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--midi", type=Path, required=True, help="Path to a single MIDI file")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--threshold", type=float, default=0.05, help="Threshold on error_prob = P(REPLACE)+P(DELETE)")
    p.add_argument("--window_k", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max_print", type=int, default=100, help="Max predicted error notes to print")
    p.add_argument("--show_top_suspects", type=int, default=20, help="Always print top-N suspicious notes by error_prob")
    p.add_argument("--onset_bin_size", type=float, default=0.01)
    p.add_argument("--drop_drums", action="store_true")
    p.add_argument("--quiet_pretty_midi_warning", action="store_true")
    p.add_argument("--out_path", type=Path, default=None, help="Optional output table path (.csv/.parquet)")
    p.add_argument("--out_format", choices=["csv", "parquet"], default="csv")
    return p.parse_args()


def load_checkpoint(path: Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def safe_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to cpu")
        return "cpu"
    return requested


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1].")

    args.device = safe_device(args.device)

    if args.quiet_pretty_midi_warning:
        warnings.filterwarnings(
            "ignore",
            message="Tempo, Key or Time signature change events found on non-zero tracks.*",
            category=RuntimeWarning,
        )

    ckpt = load_checkpoint(args.checkpoint, map_location=args.device)
    window_k = args.window_k if args.window_k is not None else int(ckpt["window_k"])

    model = NoteCorrectionModel(**ckpt["model_args"]).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Build table from MIDI (aligned with midi_to_table.py)
    df = midi_to_note_table(
        args.midi,
        onset_bin_size=args.onset_bin_size,
        drop_drums=args.drop_drums,
    )

    piece = build_piece(df, require_labels=False)
    ds = NoteWindowDataset([piece], window_k=window_k, include_labels=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    action_logits_list = []
    pitch_logits_list = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(batch)
            action_logits_list.append(out["logits_action"].detach().cpu().numpy())
            pitch_logits_list.append(out["logits_pitch"].detach().cpu().numpy())

    if action_logits_list:
        action_logits = np.concatenate(action_logits_list, axis=0)
        pitch_logits = np.concatenate(pitch_logits_list, axis=0)
    else:
        action_logits = np.zeros((0, 3), dtype=np.float32)
        pitch_logits = np.zeros((0, 128), dtype=np.float32)

    action_probs = torch.softmax(torch.tensor(action_logits), dim=1).numpy() if len(action_logits) else np.zeros((0, 3), dtype=np.float32)
    pitch_probs = torch.softmax(torch.tensor(pitch_logits), dim=1).numpy() if len(pitch_logits) else np.zeros((0, 128), dtype=np.float32)

    # ------------------------------
    # SAME threshold logic as infer.py
    # error_prob = P(REPLACE) + P(DELETE)
    # ------------------------------
    if len(action_probs):
        p_keep = action_probs[:, 0]
        p_replace = action_probs[:, 1]
        p_delete = action_probs[:, 2]
        error_prob = p_replace + p_delete
    else:
        p_keep = np.array([], dtype=np.float32)
        p_replace = np.array([], dtype=np.float32)
        p_delete = np.array([], dtype=np.float32)
        error_prob = np.array([], dtype=np.float32)

    action_ids = np.zeros(len(action_probs), dtype=np.int64)
    if args.threshold > 0.0:
        err_mask = error_prob >= args.threshold
    else:
        # Compatible fallback behavior (same as infer.py)
        err_mask = action_probs.argmax(axis=1) != 0 if len(action_probs) else np.array([], dtype=bool)

    replace_ge_delete = p_replace >= p_delete if len(action_probs) else np.array([], dtype=bool)
    action_ids[err_mask & replace_ge_delete] = 1  # REPLACE
    action_ids[err_mask & (~replace_ge_delete)] = 2  # DELETE

    # Keep semantic aligned with infer.py output
    action_conf = error_prob

    if len(pitch_probs):
        suggest_pitch = pitch_probs.argmax(axis=1)
        sorted_idx = np.argsort(-pitch_probs, axis=1)
        topk_idx = sorted_idx[:, : args.topk]
        topk_probs = np.take_along_axis(pitch_probs, topk_idx, axis=1)
    else:
        suggest_pitch = np.array([], dtype=np.int64)
        topk_idx = np.zeros((0, args.topk), dtype=np.int64)
        topk_probs = np.zeros((0, args.topk), dtype=np.float32)

    out_df = piece.df_sorted.copy()
    out_df["pred_action"] = [ACTION_NAMES[int(i)] for i in action_ids]
    out_df["pred_action_id"] = action_ids
    out_df["pred_action_prob"] = action_conf          # == pred_error_prob in this version
    out_df["pred_error_prob"] = error_prob
    out_df["p_keep"] = p_keep
    out_df["p_replace"] = p_replace
    out_df["p_delete"] = p_delete
    out_df["pred_suggest_pitch"] = np.where(action_ids == 1, suggest_pitch, -1)
    out_df["pred_topk_pitches"] = [
        json.dumps(v.tolist(), ensure_ascii=False) if action_ids[i] == 1 else "[]"
        for i, v in enumerate(topk_idx)
    ]
    out_df["pred_topk_probs"] = [
        json.dumps(v.tolist(), ensure_ascii=False) if action_ids[i] == 1 else "[]"
        for i, v in enumerate(topk_probs)
    ]

    restored = out_df.sort_values("orig_index").drop(columns=["orig_index"]).reset_index(drop=True)

    error_df = restored[restored["pred_action_id"] != 0].copy()

    print(f"midi={args.midi}")
    print(f"total_notes={len(restored)}")
    print(f"predicted_errors={len(error_df)} (threshold={args.threshold})")
    if len(error_prob) > 0:
        print(
            f"error_prob_stats: min={float(error_prob.min()):.6f} "
            f"mean={float(error_prob.mean()):.6f} "
            f"max={float(error_prob.max()):.6f}"
        )

    if len(error_df) == 0:
        print("No predicted errors above threshold.")
    else:
        cols = [
            "note_id", "onset", "pitch",
            "pred_action", "pred_action_prob",
            "p_keep", "p_replace", "p_delete",
            "pred_suggest_pitch", "pred_topk_pitches"
        ]
        preview = error_df.loc[:, cols].head(args.max_print)
        print(preview.to_string(index=False))
        if len(error_df) > args.max_print:
            print(f"... truncated, showing first {args.max_print} / {len(error_df)} error notes")

    # Always print suspicious notes for demo stability
    if args.show_top_suspects > 0 and len(restored) > 0:
        suspect_df = restored.sort_values("pred_error_prob", ascending=False).head(args.show_top_suspects)
        cols = ["note_id", "onset", "pitch", "pred_error_prob", "p_keep", "p_replace", "p_delete"]
        print(f"\nTop-{args.show_top_suspects} suspicious notes by error_prob:")
        print(suspect_df.loc[:, cols].to_string(index=False))

    if args.out_path is not None:
        write_table(restored, args.out_path, args.out_format)
        print(f"wrote: {args.out_path}")


if __name__ == "__main__":
    main()