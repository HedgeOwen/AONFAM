from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from note_model import ACTION_NAMES, NoteCorrectionModel, NoteWindowDataset, build_piece, read_table, write_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--format", choices=["csv", "parquet"], default="csv")
    p.add_argument("--window_k", type=int, default=None)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--recursive", action="store_true", help="Recursively scan input directory")
    p.add_argument(
        "--error_threshold",
        type=float,
        default=0.0,
        help=(
            "Optional post-processing threshold for non-KEEP actions. "
            "If pred_action_id != KEEP and pred_action_prob < threshold, action is forced to KEEP."
        ),
    )
    return p.parse_args()


def collect_files(path: Path, file_format: str, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    pattern = f"**/*.{file_format}" if recursive else f"*.{file_format}"
    return sorted(path.glob(pattern))


def load_checkpoint(path: Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def run_one_file(model, file_path: Path, args, window_k: int, root_input: Path) -> None:
    df = read_table(file_path, args.format)
    piece = build_piece(df, require_labels=False)
    ds = NoteWindowDataset([piece], window_k=window_k, include_labels=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    action_logits = []
    pitch_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(batch)
            action_logits.append(out["logits_action"].cpu().numpy())
            pitch_logits.append(out["logits_pitch"].cpu().numpy())

    action_logits = np.concatenate(action_logits)
    pitch_logits = np.concatenate(pitch_logits)

    action_probs = torch.softmax(torch.tensor(action_logits), dim=1).numpy()

    # error probability = P(REPLACE)+P(DELETE)
    error_prob = action_probs[:, 1] + action_probs[:, 2]

    # threshold on error_prob
    action_ids = np.zeros(len(action_probs), dtype=np.int64)
    if args.error_threshold > 0.0:
        err_mask = error_prob >= args.error_threshold
    else:
        # 不设阈值时等价于 argmax!=KEEP（兼容旧行为）
        err_mask = action_probs.argmax(axis=1) != 0

    replace_ge_delete = action_probs[:, 1] >= action_probs[:, 2]
    action_ids[err_mask & replace_ge_delete] = 1
    action_ids[err_mask & (~replace_ge_delete)] = 2

    # 输出给用户看的置信度统一用 error_prob
    action_conf = error_prob

    pitch_probs = torch.softmax(torch.tensor(pitch_logits), dim=1).numpy()
    suggest_pitch = pitch_probs.argmax(axis=1)
    sorted_idx = np.argsort(-pitch_probs, axis=1)
    topk_idx = sorted_idx[:, : args.topk]
    topk_probs = np.take_along_axis(pitch_probs, topk_idx, axis=1)

    out_df = piece.df_sorted.copy()
    out_df["pred_action"] = [ACTION_NAMES[int(i)] for i in action_ids]
    out_df["pred_action_id"] = action_ids
    out_df["pred_action_prob"] = action_conf
    out_df["pred_suggest_pitch"] = np.where(action_ids == 1, suggest_pitch, -1)
    out_df["pred_topk_pitches"] = [json.dumps(v.tolist(), ensure_ascii=False) if action_ids[i] == 1 else "[]" for i, v in enumerate(topk_idx)]
    out_df["pred_topk_probs"] = [json.dumps(v.tolist(), ensure_ascii=False) if action_ids[i] == 1 else "[]" for i, v in enumerate(topk_probs)]
    out_df["pred_error_prob"] = error_prob
    out_df["p_keep"] = action_probs[:, 0]
    out_df["p_replace"] = action_probs[:, 1]
    out_df["p_delete"] = action_probs[:, 2]

    restored = out_df.sort_values("orig_index").drop(columns=["orig_index"]).reset_index(drop=True)
    if root_input.is_dir() and args.recursive:
        rel = file_path.relative_to(root_input)
        out_path = args.output_dir / rel
    else:
        out_path = args.output_dir / file_path.name
    write_table(restored, out_path, args.format)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.error_threshold <= 1.0:
        raise ValueError("--error_threshold must be in [0, 1].")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.checkpoint, map_location=args.device)
    window_k = args.window_k if args.window_k is not None else int(ckpt["window_k"])

    model = NoteCorrectionModel(**ckpt["model_args"]).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    files = collect_files(args.input_path, args.format, args.recursive)
    if not files:
        raise RuntimeError("No input files found for inference.")

    for fp in files:
        run_one_file(model, fp, args, window_k, args.input_path)
        print(f"wrote: {fp}")


if __name__ == "__main__":
    main()