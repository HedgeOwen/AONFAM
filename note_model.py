from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

REQUIRED_COLUMNS = [
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
LABEL_COLUMNS = ["label", "correct_pitch"]
ACTION_NAMES = {0: "KEEP", 1: "REPLACE", 2: "DELETE"}
ACTION_IDS = {v: k for k, v in ACTION_NAMES.items()}


@dataclass
class PieceData:
    df_sorted: pd.DataFrame
    pitch: np.ndarray
    program: np.ndarray
    inst: np.ndarray
    velocity: np.ndarray
    duration: np.ndarray
    onset: np.ndarray
    beat: np.ndarray
    bar: np.ndarray
    label: Optional[np.ndarray] = None
    correct_pitch: Optional[np.ndarray] = None


class NoteWindowDataset(Dataset):
    def __init__(self, pieces: Sequence[PieceData], window_k: int, include_labels: bool = True):
        self.pieces = list(pieces)
        self.window_k = window_k
        self.include_labels = include_labels
        self.index_map: List[Tuple[int, int]] = []
        for p_idx, piece in enumerate(self.pieces):
            self.index_map.extend((p_idx, i) for i in range(len(piece.pitch)))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p_idx, center = self.index_map[idx]
        piece = self.pieces[p_idx]
        k = self.window_k
        length = 2 * k + 1

        pitch_seq = np.zeros(length, dtype=np.int64)
        program_seq = np.zeros(length, dtype=np.int64)
        inst_seq = np.zeros(length, dtype=np.int64)
        cont_seq = np.zeros((length, 5), dtype=np.float32)
        valid_mask = np.zeros(length, dtype=np.bool_)

        start = center - k
        end = center + k + 1
        center_onset = piece.onset[center]
        center_beat = piece.beat[center]
        center_bar = piece.bar[center]

        for j, src in enumerate(range(start, end)):
            if 0 <= src < len(piece.pitch):
                valid_mask[j] = True
                pitch_seq[j] = int(piece.pitch[src]) + 1
                program_seq[j] = int(piece.program[src]) + 1
                inst_seq[j] = int(piece.inst[src]) + 1
                delta_beat = piece.beat[src] - center_beat if center_beat >= 0 and piece.beat[src] >= 0 else 0.0
                delta_bar = piece.bar[src] - center_bar if center_bar >= 0 and piece.bar[src] >= 0 else 0.0
                cont_seq[j, 0] = piece.velocity[src] / 127.0
                safe_duration = max(float(piece.duration[src]), 0.0)
                cont_seq[j, 1] = float(np.log1p(safe_duration))
                cont_seq[j, 2] = piece.onset[src] - center_onset
                cont_seq[j, 3] = float(delta_beat)
                cont_seq[j, 4] = float(delta_bar)

        cont_seq = np.nan_to_num(cont_seq, nan=0.0, posinf=0.0, neginf=0.0)

        out: Dict[str, torch.Tensor] = {
            "pitch_seq": torch.from_numpy(pitch_seq),
            "program_seq": torch.from_numpy(program_seq),
            "inst_seq": torch.from_numpy(inst_seq),
            "cont_seq": torch.from_numpy(cont_seq),
            "valid_mask": torch.from_numpy(valid_mask),
        }

        if self.include_labels and piece.label is not None and piece.correct_pitch is not None:
            out["y_action"] = torch.tensor(int(piece.label[center]), dtype=torch.long)
            y_pitch = int(piece.correct_pitch[center]) if int(piece.label[center]) == 1 else -100
            out["y_pitch"] = torch.tensor(y_pitch, dtype=torch.long)
        return out


class NoteCorrectionModel(nn.Module):
    def __init__(
        self,
        num_programs: int,
        num_insts: int,
        pitch_emb_dim: int = 32,
        program_emb_dim: int = 16,
        inst_emb_dim: int = 16,
        cont_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.pitch_emb = nn.Embedding(129, pitch_emb_dim, padding_idx=0)
        self.program_emb = nn.Embedding(num_programs + 1, program_emb_dim, padding_idx=0)
        self.inst_emb = nn.Embedding(num_insts + 1, inst_emb_dim, padding_idx=0)
        in_dim = pitch_emb_dim + program_emb_dim + inst_emb_dim + cont_dim
        self.encoder = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * 2
        self.action_head = nn.Linear(out_dim, 3)
        self.pitch_head = nn.Linear(out_dim, 128)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Robustness for unseen/inconsistent MIDI metadata at inference time:
        # clamp categorical ids to embedding ranges to avoid CUDA index asserts.
        pitch_ids = batch["pitch_seq"].clamp(min=0, max=self.pitch_emb.num_embeddings - 1)
        program_ids = batch["program_seq"].clamp(min=0, max=self.program_emb.num_embeddings - 1)
        inst_ids = batch["inst_seq"].clamp(min=0, max=self.inst_emb.num_embeddings - 1)

        pitch_e = self.pitch_emb(pitch_ids)
        program_e = self.program_emb(program_ids)
        inst_e = self.inst_emb(inst_ids)
        x = torch.cat([pitch_e, program_e, inst_e, batch["cont_seq"]], dim=-1)
        enc, _ = self.encoder(x)
        center_idx = enc.size(1) // 2
        h_center = enc[:, center_idx, :]
        return {
            "logits_action": self.action_head(h_center),
            "logits_pitch": self.pitch_head(h_center),
        }


def read_table(path: Path, file_format: str) -> pd.DataFrame:
    if file_format == "csv":
        return pd.read_csv(path)
    if file_format == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {file_format}")


def write_table(df: pd.DataFrame, path: Path, file_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "csv":
        df.to_csv(path, index=False)
    elif file_format == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {file_format}")


def validate_columns(df: pd.DataFrame, require_labels: bool) -> None:
    required = REQUIRED_COLUMNS + (LABEL_COLUMNS if require_labels else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_piece(df: pd.DataFrame, require_labels: bool) -> PieceData:
    validate_columns(df, require_labels=require_labels)
    sorted_df = df.sort_values(["onset", "pitch", "note_id"]).reset_index(drop=False).rename(columns={"index": "orig_index"})
    kwargs = dict(
        df_sorted=sorted_df,
        pitch=sorted_df["pitch"].to_numpy(np.int64),
        program=sorted_df["program"].to_numpy(np.int64),
        inst=sorted_df["inst_id"].to_numpy(np.int64),
        velocity=sorted_df["velocity"].to_numpy(np.float32),
        duration=sorted_df["duration"].to_numpy(np.float32),
        onset=sorted_df["onset"].to_numpy(np.float32),
        beat=sorted_df["beat_id"].to_numpy(np.float32),
        bar=sorted_df["bar_id"].to_numpy(np.float32),
    )
    if require_labels:
        kwargs["label"] = sorted_df["label"].to_numpy(np.int64)
        kwargs["correct_pitch"] = sorted_df["correct_pitch"].to_numpy(np.int64)
    return PieceData(**kwargs)


def list_split_files(data_root: Path, split: str, file_format: str) -> List[Path]:
    split_dir = data_root / split
    if not split_dir.exists():
        return []
    files = sorted(split_dir.glob(f"*.{file_format}"))
    return files


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-9
    f1s = []
    for c in [0, 1, 2]:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))

    y_true_bin = (y_true != 0).astype(np.int64)
    y_pred_bin = (y_pred != 0).astype(np.int64)
    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1_bin = float(2 * p * r / (p + r + eps))
    return {"macro_f1": macro_f1, "error_f1": f1_bin}


def pitch_metrics(
    y_action_true: np.ndarray,
    y_pitch_true: np.ndarray,
    pitch_logits: np.ndarray,
    topk: int = 5,
) -> Dict[str, float]:
    mask = y_action_true == 1
    if mask.sum() == 0:
        return {"pitch_top1_acc": 0.0, "pitch_top5_acc": 0.0}
    logits = pitch_logits[mask]
    target = y_pitch_true[mask]
    top1 = logits.argmax(axis=1)
    top1_acc = float((top1 == target).mean())
    topk_idx = np.argpartition(-logits, kth=min(topk, logits.shape[1] - 1), axis=1)[:, :topk]
    topk_acc = float(np.any(topk_idx == target[:, None], axis=1).mean())
    return {"pitch_top1_acc": top1_acc, "pitch_top5_acc": topk_acc}
