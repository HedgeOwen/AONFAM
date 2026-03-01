from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from note_model import (
    NoteCorrectionModel,
    NoteWindowDataset,
    build_piece,
    classification_metrics,
    list_split_files,
    pitch_metrics,
    read_table,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--format", choices=["csv", "parquet"], default="csv")
    p.add_argument("--window_k", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_pitch", type=float, default=1.0)
    p.add_argument("--save_dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early_stop_patience", type=int, default=0, help="0 disables early stopping")
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--max_files_per_split", type=int, default=0, help="0 means use all files")
    p.add_argument("--skip_test_eval", action="store_true", help="Skip final test evaluation for faster smoke runs")
    p.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="Use weighted sampling by action label to counter heavy KEEP-class imbalance",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(data_root: Path, split: str, file_format: str, max_files: int = 0) -> List:
    pieces = []
    files = list_split_files(data_root, split, file_format)
    if max_files > 0:
        files = files[:max_files]
    for fp in files:
        df = read_table(fp, file_format)
        pieces.append(build_piece(df, require_labels=True))
    return pieces


def eval_epoch(
    model: NoteCorrectionModel,
    loader: DataLoader,
    device: str,
    lambda_pitch: float,
    class_weights: Optional[torch.Tensor],
) -> Dict[str, float]:
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    y_pitch_true = []
    logits_pitch_all = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            la = F.cross_entropy(out["logits_action"], batch["y_action"], weight=class_weights)
            lp = safe_pitch_loss(out["logits_pitch"], batch["y_pitch"])
            losses.append((la + lambda_pitch * lp).item())

            pred_action = out["logits_action"].argmax(dim=1)
            y_true.append(batch["y_action"].cpu().numpy())
            y_pred.append(pred_action.cpu().numpy())
            y_pitch_true.append(batch["y_pitch"].cpu().numpy())
            logits_pitch_all.append(out["logits_pitch"].cpu().numpy())

    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    y_pitch_true_np = np.concatenate(y_pitch_true)
    logits_pitch_np = np.concatenate(logits_pitch_all)

    metrics = classification_metrics(y_true_np, y_pred_np)
    metrics.update(pitch_metrics(y_true_np, y_pitch_true_np, logits_pitch_np, topk=5))
    end_to_end_mask = y_true_np == y_pred_np
    replace_mask = y_true_np == 1
    replace_ok = np.ones_like(end_to_end_mask, dtype=bool)
    replace_ok[replace_mask] = logits_pitch_np.argmax(axis=1)[replace_mask] == y_pitch_true_np[replace_mask]
    metrics["end_to_end_acc"] = float((end_to_end_mask & replace_ok).mean())
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def safe_pitch_loss(logits_pitch: torch.Tensor, y_pitch: torch.Tensor) -> torch.Tensor:
    valid = y_pitch != -100
    if valid.any():
        return F.cross_entropy(logits_pitch[valid], y_pitch[valid])
    return logits_pitch.new_zeros(())


def load_checkpoint(path: Path, map_location: str) -> Dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class NumpyWeightedSampler(Sampler[int]):
    """Weighted sampler implemented via NumPy to avoid torch.multinomial limits."""

    def __init__(self, weights: np.ndarray, num_samples: int) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        w = np.asarray(weights, dtype=np.float64)
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("weights must have a positive finite sum")
        self.prob = w / total
        self.num_samples = int(num_samples)

    def __iter__(self):
        idx = np.random.choice(len(self.prob), size=self.num_samples, replace=True, p=self.prob)
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    train_pieces = load_split(args.data_root, "train", args.format, args.max_files_per_split)
    val_pieces = load_split(args.data_root, "validation", args.format, args.max_files_per_split)
    test_pieces = load_split(args.data_root, "test", args.format, args.max_files_per_split)
    if not train_pieces:
        raise RuntimeError("No training files found.")
    print(f"loaded files: train={len(train_pieces)} val={len(val_pieces)} test={len(test_pieces)}")

    num_programs = max(int(p.program.max()) for p in train_pieces) + 1
    num_insts = max(int(p.inst.max()) for p in train_pieces) + 1

    train_ds = NoteWindowDataset(train_pieces, args.window_k, include_labels=True)
    val_ds = NoteWindowDataset(val_pieces, args.window_k, include_labels=True) if val_pieces else None
    test_ds = NoteWindowDataset(test_pieces, args.window_k, include_labels=True) if test_pieces else None

    if args.use_weighted_sampler:
        labels_for_sampler = np.concatenate([p.label for p in train_pieces])
        counts_for_sampler = np.bincount(labels_for_sampler, minlength=3).astype(np.float64)
        inv_freq = 1.0 / np.maximum(counts_for_sampler, 1.0)
        sample_weights = inv_freq[labels_for_sampler]
        if len(labels_for_sampler) >= (1 << 24):
            print("weighted sampler fallback: dataset has >=2^24 samples, using NumPy sampler")
            sampler = NumpyWeightedSampler(weights=sample_weights, num_samples=len(labels_for_sampler))
        else:
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(labels_for_sampler),
                replacement=True,
            )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False) if test_ds else None

    labels = np.concatenate([p.label for p in train_pieces])
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1)
    class_weights = torch.tensor(class_weights / class_weights.sum() * 3.0, device=args.device)

    model = NoteCorrectionModel(num_programs=num_programs, num_insts=num_insts).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_metric = -1.0
    best_epoch = 0
    logs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(batch)
            loss_action = F.cross_entropy(out["logits_action"], batch["y_action"], weight=class_weights)
            loss_pitch = safe_pitch_loss(out["logits_pitch"], batch["y_pitch"])
            loss = loss_action + args.lambda_pitch * loss_pitch
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses))}
        if val_loader:
            val_metrics = eval_epoch(model, val_loader, args.device, args.lambda_pitch, class_weights)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            select_metric = val_metrics["macro_f1"]
        else:
            select_metric = -row["train_loss"]
        logs.append(row)
        print(json.dumps(row, ensure_ascii=False))

        ckpt = {
            "model_state": model.state_dict(),
            "model_args": {"num_programs": num_programs, "num_insts": num_insts},
            "window_k": args.window_k,
            "format": args.format,
            "epoch": epoch,
        }
        torch.save(ckpt, args.save_dir / "last.pt")
        if select_metric > best_metric:
            best_metric = select_metric
            best_epoch = epoch
            torch.save(ckpt, args.save_dir / "best.pt")

        if args.early_stop_patience > 0 and (epoch - best_epoch) >= args.early_stop_patience:
            print(f"early_stop at epoch={epoch}, best_epoch={best_epoch}")
            break

    if test_loader and not args.skip_test_eval:
        best = load_checkpoint(args.save_dir / "best.pt", map_location=args.device)
        model.load_state_dict(best["model_state"])
        test_metrics = eval_epoch(model, test_loader, args.device, args.lambda_pitch, class_weights)
        print("test_metrics", json.dumps(test_metrics, ensure_ascii=False))

    with (args.save_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()