# Welcome to GitHub Desktop!
# Automatic Outlier Note Flagging & Correction Suggestion

This is your README. READMEs are where you can communicate what your project is and how to use it.
> ✅ 该项目**使用 PyTorch 训练**（`torch.nn` + `DataLoader` + `Adam`）。

Write your name on line 6, save it, and then head back to GitHub Desktop.
This repo provides a baseline multi-task training/inference pipeline for MIDI-note-table error detection and correction:

- `train(2).py`: trains action classification (`KEEP/REPLACE/DELETE`) + replacement pitch prediction.
- `infer.py`: runs checkpoint inference on one file or a directory and writes tables with prediction columns.
- `note_model(1).py`: shared dataset, model, and metric utilities.

## Data layout

```text
DATA_ROOT/
  train/*.csv|*.parquet
  validation/*.csv|*.parquet
  test/*.csv|*.parquet
```

Required columns:

- Metadata: `piece_id`, `note_id`, `inst_id`, `program`, `is_drum`
- Timing/note: `onset`, `offset`, `duration`, `pitch`, `velocity`
- Context index: `onset_bin`, `beat_id`, `bar_id`
- Labels (train/val/test only): `label`, `correct_pitch`

## Train

```bash
python train(2).py \
  --data_root /path/to/data \
  --format csv \
  --window_k 16 \
  --batch_size 128 \
  --epochs 10 \
  --lr 1e-3 \
  --lambda_pitch 1.0 \
  --save_dir runs/baseline \
  --device cpu
```

Optional:

- `--early_stop_patience 5` enables early stopping on validation macro-F1.
- `--skip_test_eval` skips final test metrics for quick smoke runs.
- `--max_files_per_split 3` limits file count per split for quick local debugging.
- `--eval_batch_size 512` controls validation/test batch size separately from training.
- `--use_weighted_sampler` upsamples REPLACE/DELETE labels during training to improve error recall when KEEP dominates.
- `--resume_checkpoint runs/exp/last.pt` resumes from a previous checkpoint and continues from the next epoch.

Outputs in `save_dir`:

- `best.pt`
- `last.pt`
- `train_log.json`

## Inference

```bash
python infer.py \
  --checkpoint runs/baseline/best.pt \
  --input_path /path/to/test_or_file \
  --output_dir pred_out \
  --format csv \
  --topk 5
```

Optional:

- `--recursive` recursively scans input directories and mirrors relative paths under `output_dir`.
- `--error_threshold 0.85` forces low-confidence non-KEEP predictions back to KEEP (useful to reduce false positives).

Output files keep original columns and append:

- `pred_action` (`KEEP/REPLACE/DELETE`)
- `pred_action_id` (`0/1/2`)
- `pred_action_prob`
- `pred_suggest_pitch`
- `pred_topk_pitches`
- `pred_topk_probs`

To sweep thresholds on existing predictions (without retraining), run:

```bash
python sweep_error_threshold.py \
  --pred_root /path/to/pred_out_test \
  --format parquet \
  --start 0.5 \
  --stop 0.95 \
  --step 0.05
```


## Diagnose a single MIDI (interactive-style)

If you want to pass one MIDI and get note-level error locations + correction suggestions directly:

```bash
python midi_diagnose.py \
  --checkpoint runs/baseline/best.pt \
  --midi /path/to/song.mid \
  --device cuda \
  --threshold 0.96 \
  --topk 5 \
  --max_print 100 \
  --out_path pred_one_song.csv \
  --out_format csv
```

- `--threshold 0.96` applies the same non-KEEP filtering strategy used in `infer.py`.
- Add `--print_raw_action_stats` to print action counts before/after threshold (useful when output is `predicted_errors=0`).
- Add `--force_piano_features` for out-of-domain MIDI exported from notation tools (forces `program=0, inst_id=0, is_drum=0` to match MAESTRO-like piano features).
- The command prints predicted error notes (`REPLACE/DELETE`) and suggestions to terminal, and can optionally save a full output table.

### Windows path replacement guide (for `midi_diagnose.py`)

`midi_diagnose.py` already includes **MIDI -> note-table (in-memory)** conversion via `pretty_midi`, so for single-MIDI diagnosis you do **not** need to manually run `midi_to_table.py` first.

In the command above, you only need to replace these path placeholders:

- `runs/baseline/best.pt` -> your trained checkpoint path (e.g. `E:\...\runs\full_v2_tune1\best.pt`)
- `/path/to/song.mid` -> the MIDI file to diagnose
- `pred_one_song.csv` -> where to save the diagnosis table (optional)

Concrete PowerShell one-line example:

```powershell
python .\midi_diagnose.py --checkpoint "E:\downloads\桌面\dku\CS309\project\code\runs\full_v2_tune1\best.pt" --midi "E:\downloads\桌面\dku\CS309\project\demo_midis\example.mid" --device cuda --threshold 0.96 --topk 5 --max_print 100 --out_path "E:\downloads\桌面\dku\CS309\project\code\pred_one_song.csv" --out_format csv
```

### Troubleshooting: `CUDA error ... srcIndex < srcSelectDimSize` in `midi_diagnose.py`

If you see a long CUDA indexing assert when diagnosing a single MIDI, it usually means the MIDI contains unseen or inconsistent categorical ids (for example, unusual track/program metadata) that exceed embedding ranges learned during training.

This repo now clamps `pitch/program/inst` ids to valid embedding ranges at runtime, so pull latest code and retry.
If your local copy is older, update first, or run once with `--device cpu` to get clearer error text.

## Notes

- For parquet mode, ensure environment includes `pyarrow`.
- Baseline encoder is BiGRU with center-token heads for action/pitch.


## Build clean note tables from MAESTRO MIDI

If you only have MAESTRO MIDI + metadata csv/json, first convert MIDI to note tables:

```bash
python midi_to_table.py \
  --maestro_root /path/to/maestro-v3.0.0 \
  --output_root /path/to/clean_note_tables \
  --format parquet
```

This creates `/train`, `/validation`, `/test` note-table files with required base columns.

## Clean-only data preparation

To add fully-correct (clean-only) samples into training data, run:

```bash
python prepare_clean_only.py \
  --clean_root /path/to/clean_note_tables \
  --output_root /path/to/clean_labeled_out \
  --merge_into /path/to/out_labeled \
  --format parquet
```

This script adds `label=0`, `correct_pitch=-1` (and fills optional `case/error_case_color`) then optionally merges files into your existing split folders.

## Data balance note (project setting)

This project assumes a **1:1 ratio of error vs clean samples** in the labeled dataset.
If you construct errors synthetically, keep the error/clean count balanced at 1:1.
During training, enable `--use_weighted_sampler` to mitigate any residual class imbalance among `KEEP/REPLACE/DELETE`.

## Local quick start

See `LOCAL_RUN.md` for a complete step-by-step local setup and run guide (venv, dependency install, training, and inference commands).
