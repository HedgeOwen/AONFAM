# Automatic Outlier Note Flagging & Correction Suggestion

A PyTorch baseline for **MIDI note-table error detection and correction**. The model predicts an action for each note (`KEEP/REPLACE/DELETE`) and, for `REPLACE`, a suggested pitch.

## Highlights

- BiGRU encoder with dual heads (action classification + pitch suggestion)
- Supports CSV/Parquet note-table inputs
- End-to-end training, inference, and single-MIDI diagnosis

## Key scripts

- `train.py`: trains action classification (`KEEP/REPLACE/DELETE`) + replacement pitch prediction
- `infer.py`: runs checkpoint inference on one file or a directory and writes tables with prediction columns
- `note_model.py`: dataset, model, and metric utilities
- `midi_diagnose.py`: diagnose one MIDI file and print note-level error suggestions
- `midi_to_table_train.py`: convert MAESTRO MIDI files to clean note tables
- `prepare_clean_only.py`: label clean-only tables and optionally merge into a labeled root

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

## Data balance note (project setting)

This project assumes a **1:1 ratio of error vs clean samples** in the labeled dataset.
If you construct errors synthetically, keep the error/clean count balanced at 1:1.
During training, enable `--use_weighted_sampler` to mitigate any residual class imbalance among `KEEP/REPLACE/DELETE`.

## Training

```bash
python train.py \
  --data_root /path/to/data \
  --format csv \
  --window_k 16 \
  --batch_size 128 \
  --epochs 10 \
  --lr 1e-3 \
  --lambda_pitch 1.0 \
  --save_dir runs/baseline \
  --device cpu \
  --use_weighted_sampler
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
- `pred_error_prob`
- `p_keep`, `p_replace`, `p_delete`

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

### Windows path replacement guide (for `midi_diagnose.py`)

`midi_diagnose.py` already includes **MIDI -> note-table (in-memory)** conversion via `pretty_midi`, so for single-MIDI diagnosis you do **not** need to manually run `midi_to_table_train.py` first.

Example PowerShell one-line (replace with your paths):

```powershell
python .\midi_diagnose.py --checkpoint "C:\\project\\runs\\full_v2_tune1\\best.pt" --midi "C:\\project\\demo_midis\\example.mid" --device cuda --threshold 0.96 --topk 5 --max_print 100 --out_path "C:\\project\\pred_one_song.csv" --out_format csv
```

## Notes

- For parquet mode, ensure the environment includes `pyarrow`.
- Baseline encoder is BiGRU with center-token heads for action/pitch.

## Build clean note tables from MAESTRO MIDI

If you only have MAESTRO MIDI + metadata CSV, first convert MIDI to note tables:

```bash
python midi_to_table_train.py \
  --maestro_root /path/to/maestro-v3.0.0 \
  --metadata_csv /path/to/maestro-v3.0.0.csv \
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

## Local quick start

See `LOCAL_RUN.md` for a complete step-by-step local setup and run guide (venv, dependency install, training, and inference commands).
