# Local Run Guide (PyTorch)

This is the shortest path to run the project locally.

## 1) Enter the project directory

```bash
cd /path/to/desktop-tutorial
```

## 2) Create and activate a virtual environment

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

> Dependencies include `torch`, `numpy`, `pandas`, and `pyarrow`.

## 4) Prepare the data directory

Required structure:

```text
DATA_ROOT/
  train/*.csv or *.parquet
  validation/*.csv or *.parquet
  test/*.csv or *.parquet
```

Column requirements follow the project schema (`piece_id/note_id/.../label/correct_pitch`).

Data construction rule: **error samples and clean samples are 1:1**. During training, enable `--use_weighted_sampler` to mitigate residual imbalance among `KEEP/REPLACE/DELETE`.

## 4.5) If you only have MAESTRO MIDI (no clean note tables)

Convert MAESTRO MIDI files into clean note tables first:

```bash
python midi_to_table_train.py \
  --maestro_root /absolute/path/to/maestro-v3.0.0 \
  --metadata_csv /absolute/path/to/maestro-v3.0.0.csv \
  --output_root /absolute/path/to/clean_note_tables \
  --format parquet
```

Then run `prepare_clean_only.py` to add labels and (optionally) merge into your training root.

## 5) Training (start with CPU to verify the pipeline)

```bash
python train.py \
  --data_root /absolute/path/to/DATA_ROOT \
  --format csv \
  --window_k 16 \
  --batch_size 64 \
  --epochs 2 \
  --lr 1e-3 \
  --lambda_pitch 1.0 \
  --save_dir runs/baseline \
  --device cpu \
  --max_files_per_split 3 \
  --use_weighted_sampler \
  --skip_test_eval
```

> Note: `--skip_test_eval` is useful for a quick smoke run. Remove it for full evaluation.

> Resume training: add `--resume_checkpoint runs/baseline/last.pt` to continue from the next epoch.

Outputs:
- `runs/baseline/best.pt`
- `runs/baseline/last.pt`
- `runs/baseline/train_log.json`

## 6) Inference (single file)

```bash
python infer.py \
  --checkpoint runs/baseline/best.pt \
  --input_path /absolute/path/to/DATA_ROOT/test/xxx_labeled.csv \
  --output_dir pred_out \
  --format csv \
  --topk 5 \
  --device cpu
```

## 7) Inference (directory)

```bash
python infer.py \
  --checkpoint runs/baseline/best.pt \
  --input_path /absolute/path/to/DATA_ROOT/test \
  --output_dir pred_out_test \
  --format csv \
  --topk 5 \
  --device cpu
```

If the directory contains nested subfolders:

```bash
python infer.py \
  --checkpoint runs/baseline/best.pt \
  --input_path /absolute/path/to/DATA_ROOT \
  --output_dir pred_out_all \
  --format csv \
  --recursive \
  --device cpu
```

To reduce false positives, apply threshold post-processing:

```bash
python infer.py \
  --checkpoint runs/baseline/best.pt \
  --input_path /absolute/path/to/DATA_ROOT/test \
  --output_dir pred_out_test_thr85 \
  --format parquet \
  --device cpu \
  --error_threshold 0.85
```

## Common issues

- `ModuleNotFoundError: torch/pandas/numpy`: dependencies are not installed in the current environment. Go back to steps 2–3.
- `No training files found`: check files under `--data_root/train` and file extensions.
- `Missing required columns`: input tables are missing required columns.
- `train_loss` or `val_loss` becomes `NaN`: a batch has no `label==1` (replace samples). The code now treats pitch loss as 0 for that batch.
- GPU training: change `--device cpu` to `--device cuda` (requires CUDA-compatible PyTorch).
- Training interrupted near the end: add `--skip_test_eval` or reduce data size with `--max_files_per_split`, then run full training.

## 8) Single MIDI interactive diagnosis (threshold=0.96)

```bash
python midi_diagnose.py \
  --checkpoint runs/full_v2_tune1/best.pt \
  --midi /path/to/one.mid \
  --device cuda \
  --threshold 0.96 \
  --out_path pred_one.csv \
  --out_format csv
```

This prints predicted error notes and replacement suggestions.

Windows one-line example (replace with your own paths):

```powershell
python .\midi_diagnose.py --checkpoint "C:\\project\\runs\\full_v2_tune1\\best.pt" --midi "C:\\project\\demo_midis\\example.mid" --device cuda --threshold 0.96 --out_path "C:\\project\\pred_one.csv" --out_format csv
```

Note: `midi_diagnose.py` already includes MIDI -> note-table conversion, so you do not need to convert to CSV/Parquet first.
