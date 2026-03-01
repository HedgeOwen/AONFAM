# 本地运行指南（PyTorch）

下面是最短路径：

## 1) 进入项目目录

```bash
cd /path/to/desktop-tutorial
```

## 2) 创建并激活虚拟环境

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

## 3) 安装依赖

```bash
pip install -U pip
pip install -r requirements.txt
```

> 依赖包含 `torch`、`numpy`、`pandas`、`pyarrow`。

## 4) 准备数据目录

要求结构：

```text
DATA_ROOT/
  train/*.csv 或 *.parquet
  validation/*.csv 或 *.parquet
  test/*.csv 或 *.parquet
```

并且列名满足项目约定（`piece_id/note_id/.../label/correct_pitch`）。

数据构造约定：**错误样本与干净样本 1:1**。训练时建议开启 `--use_weighted_sampler` 以缓解 `KEEP/REPLACE/DELETE` 类别不均衡。

## 4.5) 若只有 MAESTRO MIDI（没有 clean note tables）

先把 MIDI 转成 clean note tables：

```bash
python midi_to_table.py \
  --maestro_root /absolute/path/to/maestro-v3.0.0 \
  --output_root /absolute/path/to/clean_note_tables \
  --format parquet
```

然后再执行 `prepare_clean_only.py` 补标签并合并到训练目录。

## 5) 训练（先用 CPU 跑通）

```bash
python train(2).py \
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

> 说明：`--skip_test_eval` 适合先做“能跑通”验证，避免在 CPU 上最后 test 评估耗时过长。正式跑结果时去掉该参数。

> 断点续跑：可加 `--resume_checkpoint runs/baseline/last.pt`，从上次保存轮次的下一轮继续训练。

训练后会生成：
- `runs/baseline/best.pt`
- `runs/baseline/last.pt`
- `runs/baseline/train_log.json`

## 6) 推理（单文件）

```bash
python infer.py \
  --checkpoint runs/baseline/best.pt \
  --input_path /absolute/path/to/DATA_ROOT/test/xxx_labeled.csv \
  --output_dir pred_out \
  --format csv \
  --topk 5 \
  --device cpu
```

## 7) 推理（目录）

```bash
python infer.py \
  --checkpoint runs/baseline/best.pt \
  --input_path /absolute/path/to/DATA_ROOT/test \
  --output_dir pred_out_test \
  --format csv \
  --topk 5 \
  --device cpu
```

若目录下有多层子目录：

```bash
python infer.py \
  --checkpoint runs/baseline/best.pt \
  --input_path /absolute/path/to/DATA_ROOT \
  --output_dir pred_out_all \
  --format csv \
  --recursive \
  --device cpu
```
若你想减少误报，可在推理时直接加阈值后处理（低置信度非 KEEP 动作强制回到 KEEP）：

```bash
python infer.py \
  --checkpoint runs/baseline/best.pt \
  --input_path /absolute/path/to/DATA_ROOT/test \
  --output_dir pred_out_test_thr85 \
  --format parquet \
  --device cpu \
  --error_threshold 0.85
```

也可以对现有预测结果做阈值扫描（无需重训）：

```bash
python sweep_error_threshold.py \
  --pred_root /absolute/path/to/pred_out_test \
  --format parquet \
  --start 0.5 \
  --stop 0.95 \
  --step 0.05
```

## 常见问题

- `ModuleNotFoundError: torch/pandas/numpy`：说明你没在当前 Python 环境安装依赖，回到第 2、3 步。
- `No training files found`：检查 `--data_root/train` 下是否有对应扩展名文件。
- `Missing required columns`：输入表缺列，按规范补齐。
- `train_loss` 或 `val_loss` 出现 `NaN`：常见原因是某个 batch 里没有 `label==1`（替换样本），旧版会导致 pitch loss 为 NaN；请拉取最新代码（已修复为该 batch 的 pitch loss 记为 0）。
- GPU 训练：将 `--device cpu` 改为 `--device cuda`（前提是本机 CUDA + 对应 torch 版本可用）。
- 训练结束前出现 `KeyboardInterrupt`（常见在最后 test 评估阶段）：先加 `--skip_test_eval` 或减小数据量（`--max_files_per_split`）验证流程，再做全量运行。


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

This prints where predicted errors are and gives replacement suggestions.

Windows 单行示例（仅需替换你自己的文件路径）：

```powershell
python .\midi_diagnose.py --checkpoint "E:\downloads\桌面\dku\CS309\project\code\runs\full_v2_tune1\best.pt" --midi "E:\downloads\桌面\dku\CS309\project\demo_midis\example.mid" --device cuda --threshold 0.96 --out_path "E:\downloads\桌面\dku\CS309\project\code\pred_one.csv" --out_format csv
```

说明：`midi_diagnose.py` 内部已包含 MIDI->note table 转换，不需要先手动转 CSV/Parquet。

如果输出是 `predicted_errors=0`，先加 `--print_raw_action_stats` 看阈值前后的类别分布；
若 MIDI 来源不是 MAESTRO（例如 MuseScore 导出），可再加 `--force_piano_features` 试一版。

