# HeadSense

WiFi CSI head-gesture recognition, end to end: raw CSI to Doppler spectrograms, model training, device-aware pruning, and few-shot fine-tuning on the target device.

The model is **RepDopplerViT** (RepViT/EfficientViT hybrid). This dataset is single-antenna (802.11n, single-link CSI), so there is no antenna fusion step. Data is split into three phases: `phase1` (100 Hz, training data), `phase2`/`phase3` (200 Hz, test scenarios).

## Pipeline

```
raw CSI (.txt, pickled DataFrames, per phase)
   │  batchprocessing_headgest.py
   ▼
Doppler spectrograms (.npz, one file per gesture)
   │  create_train_headgest.py
   ▼
train / val / test split (.pkl)
   │  train_doppler_vit.py
   ▼
best_model.pth
   │  sparse_finetune.py   (needs a device profile — see step 4)
   ▼
pruned_model.pth
   │  device_finetune.py
   ▼
finetuned_model_K*.pth
```

Run every command from `Python_code/`.

## Requirements

```
pip install torch torchvision numpy scipy pandas scikit-learn matplotlib seaborn psutil tqdm
```

## 1. Preprocessing — `batchprocessing_headgest.py`

No CLI; edit the paths at the top of the script first:

```python
INPUT_BASE = SCRIPT_DIR / 'input_files'   # must contain phase1/, phase2/, phase3/
OUTPUT_DIR = SCRIPT_DIR / 'output'
```

```
python3 batchprocessing_headgest.py
```

Each phase folder holds one `.txt` (pickled DataFrame) per gesture, named by the gesture (e.g. `Forward.txt`). Phase 1 is sampled at 100 Hz, phases 2/3 at 200 Hz.

**Output**
| Path | Contents |
|---|---|
| `output/double_ratio/{phase}/signal_{phase}_{code}_ant0.npz` | intermediate ratio-combined CSI |
| `output/doppler/{phase}/signal_{phase}_{code}_ant0.npz` | Doppler spectrograms, key `doppler` |
| `output/csv/processing_timing.csv` | per-stage timing |

Gesture codes: `FW`=Forward, `LD`=Looking Down, `LL`=Looking Left, `LR`=Looking Right, `LU`=Looking Up, `ND`=Nodding, `SH`=Shaking.

## 2. Dataset split — `create_train_headgest.py`

```
python3 create_train_headgest.py <phase> <gestures> <window_length> <stride_length> [options]

python3 create_train_headgest.py phase1 all 250 10
```

Use `window_length 250` — that's what the rest of the pipeline (`train_doppler_vit.py`, `sparse_finetune.py`, `device_finetune.py`) defaults to and expects; a different value must be passed explicitly to every later step too.

| Option | Default | Description |
|---|---|---|
| `--input_dir` | `output/doppler` | source of `signal_*.npz` files (phase subfolders) |
| `--output_dir` | `output/doppler_train` | scenario split written under `<output_dir>/<phase>/` |
| `--val_ratio` | `0.15` | |
| `--test_ratio` | `0.15` | |
| `--seed` | `42` | |

**Output** — `output/doppler_train/{phase}/`: `train/`, `val/`, `test/` (one `.pkl` window per file), `label_dict.pkl`, `n_antennas.pkl` (always 1), `files_{split}.pkl`, `labels_{split}.pkl`.

## 3. Training — `train_doppler_vit.py`

```
python3 train_doppler_vit.py <dataset_dir> [options]

python3 train_doppler_vit.py output/doppler_train/phase1 --epochs 100 --augment
```

| Option | Default | Description |
|---|---|---|
| `--activities` | all in `label_dict.pkl` | comma-separated gesture codes |
| `--epochs` | `100` | |
| `--batch_size` | `64` | |
| `--lr` | `5e-4` | |
| `--supcon_weight` | `0.1` | |
| `--patience` | `15` | early stopping |
| `--augment` | off | training-time augmentation |
| `--output_dir` | `dataset_dir` | |

**Output** (in `dataset_dir` unless `--output_dir` given): `best_model.pth`, `best_model_fused.pth` (reparameterized), `training_curves.png`, `confusion_matrix.png`, `test_results.txt`.

## 4. Device profile

**Required.** Pruning is targeted to real hardware, not an estimate. Run this on the actual deployment device (Raspberry Pi, Jetson, etc.), using `sparse_finetune.py` itself:

```
python3 sparse_finetune.py --export_device_profile my_device.json
```

Copy `my_device.json` back to the training machine; it feeds into step 5 via `--device_profile`.

## 5. Pruning — `sparse_finetune.py`

```
python3 sparse_finetune.py <dataset_dir> --device_profile my_device.json [options]

python3 sparse_finetune.py output/doppler_train/phase1 --device_profile my_device.json
```

`dataset_dir` is the same `phase1` split folder from step 3 — it needs `best_model.pth`, `label_dict.pkl`, and `n_antennas.pkl` there (`best_model.pth` lives in the same folder by default; pass `--model_dir` only if you used a different `--output_dir` in step 3). Runs Phase 1 (L1 sparsity retraining) and Phase 2 (accuracy-constrained channel pruning), selecting the smallest model that stays within the accuracy budget for the profiled device.

| Option | Default | Description |
|---|---|---|
| `--device_profile` | — | JSON from step 4 (recommended) |
| `--target_device` | — | built-in name instead of a profile (`pi5_8gb`, `pi4_4gb`, `pi3`, `pizero2`, `jetson_nano`, `pc`, `auto`, ...) — a rough estimate only |
| `--max_acc_drop` | `0.06` | stop once accuracy drops more than this fraction |
| `--sparsity_epochs` | `30` | Phase 1 epochs |
| `--l1_lambda` | `5e-4` | Phase 1 L1 penalty weight |
| `--retrain_epochs` | `20` | fine-tune epochs after each pruning step |
| `--benchmark` | off | also measure real latency on this machine |

**Output** — `<dataset_dir>/sparse_fewshot/<hostname>/`: `pruned_model.pth`, `sweep_results.json`, `pareto_curve.png`, `run_log.txt`.

## 6. Few-shot fine-tuning — `device_finetune.py`

```
python3 device_finetune.py <pruned_model.pth> --doppler_dir <dir> --scenario <phase> [options]

python3 device_finetune.py output/doppler_train/phase1/sparse_fewshot/mypi/pruned_model.pth \
    --doppler_dir output/doppler/phase2 --scenario phase2 \
    --activities FW,LD,LL,LR,LU,ND,SH --k_shots 5,10,20,30,40,50
```

`--doppler_dir` is the flat per-gesture `.npz` folder from step 1 for the target phase (`output/doppler/{phase}`), typically `phase2` or `phase3` — the pruned model was trained/pruned on `phase1`. `--window_frames` (default `250`) and `--window_stride` (default `10`) must match step 2's values.

| Option | Default | Description |
|---|---|---|
| `--activities` | from checkpoint | |
| `--k_shots` | `5,10,20,30,40,50` | labeled windows per class to test |
| `--n_trials` | `3` | random seeds per K |
| `--finetune_epochs` | `60` | |
| `--unfreeze_k` | `5` | K at which stage3 also becomes trainable |
| `--max_windows_per_class` | all | cap windows loaded per class, to limit RAM on small devices |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `--inference_model` | — | skip fine-tuning, evaluate a saved `finetuned_model_K*.pth` |

**Output** (next to the pruned model, or `--output_dir`): `finetuned_model_K5.pth`, `finetuned_model_K10.pth`, ... (one per K), `fewshot_results.txt`, `cm_zeroshot.png`, `cm_fewshot_K*.png`.

## Output layout after a full run

```
Python_code/
├── input_files/{phase1,phase2,phase3}/*.txt   raw input
└── output/
    ├── double_ratio/{phase}/*.npz
    ├── doppler/{phase}/*.npz
    ├── doppler_train/phase1/                  step 2
    │   ├── train/ val/ test/
    │   ├── best_model.pth                     step 3
    │   ├── best_model_fused.pth
    │   └── sparse_fewshot/<hostname>/
    │       ├── pruned_model.pth               step 5
    │       ├── finetuned_model_K*.pth          step 6
    │       └── fewshot_results.txt
    └── csv/processing_timing.csv
```
