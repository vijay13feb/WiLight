# SimWiSense

WiFi CSI human activity recognition, end to end: raw CSI to Doppler spectrograms, model training, device-aware pruning, and few-shot fine-tuning on the target device.

The model is **RepDopplerViT** (RepViT/EfficientViT hybrid). This dataset is single-antenna (`n_antennas=1`), so there is no antenna fusion step. Two scenarios are processed throughout: `save_classroom` and `save_office`.

## Pipeline

```
raw CSI (.npy, one file per activity)
   │  batch_processing_simwisense.py
   ▼
Doppler spectrograms (.npz)
   │  create_train_simwisense.py
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

## 1. Preprocessing — `batch_processing_simwisense.py`

No CLI; edit the paths at the top of the script first:

```python
BASE_PATH   = SCRIPT_DIR / 'input_data'    # must contain save_classroom/ and save_office/
OUTPUT_BASE = SCRIPT_DIR / 'output_data'
```

```
python3 batch_processing_simwisense.py
```

Each scenario folder holds one `.npy` file per activity letter (e.g. `A.npy` = Push_forward).

**Output**
| Path | Contents |
|---|---|
| `output_data/{scenario}/double_ratio/{act}.npz` | intermediate ratio-combined CSI |
| `output_data/{scenario}/doppler/{act}.npz` | Doppler spectrograms, key `doppler` |
| `output_data/csv/save_classroom_timing.csv`, `save_office_timing.csv` | per-stage timing |

## 2. Dataset split — `create_train_simwisense.py`

```
python3 create_train_simwisense.py <activities> <window_length> <stride_length> [options]

python3 create_train_simwisense.py all 100 10
```

| Option | Default | Description |
|---|---|---|
| `--scenarios` | `all` | comma-separated (`save_classroom,save_office`) or `all` |
| `--temporal_stride` | `10` | subsamples the 500 Hz Doppler frames before windowing (10 → 50 Hz, so a 100-frame window covers 2 s) |
| `--val_ratio` | `0.15` | |
| `--test_ratio` | `0.15` | |
| `--seed` | `42` | |

**Output** — `output_data/doppler_train/{scenario}/`: `train/`, `val/`, `test/` (one `.pkl` window per file), `label_dict.pkl`, `n_antennas.pkl` (always 1), `files_{split}.pkl`, `labels_{split}.pkl`.

## 3. Training — `train_doppler_vit.py`

No dataset path argument — it processes both scenarios automatically from a fixed input/output layout.

```
python3 train_doppler_vit.py [options]

python3 train_doppler_vit.py --epochs 100 --augment
```

| Option | Default | Description |
|---|---|---|
| `--scenarios` | `all` | comma-separated or `all` |
| `--activities` | all in `label_dict.pkl` | comma-separated letters |
| `--epochs` | `100` | |
| `--batch_size` | `64` | |
| `--lr` | `5e-4` | |
| `--supcon_weight` | `0.1` | |
| `--patience` | `15` | early stopping |
| `--augment` | off | training-time augmentation |

**Output** — `output_data/doppler_test/{scenario}/`: `best_model.pth`, `best_model_fused.pth` (reparameterized), `training_curves.png`, `confusion_matrix_group{1-4}_*.png` (4 groups of 5 activities: A-E, F-J, K-O, P-T), `test_results.txt`.

Note the split: the dataset lives under `doppler_train/` (step 2) but the trained checkpoint is written to `doppler_test/` — step 5 needs both paths.

## 4. Device profile

**Required.** Pruning is targeted to real hardware, not an estimate. Run this on the actual deployment device (Raspberry Pi, Jetson, etc.), using `sparse_finetune.py` itself:

```
python3 sparse_finetune.py --export_device_profile my_device.json
```

Copy `my_device.json` back to the training machine; it feeds into step 5 via `--device_profile`.

## 5. Pruning — `sparse_finetune.py`

```
python3 sparse_finetune.py <dataset_dir> --model_dir <model_dir> --device_profile my_device.json [options]

python3 sparse_finetune.py output_data/doppler_train/save_classroom \
    --model_dir output_data/doppler_test/save_classroom \
    --device_profile my_device.json
```

`dataset_dir` is the scenario's split folder from step 2 (needs `label_dict.pkl`, `n_antennas.pkl`, and the `test/` split there). `--model_dir` is the scenario's checkpoint folder from step 3 (needs `best_model.pth` there) — pass it explicitly since it differs from `dataset_dir` in this pipeline. Runs Phase 1 (L1 sparsity retraining) and Phase 2 (accuracy-constrained channel pruning), selecting the smallest model that stays within the accuracy budget for the profiled device.

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
python3 device_finetune.py <pruned_model.pth> --doppler_dir <dir> --scenario <name> \
    --window_frames 100 --window_stride 10 --temporal_stride 10 [options]

python3 device_finetune.py output_data/doppler_train/save_classroom/sparse_fewshot/mypi/pruned_model.pth \
    --doppler_dir output_data/save_office/doppler --scenario save_office \
    --window_frames 100 --window_stride 10 --temporal_stride 10 \
    --activities A,C,D,F,G --k_shots 5,10,20,30,40,50
```

`--doppler_dir` is the flat per-activity `.npz` folder from step 1 (`output_data/{scenario}/doppler`), typically the *other* scenario than the one the model was trained/pruned on. `--window_frames`, `--window_stride`, and `--temporal_stride` must match the values used in step 2 (`100`, `10`, `10`) — the script's own defaults (`--window_frames 500`) do **not** match and will produce mismatched window sizes if left unset.

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
├── input_data/{save_classroom,save_office}/*.npy   raw input
└── output_data/
    ├── {save_classroom,save_office}/
    │   ├── double_ratio/*.npz
    │   └── doppler/*.npz
    ├── doppler_train/{scenario}/                   step 2
    │   ├── train/ val/ test/
    │   └── sparse_fewshot/<hostname>/
    │       ├── pruned_model.pth                    step 5
    │       ├── finetuned_model_K*.pth               step 6
    │       └── fewshot_results.txt
    ├── doppler_test/{scenario}/                     step 3
    │   ├── best_model.pth
    │   └── best_model_fused.pth
    └── csv/save_classroom_timing.csv
```
