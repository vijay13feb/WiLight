# AxSense

WiFi CSI human activity recognition, end to end: raw CSI to Doppler spectrograms, model training, device-aware pruning, and few-shot fine-tuning on the target device.

The model is **RepDopplerViT**, a RepViT/EfficientViT hybrid trained with late fusion across antennas. It is pruned specifically for the deployment device's hardware profile, then adapted with a handful of labeled windows collected on that device.

## Pipeline

```
raw CSI (.npz)
   │  batchprocessing.py
   ▼
Doppler spectrograms (.npz)
   │  creating_train.py
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

`psutil` and `tqdm` are used for resource reporting and progress bars; the scripts still run without them.

## 1. Preprocessing — `batchprocessing.py`

Converts raw CSI captures into ratio and Doppler spectrogram `.npz` files. No CLI; edit the paths at the top of the script first:

```python
BASE_PATH   = SCRIPT_DIR / 'preprocessing'   # must contain S1/ and S2/ subfolders of raw .npz
OUTPUT_BASE = SCRIPT_DIR / 'output'
TEST_MODE   = False                          # True processes a single file, for a smoke test
```

```
python3 batchprocessing.py
```

**Output**
| Path | Contents |
|---|---|
| `output/double_ratio/*.npz` | intermediate ratio-combined CSI |
| `output/doppler/*.npz` | Doppler spectrograms, key `doppler` |
| `output/csv/S1_timing.csv`, `S2_timing.csv` | per-stage timing |

Input filenames must follow `signal_{scenario}_{activity}_{antenna}.npz` (e.g. `signal_S1a_A_0.npz`) — every downstream script relies on this convention.

## 2. Dataset split — `creating_train.py`

```
python3 creating_train.py <scenario> <activities> <window_length> <stride_length> [options]

python3 creating_train.py S1a A,C,D,F,G 100 10
python3 creating_train.py S1a all 100 10        # all activities present in the data
```

| Option | Default | Description |
|---|---|---|
| `--input_dir` | `output/doppler` | source of `signal_*.npz` files |
| `--output_dir` | `output/doppler` | scenario split written under `<output_dir>/<scenario>/` |
| `--val_ratio` | `0.15` | |
| `--test_ratio` | `0.15` | |
| `--seed` | `42` | |

**Output** — `output/doppler/<scenario>/`: `train/`, `val/`, `test/` (one `.pkl` window per file), `label_dict.pkl`, `n_antennas.pkl`, `files_{split}.pkl`, `labels_{split}.pkl`.

## 3. Training — `train_doppler_vit.py`

```
python3 train_doppler_vit.py <dataset_dir> [options]

python3 train_doppler_vit.py output/doppler/S1a --epochs 100 --augment
```

`dataset_dir` is the scenario folder produced in step 2.

| Option | Default | Description |
|---|---|---|
| `--activities` | from `label_dict.pkl` | comma-separated letters |
| `--epochs` | `100` | |
| `--batch_size` | `64` | |
| `--lr` | `5e-4` | |
| `--supcon_weight` | `0.1` | supervised contrastive loss weight |
| `--patience` | `15` | early stopping |
| `--augment` | off | training-time augmentation |
| `--output_dir` | `dataset_dir` | |

**Output** (in `dataset_dir` unless `--output_dir` given): `best_model.pth`, `best_model_fused.pth` (reparameterized for inference), `training_curves.png`, `confusion_matrix.png`, `test_results.txt`.

## 4. Device profile — `extract_device_profile.py`

**Required.** Pruning is targeted to real hardware, not an estimate — run this on the actual deployment device (Raspberry Pi, Jetson, etc.) before pruning:

```
python3 extract_device_profile.py my_device.json
```

Copy `my_device.json` back to the training machine; it feeds into step 5 via `--device_profile`.

## 5. Pruning — `sparse_finetune.py`

```
python3 sparse_finetune.py <dataset_dir> --device_profile my_device.json [options]

python3 sparse_finetune.py output/doppler/S1a --device_profile my_device.json
```

`dataset_dir` is the same S1a folder from step 3 — it needs `best_model.pth`, `label_dict.pkl`, and `n_antennas.pkl` there. Runs Phase 1 (L1 sparsity retraining) and Phase 2 (accuracy-constrained channel pruning), selecting the smallest model that stays within the accuracy budget for the profiled device.

| Option | Default | Description |
|---|---|---|
| `--device_profile` | — | JSON from step 4 (recommended) |
| `--target_device` | — | built-in name instead of a profile (`pi5_8gb`, `pi4_4gb`, `pi3`, `pizero2`, `jetson_nano`, `pc`, `auto`, ...) — a rough estimate only |
| `--max_acc_drop` | `0.06` | stop once accuracy drops more than this fraction |
| `--retrain_epochs` | `15` | fine-tune epochs after each pruning step |
| `--sweep_ratios` | built-in list | comma-separated keep_ratios to search |
| `--benchmark` | off | also measure real latency on this machine |

**Output** — `output/doppler/S1a/sparse_fewshot/<hostname>/`: `pruned_model.pth`, `sweep_results.json`, `pareto_curve.png`, `run_log.txt`.

## 6. Few-shot fine-tuning — `device_finetune.py`

```
python3 device_finetune.py <pruned_model.pth> --doppler_dir <dir> --scenario <name> [options]

python3 device_finetune.py output/doppler/S1a/sparse_fewshot/mypi/pruned_model.pth \
    --doppler_dir output/doppler --scenario S2a --activities A,C,D,F,G \
    --k_shots 5,10,20,30,40,50
```

`--doppler_dir` is the flat folder of raw `.npz` files from step 1 (not the split folder from step 2) — this script globs `signal_{scenario}_{act}_*.npz` directly and builds windows itself.

| Option | Default | Description |
|---|---|---|
| `--activities` | from checkpoint | |
| `--k_shots` | `5,10,20,30,40,50` | labeled windows per class to test |
| `--n_trials` | `3` | random seeds per K |
| `--finetune_epochs` | `60` | |
| `--unfreeze_k` | `5` | K at which stage3 also becomes trainable |
| `--max_windows` | all | cap inference windows per activity, for faster runs |
| `--inference_model` | — | skip fine-tuning, evaluate a saved `finetuned_model_K*.pth` |

**Output** (next to the pruned model, or `--output_dir`): `finetuned_model_K5.pth`, `finetuned_model_K10.pth`, ... (one per K), `fewshot_results.txt`, `cm_zeroshot.png`, `cm_fewshot_K*.png`.

## Output layout after a full run

```
Python_code/
├── preprocessing/{S1,S2}/*.npz            raw input
└── output/
    ├── double_ratio/*.npz
    ├── doppler/*.npz                      flat, all scenarios
    ├── doppler/S1a/                       step 2
    │   ├── train/ val/ test/
    │   ├── best_model.pth                 step 3
    │   ├── best_model_fused.pth
    │   └── sparse_fewshot/<hostname>/
    │       ├── pruned_model.pth           step 5
    │       ├── finetuned_model_K*.pth     step 6
    │       └── fewshot_results.txt
    └── csv/S1_timing.csv
```

## Notes

- Scenario names (`S1a`, `S2a`, ...) are arbitrary but must stay consistent across steps 1–6 and match the `signal_{scenario}_*.npz` filenames.
- Steps 3 and 5 read and write the same `dataset_dir` by default — only pass a different `--output_dir` if you intend to separate them.
- `train_doppler_vit.py` is a shared module, not just an entry point: `sparse_finetune.py` and `device_finetune.py` both import the model class and training helpers from it directly.
