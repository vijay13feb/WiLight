# Sharpac

WiFi CSI human activity recognition, end to end: raw CSI to Doppler spectrograms, model training, device-aware pruning, and few-shot fine-tuning on the target device.

The model is **RepDopplerViT**, a RepViT/EfficientViT hybrid trained with late fusion across antennas. It is pruned specifically for the deployment device's hardware profile, then adapted with a handful of labeled windows collected on that device.

## Pipeline

```
raw CSI (.txt, pickled)
   │  batchprocessing_sharpac.py
   ▼
Doppler spectrograms (.npz, one file per antenna)
   │  create_train_sharp.py
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

`psutil` and `tqdm` are used for resource reporting and progress bars; the scripts still run without them. This pipeline additionally runs on PyTorch 1.8+ (Jetson Nano JetPack 4) — `device_finetune.py` falls back to compatible APIs automatically when newer ones aren't available.

## 1. Preprocessing — `batchprocessing_sharpac.py`

Converts raw pickled CSI captures into ratio and Doppler spectrogram `.npz` files, one output file per antenna. No CLI; edit the constants at the top of the script first:

```python
DATA_DIR   = SCRIPT_DIR / 'input_data'   # *.txt pickled CSI files
OUTPUT_DIR = SCRIPT_DIR / 'output'
```

```
python3 batchprocessing_sharpac.py
```

Each input file `<name>.txt` (shape `subcarriers x packets x antennas`) is split into `NUM_ANTENNAS` (4) separate recordings, bad subcarriers removed and interpolated up to `TARGET_N_SUB`, then run through the same ratio/temporal-clean/Doppler pipeline as the rest of the project.

**Output**
| Path | Contents |
|---|---|
| `output/double_ratio/<name>_ant{0-3}.npz` | intermediate ratio-combined CSI |
| `output/doppler/<name>_ant{0-3}.npz` | Doppler spectrograms, key `doppler` |
| `output/csv/processing_timing.csv` | per-stage timing |

Input filenames must follow `signal_{scenario}_{activity}.txt` (e.g. `signal_S1a_H1.txt`) so the antenna-suffixed output — `signal_S1a_H1_ant0.npz` — matches what step 2 expects.

## 2. Dataset split — `create_train_sharp.py`

```
python3 create_train_sharp.py <scenario> <activities> <window_length> <stride_length> [options]

python3 create_train_sharp.py S1a E,W,R,J,S,L,H,C 100 10
python3 create_train_sharp.py S1a all 100 10        # all activities present in the data
```

| Option | Default | Description |
|---|---|---|
| `--input_dir` | `output/doppler` | source of `signal_*.npz` files |
| `--output_dir` | `output/doppler_train` | scenario split written under `<output_dir>/<scenario>/` |
| `--val_ratio` | `0.15` | |
| `--test_ratio` | `0.15` | |
| `--seed` | `42` | |

Activity variants with a numeric suffix (`H1`, `H2`, `J1`, `J2`, ...) are treated as **separate recordings** of the same normalized class (`H`, `J`) — all of them contribute windows to that class rather than overwriting each other. Multi-antenna files for one recording are aligned by trimming to the shortest antenna's frame count.

**Output** — `output/doppler_train/<scenario>/`: `train/`, `val/`, `test/` (one `.pkl` window per file), `label_dict.pkl`, `n_antennas.pkl`, `files_{split}.pkl`, `labels_{split}.pkl`.

## 3. Training — `train_doppler_vit.py`

```
python3 train_doppler_vit.py <dataset_dir> [options]

python3 train_doppler_vit.py output/doppler_train/S1a --epochs 100 --augment
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

python3 sparse_finetune.py output/doppler_train/S1a --device_profile my_device.json
```

`dataset_dir` is the same S1a folder from step 3 — it needs `best_model.pth`, `label_dict.pkl`, and `n_antennas.pkl` there. Runs Phase 1 (L1 sparsity retraining) and Phase 2 (accuracy-constrained channel pruning), selecting the smallest model that stays within the accuracy budget for the profiled device.

| Option | Default | Description |
|---|---|---|
| `--device_profile` | — | JSON from step 4 (recommended) |
| `--target_device` | — | built-in name instead of a profile (`pi5_8gb`, `pi4_4gb`, `pi3`, `pizero2`, `jetson_nano`, `pc`, `auto`, ...) — a rough estimate only |
| `--max_acc_drop` | `0.06` | stop once accuracy drops more than this fraction |
| `--retrain_epochs` | `10` | fine-tune epochs after each pruning step |
| `--sweep_ratios` | built-in list | comma-separated keep_ratios to search |
| `--benchmark` | off | also measure real latency on this machine |

**Output** — `output/doppler_train/S1a/sparse_fewshot/<hostname>/`: `pruned_model.pth`, `sweep_results.json`, `pareto_curve.png`, `run_log.txt`.

## 6. Few-shot fine-tuning — `device_finetune.py`

```
python3 device_finetune.py <pruned_model.pth> --doppler_dir <dir> --scenario <name> [options]

python3 device_finetune.py output/doppler_train/S1a/sparse_fewshot/mypi/pruned_model.pth \
    --doppler_dir output/doppler --scenario S2a --activities E,W,R,J,S,L,H,C \
    --k_shots 5,10,20,30,40,50
```

`--doppler_dir` is the flat folder of raw `.npz` files from step 1 (not the split folder from step 2) — this script globs `signal_{scenario}_{act}[N]_*.npz` directly (numbered variants like `H1`/`H2` are auto-concatenated) and builds windows itself.

| Option | Default | Description |
|---|---|---|
| `--activities` | from checkpoint | |
| `--k_shots` | `5,10,20,30,40,50` | labeled windows per class to test |
| `--n_trials` | `3` | random seeds per K |
| `--finetune_epochs` | `60` | |
| `--unfreeze_k` | `5` | K at which stage3 also becomes trainable |
| `--max_windows_per_class` | all | cap windows loaded per class, to limit RAM on small devices |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` — force a device |
| `--window_frames` | `360` | must match `window_length` from step 2 |
| `--window_stride` | `10` | must match `stride_length` from step 2 |
| `--inference_model` | — | skip fine-tuning, evaluate a saved `finetuned_model_K*.pth` |

If some activities in `--activities` have no data under `--doppler_dir`/`--scenario`, they're dropped automatically (with a warning) rather than failing — fine-tuning and evaluation proceed on whatever is present. On CPU, antenna forward passes run one-per-core via a thread pool when there are enough cores, which measures noticeably faster than a single batched call.

**Output** (next to the pruned model, or `--output_dir`): `finetuned_model_K5.pth`, `finetuned_model_K10.pth`, ... (one per K), `fewshot_results.txt`, `cm_zeroshot.png`, `cm_fewshot_K*.png`.

## Output layout after a full run

```
Python_code/
├── input_data/*.txt                       raw input
└── output/
    ├── double_ratio/*.npz
    ├── doppler/*.npz                      flat, all scenarios, one file per antenna
    └── doppler_train/S1a/                 step 2
        ├── train/ val/ test/
        ├── best_model.pth                 step 3
        ├── best_model_fused.pth
        └── sparse_fewshot/<hostname>/
            ├── pruned_model.pth           step 5
            ├── finetuned_model_K*.pth     step 6
            └── fewshot_results.txt
```
