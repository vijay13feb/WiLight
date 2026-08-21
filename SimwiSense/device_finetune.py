#!/usr/bin/env python3
"""
device_finetune.py  —  Device-Side Few-Shot Fine-Tuning (Phase 3)

Run on the target device after copying pruned_model.pth from the server.
Support is K windows/class (stratified); loss is Focal + CrossEntropy (+
SupCon once stage3 unfreezes); head is always trainable, stage3 unfreezes
for K >= unfreeze_k. Reports inference latency/MACs/RAM and fine-tuning
time/FLOPs/RAM, plus per-K accuracy, F1, and per-class recall.

Requirements:
    pip install torch torchvision scikit-learn matplotlib seaborn psutil

Usage:
    python3 device_finetune.py pruned_model.pth \\
        --doppler_dir output_data/save_classroom/doppler --scenario save_classroom \\
        --activities A,C,D,F,G --k_shots 5,10,20,30,40,50 --lr 5e-4 --batch_size 16
"""

import argparse
import copy
import math
import os
import platform
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import resource as _resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False   # Windows

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def _tqdm(iterable=None, **kwargs):   # no-op fallback if tqdm not installed
        return iterable if iterable is not None else range(kwargs.get('total', 0))

# Jetson Nano JetPack 4 ships torch 1.8, which lacks inference_mode (added 1.9).
_INFERENCE_MODE = getattr(torch, 'inference_mode', torch.no_grad)

# Radar / windowing constants; WINDOW_FRAMES/WINDOW_STRIDE are overridden at
# runtime via --window_frames/--window_stride.
SAMPLING_HZ    = 500
WINDOW_FRAMES  = 500
WINDOW_STRIDE  = 10

sys.path.insert(0, str(Path(__file__).parent))
from train_doppler_vit import (
    RepDopplerViT, FocalLoss, SupConLoss,
    count_params, set_seed,
    plot_confusion_matrix_group, ACTIVITY_MAP,
    _cross_entropy_smooth,
)


def _torch_load(path, map_location):
    """torch.load with weights_only=False where supported (kwarg needs ≥1.13)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def plot_confusion_matrix(cm, class_names, path):
    plot_confusion_matrix_group(
        cm_group=cm, class_names_group=class_names,
        group_label='all_classes', path=path)


def _infer_depth(state_dict: dict):
    depths = []
    for i in range(1, 4):
        idxs = [int(k.split('.')[1]) for k in state_dict if k.startswith(f'stage{i}.')]
        depths.append(max(idxs) + 1 if idxs else 2)
    return tuple(depths)


# System resource helpers

def _rss_mb() -> float:
    """Current process RSS in MB (0 if psutil unavailable)."""
    if HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    return 0.0


def _cpu_time_s() -> float:
    """Process CPU time (user+sys) in seconds (0 if resource unavailable)."""
    if HAS_RESOURCE:
        u = _resource.getrusage(_resource.RUSAGE_SELF)
        return u.ru_utime + u.ru_stime
    return 0.0


def _peak_rss_mb() -> float:
    """Peak RSS since process start in MB (Linux ru_maxrss is in KB)."""
    if HAS_RESOURCE:
        return _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024
    return _rss_mb()


# FLOPs / MACs counting  (no external library — portable to Pi 4)

@torch.no_grad()
def count_model_macs(model: nn.Module,
                     input_shape: Tuple[int, ...] = (1, 1, 100, 100)) -> int:
    """Count MACs for one forward pass via Conv2d / Linear forward hooks.
    1 MAC = 1 fused multiply-add = 2 FLOPs."""
    total = [0]

    def _conv_hook(m, inp, out):
        _, C_out, H, W = out.shape
        C_in = inp[0].shape[1]
        kH = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
        kW = m.kernel_size[1] if isinstance(m.kernel_size, tuple) else m.kernel_size
        total[0] += (C_in // m.groups) * kH * kW * C_out * H * W

    def _linear_hook(m, inp, out):
        total[0] += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))

    was_training = model.training
    model.eval()
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device('cpu')
    model(torch.zeros(*input_shape, device=dev))
    if was_training:
        model.train()
    for h in hooks:
        h.remove()
    return total[0]



# In-memory dataset

class MemDataset(Dataset):
    """expand=True yields one (1,T,V) per-antenna frame per window (fine-tuning);
    expand=False yields the full (n_ant,T,V) window (late-fusion eval)."""
    def __init__(self, windows: List[np.ndarray], labels: List[int],
                 augment: bool = False, expand: bool = True):
        self.windows = windows
        self.labels  = labels
        self.augment = augment
        self.expand  = expand
        self._n_ant  = windows[0].shape[0] if windows else 1
        # Zero-copy where possible: windows are float32 views into a shared
        # frame buffer (overlapping strided windows), so avoid materializing
        # every copy — that OOMs 2 GB devices (Jetson Nano / Pi Zero 2).
        self._tensors = [
            torch.from_numpy(w) if w.dtype == np.float32
            else torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
            for w in windows
        ]

    @property
    def n_antennas(self) -> int:
        return self._n_ant

    def __len__(self) -> int:
        return len(self.labels) * (self.n_antennas if self.expand else 1)

    def __getitem__(self, idx):
        if self.expand:
            fi, ai = divmod(idx, self.n_antennas)
        else:
            fi, ai = idx, None

        x = self._tensors[fi]
        if ai is not None:
            x = x[ai : ai + 1]   # slice before augment to avoid augmenting all antennas

        if self.augment:
            # torch.roll copies, so the shared/cached buffer is never mutated.
            x = torch.roll(x, random.randint(-8, 8), dims=1)
            v0 = random.randint(0, 85)
            x[:, :, v0 : v0 + random.randint(0, 15)] = 0.0
            t0 = random.randint(0, max(0, x.shape[1] - 8))
            x[:, t0 : t0 + random.randint(0, 8), :] = 0.0
            x = (x * random.uniform(0.85, 1.15) +
                 0.003 * torch.randn_like(x)).clamp(0, 1)

        return x, self.labels[fi]


# S2a Data Loading

def load_scenario_windows(doppler_dir: Path,
                          activities: List[str],
                          scenario: str = 'S2a',
                          window_length: int = 100,
                          stride_length: int  = 10,
                          temporal_stride: int = 10,
                          max_windows: Optional[int] = None,
                          ) -> Tuple[List[np.ndarray], List[int], Dict]:
    """Load Doppler windows for one activity per file (doppler_dir/{act}.npz).
    Raw Doppler is 500 Hz; temporal_stride subsamples before windowing (10 ->
    50 Hz, so a 100-frame window covers 2 s). max_windows caps frames loaded
    per class to limit RAM."""
    label_dict: Dict = {act: idx for idx, act in enumerate(activities)}
    windows: List[np.ndarray] = []
    labels:  List[int]        = []

    for act in activities:
        fpath = doppler_dir / f"{act}.npz"

        if not fpath.exists():
            print(f"  [warn] activity {act}: file not found")
            continue

        data = np.load(fpath)["doppler"].astype(np.float32)

        if temporal_stride > 1:
            data = data[::temporal_stride, :]

        min_f   = data.shape[0]
        avail_w = (min_f - window_length) // stride_length + 1
        if max_windows is not None and avail_w > max_windows:
            min_f = (max_windows - 1) * stride_length + window_length
            data  = data[:min_f]
        mc    = data[np.newaxis, ...]   # (1, n_frames, n_vel_bins)
        n_win = (min_f - window_length) // stride_length + 1

        for k in range(n_win):
            s = k * stride_length
            windows.append(mc[:, s:s+window_length, :])
            labels.append(label_dict[act])

        eff_hz   = 500 / temporal_stride
        cap_note = (f' (capped from {avail_w} available)'
                    if max_windows is not None and avail_w > max_windows else '')
        print(f'  [{act}]: {min_f} frames (sub×{temporal_stride}, {eff_hz:.0f}Hz) '
              f'→ {n_win} windows{cap_note}')
    return windows, labels, label_dict


def cap_windows_per_class(windows: List[np.ndarray], labels: List[int],
                          max_per_class: int) -> Tuple[List[np.ndarray], List[int]]:
    """Cap query/inference windows to max_per_class per class (evenly spaced).
    Only applied to the inference set — fine-tuning always uses the full split."""
    from collections import defaultdict
    class_indices: Dict = defaultdict(list)
    for i, lb in enumerate(labels):
        class_indices[lb].append(i)
    selected = []
    for lb in sorted(class_indices):
        idxs = class_indices[lb]
        if len(idxs) > max_per_class:
            sub = np.linspace(0, len(idxs) - 1, max_per_class, dtype=int)
            idxs = [idxs[j] for j in sub]
        selected.extend(idxs)
    selected.sort()
    return [windows[i] for i in selected], [labels[i] for i in selected]


def split_fewshot(windows, labels, k_shot: int, seed: int):
    """Stratified K-shot split → (sup_w, sup_l, q_w, q_l)."""
    rng = np.random.default_rng(seed)
    sup_w, sup_l, q_w, q_l = [], [], [], []
    for c in sorted(set(labels)):
        idx_c = [i for i, lb in enumerate(labels) if lb == c]
        rng.shuffle(idx_c)
        k = min(k_shot, len(idx_c))
        for i in idx_c[:k]:
            sup_w.append(windows[i]); sup_l.append(labels[i])
        for i in idx_c[k:]:
            q_w.append(windows[i]);   q_l.append(labels[i])
    return sup_w, sup_l, q_w, q_l


# Phase 3 — Few-Shot Fine-Tuning

def _mixup_batch(x: torch.Tensor, y: torch.Tensor,
                 alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup: interpolate two random samples. Returns (mixed_x, y_a, y_b, lam)."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def few_shot_finetune(model: RepDopplerViT,
                      support_ds: MemDataset,
                      device: torch.device,
                      n_antennas: int,
                      macs_per_ant: int,
                      lr: float             = 5e-4,
                      epochs: int           = 60,
                      unfreeze_stage3: bool = False,
                      batch_size: int       = 32,
                      gamma: float          = 2.0,
                      patience: int         = 10,
                      warmup_epochs: int    = 5,
                      mixup_alpha: float    = 0.2,
                      supcon_weight: float  = 0.1,
                      supcon_temp: float    = 0.07,
                      class_ids: Optional[List[int]] = None,
                      verbose: bool         = True,
                      ) -> Tuple[RepDopplerViT, Dict]:
    """Fine-tune pruned RepDopplerViT on K-shot support set.
    Loss: Focal + CE + SupCon (optional), warmup LR, early stopping, mixup.
    class_ids: model-output indices of the activities being fine-tuned, in
    the same order as the support labels 0..N-1 (None = all model classes).
    Returns (fine-tuned model, stats dict).
    """
    if not getattr(model, '_ft_reparameterized', False):
        model.reparameterize()
        model._ft_reparameterized = True

    # hybrid_stage3 models have down_pre + stage3_local + stage3_attn instead
    # of a single stage3 module — collect whichever exists.
    _hybrid = getattr(model, '_hybrid_stage3', False)
    if _hybrid:
        _s3_modules = [model.down_pre, model.stage3_local, model.stage3_attn]
    else:
        _s3_modules = [model.stage3]
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    if unfreeze_stage3:
        for mod in _s3_modules:
            for p in mod.parameters():
                p.requires_grad = True

    freeze_bn  = (not unfreeze_stage3 and getattr(model, '_simple_head', False))
    use_supcon = supcon_weight > 0.0 and unfreeze_stage3
    use_mixup  = mixup_alpha > 0.0

    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen     = sum(p.numel() for p in model.parameters()) - trainable
    train_frac = trainable / (trainable + frozen)

    if verbose:
        print(f'    Trainable: {trainable:,}  |  Frozen: {frozen:,}  '
              f'|  stage3_unfrozen={unfreeze_stage3}'
              + (f'  |  supcon w={supcon_weight} τ={supcon_temp}' if use_supcon else '')
              + (f'  |  mixup α={mixup_alpha}' if use_mixup else ''))

    n_expand      = len(support_ds)
    ft_flops_samp = int(2 * macs_per_ant * (1 + train_frac))
    ft_flops_ep   = n_expand * ft_flops_samp

    if verbose:
        print(f'    Fine-tune FLOPs/epoch : {ft_flops_ep/1e6:.1f} M  '
              f'({n_expand} expand samples × {ft_flops_samp/1e6:.1f} M/sample)')
        print(f'    Fine-tune FLOPs total : {ft_flops_ep * epochs/1e9:.2f} G  '
              f'({epochs} epochs)')

    # Head-only fast path: frozen backbone gives the same features every epoch,
    # so cache them once and train only the head on the cached vectors.
    _head_has_conv = any(isinstance(m, nn.Conv2d) for m in model.head.modules())
    use_feat_cache = not unfreeze_stage3 and not _head_has_conv
    if use_feat_cache:
        feat_list, lbl_list = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in DataLoader(support_ds, batch_size=batch_size,
                                      shuffle=False, num_workers=0):
                xb_d = xb.to(device)
                # Run backbone stages only (no head)
                f = model.stem(xb_d)
                f = model.down1(model.stage1(f))
                f = model.down2(model.stage2(f))
                if model._hybrid_stage3:
                    f = model.down_pre(f)
                    f = model.stage3_local(f)
                    f = model.stage3_attn(f)
                else:
                    f = model.stage3(f)
                # Apply only spatial head layers (GAP + Flatten) → flat [B, C] vector
                f = model.head[0](f)   # AdaptiveAvgPool2d(1)
                f = model.head[1](f)   # Flatten
                feat_list.append(f.cpu())
                lbl_list.append(yb)
        feats_all = torch.cat(feat_list)
        lbls_all  = torch.cat(lbl_list)
        feat_loader = DataLoader(
            torch.utils.data.TensorDataset(feats_all, lbls_all),
            batch_size=min(max(batch_size, 2), len(feats_all)), shuffle=True, num_workers=0)
        # head[2:] = BN1d + Dropout + Linear — the learnable part in head-only training
        _head_tail = nn.Sequential(*list(model.head.children())[2:])
        if verbose:
            print(f'    Feature cache: {feats_all.shape[0]} × '
                  f'{feats_all.shape[1]}-dim  (backbone runs once, head-only train)')
    else:
        feat_loader = None
        _head_tail  = None

    # Jetson boards (aarch64 + CUDA) share one RAM between CPU/GPU, so
    # pin_memory there just locks scarce memory instead of helping.
    _pin = (device.type == 'cuda' and platform.machine() != 'aarch64')
    loader = DataLoader(support_ds,
                        batch_size=min(max(batch_size, 2), len(support_ds)),
                        shuffle=True, num_workers=0,
                        pin_memory=_pin)

    focal  = FocalLoss(gamma=gamma, label_smoothing=0.05)
    # label_smoothing kwarg on CrossEntropyLoss needs torch >=1.10 (Jetson ships 1.8).
    def ce(logits, targets):
        return _cross_entropy_smooth(logits, targets, smoothing=0.05)
    supcon = SupConLoss(temperature=supcon_temp) if use_supcon else None

    # class_ids restricts the loss to a subset of output classes when some
    # activities are absent on this device; labels are 0..N-1 in that order.
    cid = (torch.as_tensor(class_ids, dtype=torch.long, device=device)
           if class_ids is not None else None)
    def _sel(lg):
        return lg.index_select(1, cid) if cid is not None else lg

    if unfreeze_stage3:
        s3_params = [p for mod in _s3_modules for p in mod.parameters()]
        param_groups = [
            {'params': s3_params,                      'lr': lr},
            {'params': list(model.head.parameters()),  'lr': lr * 0.5},
        ]
    else:
        param_groups = [{'params': list(model.head.parameters()), 'lr': lr}]

    optim = torch.optim.Adam(param_groups, lr=lr)

    # Warmup -> cosine via LambdaLR since LinearLR/SequentialLR need torch
    # >=1.11 (Jetson Nano JetPack 4 ships 1.8).
    warm = min(warmup_epochs, max(epochs // 5, 1))
    _eta = 1e-6 / max(lr, 1e-12)
    def _lr_lambda(ep: int) -> float:
        if ep < warm:
            return 0.1 + 0.9 * ep / max(warm, 1)
        t = (ep - warm) / max(epochs - warm, 1)
        return _eta + (1.0 - _eta) * 0.5 * (1.0 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=_lr_lambda)

    # Resources
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    rss_before  = _rss_mb()
    cpu_t_start = _cpu_time_s()
    wall_start  = time.perf_counter()
    if HAS_PSUTIL:
        psutil.cpu_percent(interval=None)

    model.train()
    if freeze_bn:
        for m in model.head.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

    compute_s  = 0.0
    best_loss  = float('inf')
    no_improve = 0
    actual_eps = 0

    ep_bar = _tqdm(range(epochs), desc='    fine-tune', unit='ep',
                   ncols=80, leave=True, disable=not HAS_TQDM)
    active_loader = feat_loader if use_feat_cache else loader
    for ep in ep_bar:
        batch_loss, n_batches = 0.0, 0
        for x, y in active_loader:
            x, y = x.to(device), y.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            _t = time.perf_counter()
            optim.zero_grad(set_to_none=True)

            if use_feat_cache:
                # x is the cached flat feature (after GAP+Flatten, before BN).
                # _head_tail = BN1d + Dropout + Linear — avoids re-running GAP on 2D tensor.
                if use_mixup:
                    x_m, y_a, y_b, lam = _mixup_batch(x, y, mixup_alpha)
                    logits = _sel(_head_tail(x_m))
                    loss = (lam       * (focal(logits, y_a) + ce(logits, y_a)) +
                            (1 - lam) * (focal(logits, y_b) + ce(logits, y_b)))
                else:
                    logits = _sel(_head_tail(x))
                    loss   = focal(logits, y) + ce(logits, y)
            elif use_supcon:
                logits, feats = model(x, return_features=True)
                logits  = _sel(logits)
                sc_loss = supcon(feats, y)
                if use_mixup:
                    x_m, y_a, y_b, lam = _mixup_batch(x, y, mixup_alpha)
                    logits_m = _sel(model(x_m))
                    fc_loss  = (lam       * (focal(logits_m, y_a) + ce(logits_m, y_a)) +
                                (1 - lam) * (focal(logits_m, y_b) + ce(logits_m, y_b)))
                else:
                    fc_loss = focal(logits, y) + ce(logits, y)
                loss = fc_loss + supcon_weight * sc_loss
            elif use_mixup:
                x_m, y_a, y_b, lam = _mixup_batch(x, y, mixup_alpha)
                logits = _sel(model(x_m))
                loss   = (lam       * (focal(logits, y_a) + ce(logits, y_a)) +
                          (1 - lam) * (focal(logits, y_b) + ce(logits, y_b)))
            else:
                logits = _sel(model(x))
                loss   = focal(logits, y) + ce(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optim.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            compute_s  += time.perf_counter() - _t
            batch_loss += loss.item()
            n_batches  += 1

        sched.step()
        actual_eps += 1
        ep_loss = batch_loss / max(n_batches, 1)
        if HAS_TQDM:
            ep_bar.set_postfix(loss=f'{ep_loss:.4f}')

        if ep_loss < best_loss - 1e-4:
            best_loss  = ep_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f'    Early stop at epoch {ep+1}/{epochs}'
                          f'  (no improvement for {patience} epochs)')
                break

    wall_elapsed = time.perf_counter() - wall_start
    cpu_t_used   = _cpu_time_s() - cpu_t_start
    rss_after    = _rss_mb()
    gpu_peak_mb  = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    if device.type == 'cuda' else 0.0)
    cpu_pct_avg  = psutil.cpu_percent(interval=None) if HAS_PSUTIL else 0.0

    for p in model.parameters():
        p.requires_grad = True

    ft_stats = {
        'wall_s':          round(wall_elapsed,          2),
        'compute_s':       round(compute_s,              2),
        'cpu_time_s':      round(cpu_t_used,             2),
        'cpu_util_pct':    round(cpu_pct_avg,            1),
        'rss_delta_mb':    round(rss_after - rss_before, 2),
        'rss_peak_mb':     round(_peak_rss_mb(),         2),
        'gpu_peak_mb':     round(gpu_peak_mb,            2),
        'flops_per_epoch': ft_flops_ep,
        'flops_total':     ft_flops_ep * actual_eps,
        'n_epochs':        actual_eps,
        'n_epochs_max':    epochs,
        'n_samples':       n_expand,
        'trainable':       trainable,
        'frozen':          frozen,
    }
    return model, ft_stats


# Evaluation

@_INFERENCE_MODE()
def eval_query(model: RepDopplerViT,
               query_ds: MemDataset,
               device: torch.device,
               n_antennas: int,
               class_ids: Optional[List[int]] = None,
               ) -> Tuple[float, float, float, np.ndarray, float, int]:
    """Late-fusion eval, batch_size=1. Returns (acc, f1_macro, f1_weighted, cm,
    model_inf_s, n_windows), where model_inf_s is pure forward-pass time.
    class_ids restricts argmax/confusion matrix to a subset of model-output
    classes (None = use raw model outputs)."""
    model.eval()
    loader  = DataLoader(query_ds, batch_size=1, shuffle=False, num_workers=0)
    preds, targets = [], []
    model_inf_s = 0.0
    n_windows   = 0
    # Multi-core CPU: run each window's n_ant antenna forwards on separate
    # cores instead of one batched call — measured ~1.65x faster (cache fit).
    # n_antennas>1 guard: with single-antenna data a 1-worker pool would just
    # drop torch to 1 intra-op thread and slow the forward down instead.
    use_par = (device.type == 'cpu' and n_antennas > 1
               and (os.cpu_count() or 1) >= n_antennas)
    pool    = ThreadPoolExecutor(max_workers=n_antennas) if use_par else None
    n_prev_threads = torch.get_num_threads()
    if use_par:
        torch.set_num_threads(1)

    def _fwd(t):
        # inference_mode/no_grad is thread-local, so re-enter it in pool workers.
        with _INFERENCE_MODE():
            return model(t)

    # Jetson JIT-compiles CUDA kernels on first forward, adding hundreds of ms
    # to the first windows — warm the cache before the timing loop starts.
    if device.type == 'cuda' and len(query_ds) > 0:
        x0, _ = query_ds[0]
        xw = x0.unsqueeze(0).reshape(n_antennas, 1,
                                     x0.shape[1], x0.shape[2]).to(device)
        for _ in range(3):
            model(xw)
        torch.cuda.synchronize(device)

    try:
        batch_bar = _tqdm(loader, desc='    eval', unit='batch',
                          ncols=80, leave=False, disable=not HAS_TQDM)
        for x, y in batch_bar:
            B = x.shape[0]
            if use_par:
                xs = [x[:, i : i + 1] for i in range(n_antennas)]  # (B,1,T,V) each
                _t = time.perf_counter()
                outs = list(pool.map(_fwd, xs))
                model_inf_s += time.perf_counter() - _t
                logits = torch.stack(outs, dim=1).sum(1)           # (B, C)
            else:
                x_exp = x.reshape(B * n_antennas, 1,
                                  x.shape[2], x.shape[3]).to(device)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                _t = time.perf_counter()
                out = model(x_exp)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                model_inf_s += time.perf_counter() - _t
                logits = out.reshape(B, n_antennas, -1).sum(1)
            if class_ids is not None:
                # Model head may have more outputs than the evaluated activity subset.
                logits = logits[:, class_ids]
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(y.numpy())
            n_windows += B
    finally:
        if use_par:
            torch.set_num_threads(n_prev_threads)
            pool.shutdown()
    # metrics computed outside the timing window
    label_set = (list(range(len(class_ids))) if class_ids is not None
                 else sorted(set(targets)))
    acc    = accuracy_score(targets, preds)
    f1_mac = f1_score(targets, preds, average='macro',    zero_division=0, labels=label_set)
    f1_wt  = f1_score(targets, preds, average='weighted', zero_division=0, labels=label_set)
    cm     = confusion_matrix(targets, preds, labels=label_set)
    return acc, f1_mac, f1_wt, cm, model_inf_s, n_windows


def _recall_from_cm(cm: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.diag(cm).astype(float) / cm.sum(axis=1)
        return np.nan_to_num(r)


@torch.no_grad()
def profile_single_window(model: nn.Module,
                           device: torch.device,
                           n_antennas: int,
                           vel_bins: int = 100,
                           n_warmup: int = 3) -> Tuple[float, float, float]:
    """Profile latency + peak memory for one deployment-mode window.
    Returns (latency_ms, cpu_rss_peak_delta_mb, peak_gpu_mb, parallel_latency_ms) —
    parallel_latency_ms is the one-antenna-per-core thread-pool timing (CPU
    multi-core with n_antennas>1 only, else None)."""
    model.eval()
    x = torch.randn(n_antennas, 1, WINDOW_FRAMES, vel_bins, device=device)
    for _ in _tqdm(range(n_warmup), desc='    warmup', unit='pass',
                   ncols=80, leave=False, disable=not HAS_TQDM):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - t0) * 1000
    gpu_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
              if device.type == 'cuda' else 0.0)
    # CPU RAM: sample RSS every 1 ms during a dedicated forward pass
    rss_samples: list = []
    stop_evt = threading.Event()
    def _sampler():
        while not stop_evt.is_set():
            rss_samples.append(_rss_mb())
            time.sleep(0.001)
    rss_before = _rss_mb()
    sampler_t = threading.Thread(target=_sampler, daemon=True)
    sampler_t.start()
    model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    stop_evt.set()
    sampler_t.join()
    peak_rss = max(rss_samples) if rss_samples else _rss_mb()
    cpu_rss_delta = max(peak_rss - rss_before, 0.0)

    # Parallel per-antenna latency (CPU, multi-antenna only): one batch-1
    # forward per core via a thread pool. GPU and single-antenna data skip
    # this since a single call is already optimal there. Any failure just
    # skips the metric; the standard latency above is still reported.
    par_ms: Optional[float] = None
    if (device.type == 'cpu' and n_antennas > 1
            and (os.cpu_count() or 1) >= n_antennas):
        try:
            n_prev = torch.get_num_threads()
            torch.set_num_threads(1)   # intra-op off; parallelism via the pool
            try:
                xs = [x[i:i + 1] for i in range(n_antennas)]
                def _fwd(t):
                    with torch.no_grad():
                        return model(t)
                with ThreadPoolExecutor(max_workers=n_antennas) as pool:
                    for _ in range(n_warmup):
                        list(pool.map(_fwd, xs))
                    t0 = time.perf_counter()
                    list(pool.map(_fwd, xs))
                    par_ms = (time.perf_counter() - t0) * 1000
            finally:
                torch.set_num_threads(n_prev)
        except Exception:
            par_ms = None

    return latency_ms, cpu_rss_delta, gpu_mb, par_ms


def zero_shot_eval(model, all_windows, all_labels,
                   device, n_antennas, acts, out_dir,
                   scenario: str = 'S2a',
                   max_inf_windows: Optional[int] = None,
                   class_ids: Optional[List[int]] = None,
                   ) -> Tuple[float, float, np.ndarray, Dict]:
    """Eval pruned model on scenario data without any fine-tuning.
    max_inf_windows: if set, cap inference to this many windows per activity."""
    print(f'\n── Zero-Shot (pruned model on {scenario}, no fine-tuning) ──')
    inf_w, inf_l = ((cap_windows_per_class(all_windows, all_labels, max_inf_windows))
                    if max_inf_windows else (all_windows, all_labels))
    q_ds  = MemDataset(inf_w, inf_l, expand=False)
    total = len(inf_l)
    if max_inf_windows:
        print(f'  Inference windows      : {total}  '
              f'(capped to {max_inf_windows}/activity from {len(all_labels)} total)')
    else:
        print(f'  Total testing windows  : {total}  '
              f'({total * n_antennas} per-antenna samples)')

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    rss_before = _rss_mb()

    acc, f1_mac, f1_wt, cm, inf_model_s, n_win = eval_query(
        model, q_ds, device, n_antennas, class_ids=class_ids)

    inf_rss_delta = _rss_mb() - rss_before
    inf_rss_peak  = _rss_mb()
    inf_gpu_mb    = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                     if device.type == 'cuda' else 0.0)

    recall = _recall_from_cm(cm)
    print(f'  Accuracy   : {acc:.4f}')
    print(f'  F1 macro   : {f1_mac:.4f}   F1 weighted: {f1_wt:.4f}')
    print(f'  Per-class recall:')
    for a, r in zip(acts, recall):
        print(f'    {a}:{ACTIVITY_MAP.get(a, a):<12} {r:.4f}')

    win_real_s = WINDOW_FRAMES / SAMPLING_HZ
    per_win_ms = (inf_model_s * 1000) / n_win if n_win > 0 else 0.0
    throughput  = n_win / inf_model_s if inf_model_s > 0 else 0.0
    rtf         = (n_win * win_real_s) / inf_model_s if inf_model_s > 0 else 0.0

    print(f'  Inference profile (query-set eval):')
    print(f'    Query shape   : {n_win} windows × '
          f'({n_antennas} ant × {WINDOW_FRAMES} frames × 100 vel-bins)')
    print(f'    Window        : {WINDOW_FRAMES}/{SAMPLING_HZ} Hz = {win_real_s:.3f} s of real data')
    print(f'    Model time    : {inf_model_s:.3f} s  '
          f'(pure forward-pass, all {n_win} windows)')
    print(f'    Per window    : {per_win_ms:.3f} ms/window')
    print(f'      Includes: {n_antennas} per-ant forward passes + late-fusion sum')
    print(f'      Excludes: DataLoader, .to(device), tensor conversion, metrics')
    print(f'    Throughput    : {throughput:.1f} windows/s  →  '
          f'{throughput * win_real_s:.1f} s of radar data/s')
    print(f'    Real-time fac.: {rtf:.1f}×')
    print(f'    Peak CPU RAM  : {inf_rss_peak:.1f} MB')
    print(f'    Peak GPU RAM  : {inf_gpu_mb:.1f} MB')
    print(f'    RSS delta     : {inf_rss_delta:+.1f} MB')

    sw_lat_ms, sw_rss_mb, sw_gpu_mb, sw_par_ms = profile_single_window(
        model, device, n_antennas)
    print(f'  Single-window deployment inference (batch=1):')
    print(f'    Input         : {n_antennas} ant × {WINDOW_FRAMES} frames × 100 vel-bins')
    print(f'    Latency       : {sw_lat_ms:.3f} ms / window  ← report in paper')
    if sw_par_ms is not None:
        print(f'    Latency ({n_antennas}-core): {sw_par_ms:.3f} ms / window  '
              f'(one antenna per core, {sw_lat_ms/max(sw_par_ms,1e-9):.2f}× faster)')
    print(f'    Peak CPU RAM  : {sw_rss_mb:.1f} MB  ← report in paper')
    print(f'    Peak GPU RAM  : {sw_gpu_mb:.1f} MB')

    if HAS_PLOT:
        class_names = [f'{a}:{ACTIVITY_MAP.get(a,a)}' for a in acts]
        plot_confusion_matrix(cm, class_names, out_dir / 'cm_zeroshot.png')

    zs_inf_stats = {
        'inf_model_s':   inf_model_s,
        'n_windows':     n_win,
        'per_win_ms':    per_win_ms,
        'throughput':    throughput,
        'rtf':           rtf,
        'rss_peak_mb':   inf_rss_peak,
        'rss_delta_mb':  inf_rss_delta,
        'gpu_mb':        inf_gpu_mb,
        'sw_lat_ms':     sw_lat_ms,
        'sw_par_ms':     sw_par_ms,
        'sw_rss_mb':     sw_rss_mb,
        'sw_gpu_mb':     sw_gpu_mb,
    }
    return acc, f1_mac, recall, zs_inf_stats


# Main few-shot experiment loop

def run_fewshot_experiment(pruned: RepDopplerViT,
                           all_windows, all_labels,
                           n_antennas: int,
                           device: torch.device,
                           macs_per_ant: int,
                           k_shots: List[int],
                           n_trials: int,
                           acts: List[str],
                           finetune_epochs: int,
                           out_dir: Path,
                           lr: float            = 5e-4,
                           batch_size: int      = 32,
                           gamma: float         = 2.0,
                           patience: int        = 10,
                           warmup_epochs: int   = 5,
                           mixup_alpha: float   = 0.2,
                           unfreeze_k: int      = 5,
                           supcon_weight: float = 0.1,
                           supcon_temp: float   = 0.07,
                           max_inf_windows: Optional[int] = None,
                           save_meta: Optional[dict] = None,
                           class_ids: Optional[List[int]] = None,
                           ) -> Dict:
    """
    For each K: n_trials random seeds → fine-tune → eval → aggregate.
    Stage3 auto-unfrozen for K >= unfreeze_k (default 5).
    max_inf_windows: cap query set per activity for fast inference timing.
    Fine-tuning always uses the full split (no cap on support set).
    Confusion matrix saved is the sum of all trial CMs (row-normalised =
    mean recall per class, consistent with reported mean statistics).
    """
    print(f'\n── Phase 3: Few-Shot Fine-Tuning  (trials={n_trials}, '
          f'γ={gamma}, lr={lr}, bs={batch_size}, epochs={finetune_epochs}) ──')
    print(f'   warmup={warmup_epochs}ep  mixup_α={mixup_alpha}  '
          f'supcon_w={supcon_weight}  stage3_unfreeze_k≥{unfreeze_k}')
    results:  Dict    = {}
    n_classes         = len(set(all_labels))
    total_windows     = len(all_labels)
    class_names       = [f'{a}:{ACTIVITY_MAP.get(a,a)}' for a in acts]

    for K in k_shots:
        accs, f1s_mac, f1s_wt = [], [], []
        recalls_per_class      = [[] for _ in range(n_classes)]
        wall_times, cpu_times  = [], []
        ft_gpu_mbs, ft_rss_mbs, ft_rss_peaks = [], [], []
        inf_wall_times = []
        inf_gpu_mbs, inf_rss_mbs, inf_rss_peaks = [], [], []
        sw_latency_ms, sw_rss_mbs, sw_gpu_mbs = [], [], []
        ft_flops_ep            = None
        ft_flops_tot_trial     = None
        do_stage3              = K >= unfreeze_k
        best_m_trial:   Optional[RepDopplerViT] = None
        best_acc_trial: float = -1.0

        n_support   = K * n_classes
        n_query_est = total_windows - n_support

        print(f'\n  K={K} shots/class  →  {n_support} support windows  '
              f'|  ~{n_query_est} query windows  '
              f'|  stage3_unfreeze={do_stage3}')
        print(f'     Support per-antenna : {n_support * n_antennas} samples  '
              f'(expand, used for training)')
        print(f'     Query   per-antenna : ~{n_query_est * n_antennas} samples  '
              f'(late-fusion eval)')

        # Skip K if there are not enough windows to form a non-empty query set
        windows_per_class = total_windows // n_classes
        if K >= windows_per_class:
            print(f'  [skip] K={K}: {windows_per_class} windows/class — '
                  f'all would go to support, query set empty. '
                  f'Use --max_windows_per_class > {K} or reduce --k_shots.')
            continue

        all_cms: List[np.ndarray] = []   # accumulate all trial CMs for mean plot
        q_l_last: List[int]       = []

        trial_bar = _tqdm(range(n_trials), desc=f'  K={K:>3} trials', unit='trial',
                          ncols=80, leave=True, disable=not HAS_TQDM)
        for trial in trial_bar:
            seed = 42 + trial * 7
            sup_w, sup_l, q_w, q_l = split_fewshot(
                all_windows, all_labels, k_shot=K, seed=seed)

            if not q_w:
                print(f'    [skip] trial {trial+1}: empty query set after split')
                continue

            sup_ds = MemDataset(sup_w, sup_l, augment=True,  expand=True)
            # Fine-tuning uses full query split; inference caps per activity for speed
            inf_w, inf_l = (cap_windows_per_class(q_w, q_l, max_inf_windows)
                            if max_inf_windows else (q_w, q_l))
            q_ds   = MemDataset(inf_w, inf_l, augment=False, expand=False)

            m = copy.deepcopy(pruned).to(device)
            m, ft_stats = few_shot_finetune(
                m, sup_ds, device, n_antennas, macs_per_ant,
                lr=lr, epochs=finetune_epochs,
                unfreeze_stage3=do_stage3,
                batch_size=batch_size, gamma=gamma,
                patience=patience,
                warmup_epochs=warmup_epochs,
                mixup_alpha=mixup_alpha,
                supcon_weight=supcon_weight,
                supcon_temp=supcon_temp,
                class_ids=class_ids,
            )

            # Inference profiling for query-set eval
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            rss_inf_before = _rss_mb()

            acc, f1_mac, f1_wt, cm, inf_model_s, n_q_win = eval_query(
                m, q_ds, device, n_antennas, class_ids=class_ids)

            inf_rss = _rss_mb() - rss_inf_before
            inf_rss_peak = _rss_mb()
            inf_gpu = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                       if device.type == 'cuda' else 0.0)

            recall = _recall_from_cm(cm)

            if acc > best_acc_trial:
                best_acc_trial = acc
                best_m_trial   = copy.deepcopy(m)

            accs.append(acc); f1s_mac.append(f1_mac); f1s_wt.append(f1_wt)
            for ci, r in enumerate(recall):
                recalls_per_class[ci].append(r)
            wall_times.append(ft_stats['compute_s'])
            cpu_times.append(ft_stats['cpu_time_s'])
            ft_gpu_mbs.append(ft_stats['gpu_peak_mb'])
            ft_rss_mbs.append(ft_stats['rss_delta_mb'])
            ft_rss_peaks.append(ft_stats['rss_peak_mb'])
            ft_flops_ep        = ft_stats['flops_per_epoch']
            ft_flops_tot_trial = ft_stats['flops_total']
            inf_wall_times.append(inf_model_s)
            inf_gpu_mbs.append(inf_gpu)
            inf_rss_mbs.append(inf_rss)
            inf_rss_peaks.append(inf_rss_peak)

            all_cms.append(cm)   # collect for mean confusion matrix
            q_l_last = q_l

            n_q = len(q_l)
            per_win_ms_trial = (inf_model_s * 1000) / n_q_win if n_q_win > 0 else 0.0
            if HAS_TQDM:
                trial_bar.set_postfix(acc=f'{acc:.4f}', f1=f'{f1_mac:.4f}',
                                      ft=f'{ft_stats["compute_s"]:.1f}s',
                                      inf=f'{inf_model_s:.3f}s')
            print(f'    trial {trial+1} (seed={seed}): '
                  f'acc={acc:.4f}  f1={f1_mac:.4f}  '
                  f'(query={n_q} win)  '
                  f'ft={ft_stats["compute_s"]:.1f}s  '
                  f'inf={inf_model_s:.3f}s  '
                  f'inf/win={per_win_ms_trial:.3f}ms  '
                  f'RAM{ft_stats["rss_delta_mb"]:+.0f}MB  '
                  + (f'GPU={ft_stats["gpu_peak_mb"]:.0f}MB'
                     if ft_stats["gpu_peak_mb"] > 0 else ''))

        if not accs:
            print(f'  [skip] K={K}: no completed trials (query set was empty for all trials)')
            continue

        # Save best-trial model for this K (already reparameterized inside few_shot_finetune)
        if save_meta is not None and best_m_trial is not None:
            ft_path = out_dir / f'finetuned_model_K{K}.pth'
            _hybrid_ft = getattr(best_m_trial, '_hybrid_stage3', False)
            _d3_ft     = len(best_m_trial.stage3_local if _hybrid_ft else best_m_trial.stage3)
            torch.save({
                'model_state':    best_m_trial.state_dict(),
                'channels':       save_meta['channels'],
                'depth':          (len(best_m_trial.stage1),
                                   len(best_m_trial.stage2), _d3_ft),
                'n_classes':      save_meta['n_classes'],
                'n_antennas':     save_meta['n_antennas'],
                'label_dict':     save_meta['label_dict'],
                'attn_mlp_ratio': save_meta['attn_mlp_ratio'],
                'simple_head':    save_meta['simple_head'],
                'mobile_stage3':  save_meta['mobile_stage3'],
                'hybrid_stage3':  save_meta['hybrid_stage3'],
                'k_shot':         K,
                'acc':            best_acc_trial,
                'reparameterized': True,
                'finetuned':      True,
                'prune_stats':    save_meta.get('prune_stats'),
            }, ft_path)
            print(f'  Best K={K} model (acc={best_acc_trial:.4f}) → {ft_path}')

        # Profile single-window latency once per K (using last fine-tuned model)
        sw_lat, sw_rss, sw_gpu, sw_par = profile_single_window(m, device, n_antennas)
        sw_latency_ms = [sw_lat]
        sw_rss_mbs    = [sw_rss]
        sw_gpu_mbs    = [sw_gpu]

        mean_acc = float(np.mean(accs));    std_acc = float(np.std(accs))
        mean_f1  = float(np.mean(f1s_mac)); std_f1  = float(np.std(f1s_mac))

        print(f'  ► K={K}  Accuracy = {mean_acc:.4f} ± {std_acc:.4f}  '
              f'|  F1 macro = {mean_f1:.4f} ± {std_f1:.4f}')

        print(f'     Per-class recall (mean ± std):')
        per_class_stats = {}
        for ci, a in enumerate(acts):
            rm = float(np.mean(recalls_per_class[ci]))
            rs = float(np.std(recalls_per_class[ci]))
            per_class_stats[a] = (rm, rs)
        hard_a = min(per_class_stats, key=lambda a: per_class_stats[a][0])
        for a, (rm, rs) in per_class_stats.items():
            flag = '  ← hardest' if (a == hard_a and rm < 1.0) else ''
            print(f'       {a}:{ACTIVITY_MAP.get(a,"?"):<12} '
                  f'{rm:.4f} ± {rs:.4f}{flag}')

        print(f'     Fine-tune profile (mean across {n_trials} trials):')
        print(f'       Compute time  : {np.mean(wall_times):.1f} ± {np.std(wall_times):.1f} s'
              f'  (forward + loss + backward + optim only)')
        print(f'       CPU time      : {np.mean(cpu_times):.1f} ± {np.std(cpu_times):.1f} s')
        print(f'       FLOPs/epoch   : {ft_flops_ep/1e6:.1f} M')
        print(f'       FLOPs total   : {ft_flops_tot_trial/1e9:.2f} G  '
              f'({finetune_epochs} epochs)')
        print(f'       Peak CPU RAM  : {np.mean(ft_rss_peaks):.1f} MB')
        print(f'       Peak GPU RAM  : {np.mean(ft_gpu_mbs):.1f} MB')
        print(f'       RSS delta     : {np.mean(ft_rss_mbs):+.1f} MB')

        n_q_mean       = len(q_l_last) if q_l_last else n_query_est
        win_real_s     = WINDOW_FRAMES / SAMPLING_HZ
        inf_model_mean = float(np.mean(inf_wall_times))
        per_win_ms     = (inf_model_mean * 1000) / n_q_mean if n_q_mean > 0 else 0.0
        throughput     = n_q_mean / inf_model_mean if inf_model_mean > 0 else 0.0
        rtf            = (n_q_mean * win_real_s) / inf_model_mean if inf_model_mean > 0 else 0.0

        print(f'     Inference profile — query-set eval (mean across {n_trials} trials):')
        print(f'       Query shape   : {n_q_mean} windows × '
              f'({n_antennas} ant × {WINDOW_FRAMES} frames × 100 vel-bins)')
        print(f'       Window = {WINDOW_FRAMES}/{SAMPLING_HZ} Hz = {win_real_s:.3f} s of real data')
        print(f'       Model time    : {inf_model_mean:.3f} ± {np.std(inf_wall_times):.3f} s'
              f'  (pure model forward-pass, all {n_q_mean} windows)')
        print(f'       Per window    : {per_win_ms:.3f} ms/window'
              f'  ← model time / n_windows')
        print(f'         What 1 window includes : {n_antennas} per-ant forward passes + late-fusion sum')
        print(f'         What is excluded       : DataLoader, .to(device), tensor conversion, metrics')
        print(f'       Throughput    : {throughput:.1f} windows/s  →  '
              f'{throughput * win_real_s:.1f} s of radar data/s')
        print(f'       Real-time fac.: {rtf:.1f}×  '
              f'({n_q_mean * win_real_s:.0f} s data in {inf_model_mean:.3f} s model time)')
        print(f'       Peak CPU RAM  : {np.mean(inf_rss_peaks):.1f} MB')
        print(f'       Peak GPU RAM  : {np.mean(inf_gpu_mbs):.1f} MB')
        print(f'       RSS delta     : {np.mean(inf_rss_mbs):+.1f} MB'
              f'  (change in physical RAM during eval)')

        print(f'     Single-window deployment inference (batch=1, mean across {n_trials} trials):')
        print(f'       Input         : {n_antennas} ant × {WINDOW_FRAMES} frames × 100 vel-bins  (1 radar window)')
        print(f'       Latency       : {np.mean(sw_latency_ms):.3f} ± {np.std(sw_latency_ms):.3f} ms')
        if sw_par is not None:
            print(f'       Latency ({n_antennas}-core): {sw_par:.3f} ms  '
                  f'(one antenna per core)')
        print(f'       Peak CPU RAM  : {np.mean(sw_rss_mbs):.1f} MB  ← report this in paper')
        print(f'       Peak GPU RAM  : {np.mean(sw_gpu_mbs):.1f} MB')

        if HAS_PLOT and all_cms:
            # Summed CM row-normalises to mean recall per class.
            mean_cm = np.sum(all_cms, axis=0)
            plot_confusion_matrix(mean_cm, class_names,
                                  out_dir / f'cm_fewshot_K{K}.png')

        results[K] = {
            'acc': mean_acc, 'acc_std': std_acc,
            'f1_mac': mean_f1, 'f1_mac_std': std_f1,
            'f1_wt': float(np.mean(f1s_wt)), 'f1_wt_std': float(np.std(f1s_wt)),
            'trials_acc': accs,
            'n_support': n_support, 'n_query': len(q_l_last) if q_l_last else n_query_est,
            'per_class_recall': per_class_stats,
            'ft_compute_mean': float(np.mean(wall_times)),
            'ft_compute_std':  float(np.std(wall_times)),
            'ft_cpu_mean':     float(np.mean(cpu_times)),
            'ft_gpu_mb':       float(np.mean(ft_gpu_mbs)),
            'ft_rss_peak_mb':  float(np.mean(ft_rss_peaks)),
            'ft_rss_mb':       float(np.mean(ft_rss_mbs)),
            'ft_flops_ep':     ft_flops_ep,
            'ft_flops_tot':    ft_flops_tot_trial,
            'inf_model_mean': float(np.mean(inf_wall_times)),
            'inf_model_std':  float(np.std(inf_wall_times)),
            'inf_gpu_mb':         float(np.mean(inf_gpu_mbs)),
            'inf_rss_peak_mb':    float(np.mean(inf_rss_peaks)),
            'inf_rss_mb':         float(np.mean(inf_rss_mbs)),
            'single_win_lat_ms':  float(np.mean(sw_latency_ms)),
            'single_win_par_ms':  float(sw_par) if sw_par is not None else None,
            'single_win_rss_mb':  float(np.mean(sw_rss_mbs)),
            'single_win_gpu_mb':  float(np.mean(sw_gpu_mbs)),
        }

    return results


# Results persistence

def write_results(path: Path,
                  ckpt_path: Path,
                  channels, n_params: int,
                  device: torch.device,
                  macs: int, n_antennas: int,
                  acts: List[str],
                  zs_acc: float, zs_f1: float,
                  zs_recall: np.ndarray,
                  results: Dict,
                  gamma: float, lr: float,
                  batch_size: int, epochs: int,
                  prune_stats: Optional[dict] = None,
                  device_prof: Optional[dict] = None,
                  windows_per_class: Optional[int] = None,
                  total_windows: Optional[int] = None,
                  zs_inf_stats: Optional[dict] = None):
    """Write comprehensive results to a text file."""
    sep  = '=' * 66
    sep2 = '─' * 66

    # Pruning summary block
    ps         = prune_stats or {}
    dp         = device_prof or {}
    keep_ratio = ps.get('keep_ratio', 1.0)
    p_before   = ps.get('params_before', None)
    p_after    = ps.get('params_after',  n_params)
    mb_before  = ps.get('mb_before',     None)
    mb_after   = ps.get('mb_after',      n_params * 4 / 1024**2)
    comp       = (p_before / p_after) if (p_before and p_after) else 1.0
    kept_pct   = (p_after  / p_before * 100) if p_before else 100.0
    dev_name   = dp.get('cpu_model', 'unknown')
    gflops     = dp.get('gflops', None)

    prune_lines = [
        '',
        '=== Pruning Summary (Phase 1 + 2) ===',
        f'Target hardware  : {dev_name}',
    ]
    if gflops is not None:
        prune_lines.append(f'Device GFLOPS    : {gflops:.2f} G-MACs/s')
    prune_lines += [
        f'Keep ratio       : {keep_ratio:.2f}',
        f'Channels (pruned): {channels}  (original: (16, 32, 64))',
    ]
    if p_before is not None:
        prune_lines += [
            f'Params           : {p_before:,} → {p_after:,}'
            f'  ({kept_pct:.1f}% kept,  {comp:.2f}× smaller)',
            f'Model size       : {mb_before:.2f} MB → {mb_after:.3f} MB  (float32)',
        ]
    else:
        prune_lines.append(f'Params           : {p_after:,}  ({mb_after:.3f} MB)')
    prune_lines.append('')

    win_cap_line = (
        f'Windows/class: {windows_per_class} (capped)  |  '
        f'Total windows loaded: {total_windows}'
        if windows_per_class is not None else
        f'Windows/class: all available  |  Total windows loaded: {total_windows}'
    )

    lines = [sep,
             'Device-Side Few-Shot Results',
             sep,
             f'Model      : {ckpt_path}',
             ] + prune_lines + [
             f'Channels   : {channels}',
             f'Params     : {n_params:,}  ({n_params*4/1024**2:.2f} MB)',
             f'Device     : {device}',
             f'Hyper-params: gamma={gamma}  lr={lr}  batch_size={batch_size}'
             f'  epochs={epochs}',
             win_cap_line,
             '',
             '=== Model FLOPs ===',
             f'  MACs  per antenna     : {macs/1e6:.2f} M',
             f'  FLOPs per antenna     : {macs*2/1e6:.2f} M',
             f'  FLOPs per {n_antennas}-ant window: {macs*2*n_antennas/1e6:.2f} M'
             f'  ({macs*2*n_antennas/1e9:.4f} GFLOPs)',
             '',
             '=== Zero-Shot Results ===',
             f'Accuracy   : {zs_acc:.4f}',
             f'F1 macro   : {zs_f1:.4f}',
             'Per-class recall:',
             ]
    for a, r in zip(acts, zs_recall):
        lines.append(f'  {a}:{ACTIVITY_MAP.get(a,"?"):<12} {r:.4f}')

    if zs_inf_stats:
        zi         = zs_inf_stats
        win_real_s = WINDOW_FRAMES / SAMPLING_HZ
        lines += [
            '',
            'Zero-Shot Inference Profile:',
            f'  Query shape       : {zi["n_windows"]} windows × '
            f'({n_antennas} ant × {WINDOW_FRAMES} frames × 100 vel-bins)',
            f'  Window duration   : {WINDOW_FRAMES}/{SAMPLING_HZ} Hz = {win_real_s:.3f} s of real data',
            f'  Model time        : {zi["inf_model_s"]:.3f} s  '
            f'(pure forward-pass, all {zi["n_windows"]} windows)',
            f'  Per window        : {zi["per_win_ms"]:.3f} ms/window',
            f'    Includes: {n_antennas} per-ant forward passes + late-fusion sum',
            f'    Excludes: DataLoader, .to(device), tensor conversion, metrics',
            f'  Throughput        : {zi["throughput"]:.1f} windows/s  →  '
            f'{zi["throughput"] * win_real_s:.1f} s of radar data/s',
            f'  Real-time factor  : {zi["rtf"]:.1f}×',
            f'  Peak CPU RAM      : {zi["rss_peak_mb"]:.1f} MB',
            f'  Peak GPU RAM      : {zi["gpu_mb"]:.1f} MB',
            f'  RSS delta         : {zi["rss_delta_mb"]:+.1f} MB',
            '',
            'Zero-Shot Single-window deployment inference (batch=1):',
            f'  Input             : {n_antennas} ant × {WINDOW_FRAMES} frames × 100 vel-bins',
            f'  Latency           : {zi["sw_lat_ms"]:.3f} ms / window  ← report in paper',
        ]
        if zi.get('sw_par_ms') is not None:
            lines.append(f'  Latency ({n_antennas}-core)  : {zi["sw_par_ms"]:.3f} ms / window'
                         f'  (one antenna per core)')
        lines += [
            f'  Peak CPU RAM      : {zi["sw_rss_mb"]:.1f} MB  ← report in paper',
            f'  Peak GPU RAM      : {zi["sw_gpu_mb"]:.1f} MB',
        ]
    lines.append('')

    for K, r in results.items():
        lines += [
            sep2,
            f'K={K} Results',
            sep2,
            f'Support : {r["n_support"]} windows ({K}/class)  →  '
            f'{r["n_support"] * n_antennas} per-antenna training samples',
            f'Query   : {r["n_query"]} windows  →  '
            f'{r["n_query"] * n_antennas} per-antenna eval samples',
            '',
            f'Accuracy (mean±std) : {r["acc"]:.4f} ± {r["acc_std"]:.4f}',
            f'F1 macro (mean±std) : {r["f1_mac"]:.4f} ± {r["f1_mac_std"]:.4f}',
            f'F1 weighted         : {r["f1_wt"]:.4f} ± {r["f1_wt_std"]:.4f}',
            f'Trials (acc)        : {[round(a,4) for a in r["trials_acc"]]}',
            '',
            'Per-class recall (mean ± std):',
        ]
        hard_a = min(r['per_class_recall'], key=lambda a: r['per_class_recall'][a][0])
        for a, (rm, rs) in r['per_class_recall'].items():
            flag = '  ← hardest' if (a == hard_a and rm < 1.0) else ''
            lines.append(f'  {a}:{ACTIVITY_MAP.get(a,"?"):<12} '
                         f'{rm:.4f} ± {rs:.4f}{flag}')
        nq         = r["n_query"]
        win_real_s = WINDOW_FRAMES / SAMPLING_HZ
        imodel     = r["inf_model_mean"]
        per_win_ms = (imodel * 1000) / nq if nq > 0 else 0.0
        throughput = nq / imodel if imodel > 0 else 0.0
        rtf        = (nq * win_real_s) / imodel if imodel > 0 else 0.0
        lines += [
            '',
            'Fine-tuning Profile:',
            f'  Compute time  : {r["ft_compute_mean"]:.1f} ± {r["ft_compute_std"]:.1f} s / trial'
            f'  (forward + loss + backward + optim only)',
            f'  CPU time      : {r["ft_cpu_mean"]:.1f} s / trial',
            f'  FLOPs / epoch : {r["ft_flops_ep"]/1e6:.1f} M',
            f'  FLOPs total   : {r["ft_flops_tot"]/1e9:.2f} G  ({epochs} epochs)',
            f'  Peak CPU RAM  : {r["ft_rss_peak_mb"]:.1f} MB',
            f'  Peak GPU RAM  : {r["ft_gpu_mb"]:.1f} MB',
            f'  RSS delta     : {r["ft_rss_mb"]:+.1f} MB',
            '',
            f'Inference Profile (pure model forward-pass only):',
            f'  Query shape       : {nq} windows × '
            f'({n_antennas} ant × {WINDOW_FRAMES} frames × 100 vel-bins)',
            f'  Window duration   : {WINDOW_FRAMES}/{SAMPLING_HZ} Hz = {win_real_s:.3f} s of real radar data',
            f'  Total data classed: {nq} × {win_real_s:.3f} s = {nq*win_real_s:.0f} s of radar data',
            f'  Model time        : {imodel:.3f} ± {r["inf_model_std"]:.3f} s / trial'
            f'  (pure model forward-pass for all {nq} windows)',
            f'  Per window        : {per_win_ms:.3f} ms/window  ← model time / n_windows',
            f'    Includes: {n_antennas} per-ant forward passes + late-fusion sum per window',
            f'    Excludes: DataLoader, .to(device), tensor conversion, metrics',
            f'  Throughput        : {throughput:.1f} windows/s  →  '
            f'{throughput*win_real_s:.1f} s of radar data per second',
            f'  Real-time factor  : {rtf:.1f}×  (>1 means faster than real time)',
            f'  Peak CPU RAM      : {r["inf_rss_peak_mb"]:.1f} MB',
            f'  Peak GPU RAM      : {r["inf_gpu_mb"]:.1f} MB',
            f'  RSS delta         : {r["inf_rss_mb"]:+.1f} MB'
            f'  (change in physical RAM pages during eval)',
            '',
            'Single-window deployment inference (batch=1, independent of dataset size):',
            f'  Input             : {n_antennas} ant × {WINDOW_FRAMES} frames × 100 vel-bins',
            f'  Latency           : {r["single_win_lat_ms"]:.3f} ms / window  ← report in paper',
        ]
        if r.get('single_win_par_ms') is not None:
            lines.append(f'  Latency ({n_antennas}-core)  : {r["single_win_par_ms"]:.3f} ms / window'
                         f'  (one antenna per core)')
        lines += [
            f'  Peak CPU RAM      : {r["single_win_rss_mb"]:.1f} MB  ← report in paper',
            f'  Peak GPU RAM      : {r["single_win_gpu_mb"]:.1f} MB',
            '',
        ]

    # Summary table
    lines += [sep,
              'SUMMARY TABLE',
              sep,
              f'  Model FLOPs : {macs*2*n_antennas/1e6:.2f} M / {n_antennas}-ant window'
              f'  ({macs/1e6:.2f} M MACs/ant)',
              '',
              '  Columns: FT-Compute = fine-tune compute only;'
              ' Inf/win = per-window model forward-pass time',
              f'  {"K":>6}  {"Acc":>14}  {"F1mac":>14}  {"Hardest-Recall":>16}'
              f'  {"Support":>8}  {"Query":>7}'
              f'  {"FT-Compute":>10}  {"FT-CPU":>7}'
              f'  {"Inf-Model":>9}  {"ms/win":>8}  {"FLOPs(G)":>10}',
              '  ' + '─' * 115,
              ]
    for K, r in results.items():
        hard_a, (h_rm, h_rs) = min(r['per_class_recall'].items(),
                                   key=lambda kv: kv[1][0])
        nq_  = r["n_query"]
        im_  = r["inf_model_mean"]
        pw_  = (im_ * 1000) / nq_ if nq_ > 0 else 0.0
        lines.append(
            f'  {K:>6}'
            f'  {r["acc"]:.4f}±{r["acc_std"]:.4f}'
            f'  {r["f1_mac"]:.4f}±{r["f1_mac_std"]:.4f}'
            f'  {hard_a}:{h_rm:.4f}±{h_rs:.4f}'
            f'  {r["n_support"]:>8}'
            f'  {nq_:>7}'
            f'  {r["ft_compute_mean"]:>8.1f}s'
            f'  {r["ft_cpu_mean"]:>5.1f}s'
            f'  {im_:>7.3f}s'
            f'  {pw_:>6.3f}ms'
            f'  {r["ft_flops_tot"]/1e9:>10.2f}'
        )

    path.write_text('\n'.join(lines) + '\n')


# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            'DEVICE-SIDE: Few-Shot Fine-Tuning + Evaluation (Phase 3).\n'
            'Loads pruned_model.pth from server-side sparse_finetune.py.\n\n'
            'Example:\n'
            '  python3 device_finetune.py pruned_model.pth \\\n'
            '      --doppler_dir output/doppler --activities A,C,D,F,G \\\n'
            '      --k_shots 5,10,20,30,40,50 --lr 5e-4 --batch_size 16'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('pruned_model',
        help='Path to pruned_model.pth from sparse_finetune.py.')
    p.add_argument('--doppler_dir', required=True,
        help='Directory containing signal_<SCENARIO>_<ACT>_<N>.npz files.')
    p.add_argument('--scenario', default='S2a',
        help='Scenario prefix in filenames, e.g. S2a, S1a, S3b. '
             'Files must be named signal_{scenario}_{act}_{ant}.npz. '
             'Default: S2a')
    p.add_argument('--activities', default=None,
        help='Comma-separated activity letters. '
             'Default: read from pruned_model.pth label_dict (matches training automatically).')
    p.add_argument('--k_shots', default='5,10,20,30,40,50',
        help='Comma-separated K values.')
    p.add_argument('--n_trials', type=int, default=3,
        help='Random trials per K.')
    p.add_argument('--finetune_epochs', type=int, default=60,
        help='Fine-tuning epochs per trial.')
    p.add_argument('--lr', type=float, default=5e-4,
        help='Learning rate for trainable parameters.')
    p.add_argument('--batch_size', type=int, default=32,
        help='Mini-batch size for fine-tuning.')
    p.add_argument('--gamma', type=float, default=2.0,
        help='Focal loss γ.')
    p.add_argument('--patience', type=int, default=10,
        help='Early-stopping patience (epochs with no improvement).')
    p.add_argument('--warmup_epochs', type=int, default=5,
        help='Linear LR warmup epochs before cosine annealing.')
    p.add_argument('--mixup_alpha', type=float, default=0.2,
        help='Mixup beta parameter (0 = off).')
    p.add_argument('--supcon_weight', type=float, default=0.1,
        help='Weight of SupConLoss during fine-tuning (0 = off).')
    p.add_argument('--supcon_temp', type=float, default=0.07,
        help='SupConLoss temperature.')
    p.add_argument('--unfreeze_k', type=int, default=5,
        help='Unfreeze stage3 when K >= this value.')
    p.add_argument('--inference_model', default=None,
        help='Path to a saved finetuned_model_K*.pth.  '
             'Skips all fine-tuning and runs inference only on --scenario data.  '
             'The model is already reparameterized — no extra fusing step needed.')
    p.add_argument('--max_windows_per_class', type=int, default=None,
        help='Cap the number of windows loaded per activity class '
             '(default: None = use all available).  '
             'Limits RAM by loading only the first N windows worth of frames '
             'from the .npz files.  Reported in results output.')
    p.add_argument('--temporal_stride', type=int, default=10,
        help='Subsample Doppler frames by this factor before windowing. '
             'The raw Doppler was recorded at 500 Hz (SLIDING=1). '
             'Default 10 → effective 50 Hz → 100-frame window covers 2 s of activity. '
             'Use 1 to keep original 500 Hz (gives 0.2 s windows — too short for HAR).')
    p.add_argument('--window_frames', type=int, default=500,
        help='Doppler frames per window — must match window_length used in '
             'create_train_simwisense.py (default 500).')
    p.add_argument('--window_stride', type=int, default=10,
        help='Stride between windows — must match stride_length used in '
             'create_train_simwisense.py (default 10).')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
        help='Compute device.  Default "auto" uses CUDA whenever available '
             '(Jetson GPU included).  "cpu"/"cuda" force a specific device.')
    p.add_argument('--output_dir', default=None,
        help='Output dir (default: same folder as pruned_model.pth).')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# Main

def main():
    args = parse_args()
    set_seed(args.seed)

    global WINDOW_FRAMES, WINDOW_STRIDE
    WINDOW_FRAMES = args.window_frames
    WINDOW_STRIDE = args.window_stride

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)


    if device.type == 'cpu':
        # Capped at 4: on this small model, more intra-op threads just add
        # synchronization overhead.
        n_cores = os.cpu_count() or 1
        torch.set_num_threads(4 if n_cores >= 4 else 1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass   # some builds forbid this once parallel work has started
        print(f'  [cpu] {n_cores} logical cores → '
              f'{torch.get_num_threads()} torch thread(s)')

    k_shots = [int(k.strip()) for k in args.k_shots.split(',')]

    ckpt_path = Path(args.pruned_model)
    out_dir   = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pruned model
    ckpt = _torch_load(ckpt_path, map_location=device)
    channels    = ckpt['channels']
    n_classes   = ckpt['n_classes']
    n_antennas  = ckpt['n_antennas']
    label_dict  = ckpt['label_dict']

    # Resolve activity list
    if args.activities is not None:
        acts = [a.strip().upper() for a in args.activities.split(',')]
    else:
        acts = sorted(label_dict.keys(), key=lambda k: label_dict[k])
        print(f'  [activities] auto-detected from checkpoint: {acts}')
    macs_stored = ckpt.get('macs_per_antenna', None)
    prune_stats = ckpt.get('prune_stats', {})
    device_prof = ckpt.get('device_prof', {})

    sd    = ckpt['model_state']
    depth = ckpt.get('depth') or _infer_depth(sd)
    # Older checkpoints don't store these flags; infer from weight shapes so
    # the constructed model matches exactly (class defaults would mismatch).
    simple_head = ckpt.get('simple_head', 'head.0.0.weight' not in sd)
    if 'attn_mlp_ratio' in ckpt:
        attn_mlp_ratio = ckpt['attn_mlp_ratio']
    else:
        _w = sd['stage1.0.channel_mixer.1.weight']   # (dim*ratio, dim, 1, 1)
        attn_mlp_ratio = max(1, _w.shape[0] // _w.shape[1])
        print(f'  [compat] attn_mlp_ratio inferred from weights: {attn_mlp_ratio}')
    mobile_stage3 = ckpt.get('mobile_stage3',
                             any(k.startswith('stage3.0.dw_mix') for k in sd))
    hybrid_stage3 = ckpt.get('hybrid_stage3',
                             any(k.startswith('stage3_local') for k in sd))
    model = RepDopplerViT(
        n_classes=n_classes, in_channels=1, channels=channels, depth=depth,
        attn_mlp_ratio=attn_mlp_ratio, simple_head=simple_head,
        mobile_stage3=mobile_stage3, hybrid_stage3=hybrid_stage3,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    n_params = count_params(model)

    # Derived pruning info
    p_before     = prune_stats.get('params_before', None)
    p_after      = prune_stats.get('params_after',  n_params)
    mb_before    = prune_stats.get('mb_before',     None)
    mb_after     = prune_stats.get('mb_after',      n_params * 4 / 1024**2)
    keep_ratio   = prune_stats.get('keep_ratio',    1.0)
    comp_ratio   = (p_before / p_after) if (p_before and p_after) else 1.0
    param_pct    = (p_after / p_before * 100) if p_before else 100.0
    dev_name     = (device_prof or {}).get('cpu_model', 'unknown')

    print('=' * 66)
    print('Device-Side Few-Shot Adaptation (Phase 3)')
    print('=' * 66)
    print(f'  Device       : {device}')
    print(f'  Activities   : {label_dict}')
    print()
    print(f'  ── Pruning Summary (Phase 1+2) ──────────────────────────')
    print(f'  Target HW    : {dev_name}')
    print(f'  Keep ratio   : {keep_ratio:.2f}')
    print(f'  Channels     : {channels}  '
          f'(original: (16, 32, 64))')
    if p_before is not None:
        print(f'  Params       : {p_before:,} → {p_after:,}  '
              f'({param_pct:.1f}% kept,  {comp_ratio:.2f}× smaller)')
        print(f'  Model size   : {mb_before:.2f} MB → {mb_after:.3f} MB  (float32)')
    else:
        print(f'  Params       : {p_after:,}  ({n_params*4/1024**2:.2f} MB)')
    print()
    print(f'  K shots      : {k_shots}  (×{args.n_trials} trials each)')
    print(f'  Hyper-params : γ={args.gamma}  lr={args.lr}  '
          f'bs={args.batch_size}  epochs={args.finetune_epochs}')
    if not HAS_PSUTIL:
        print('  [note] psutil not installed — CPU% monitoring disabled.')

    # MACs / FLOPs (static, from model architecture)
    macs = count_model_macs(model, (1, 1, WINDOW_FRAMES, 100))
    print(f'\n── Model FLOPs ─────────────────────────────────────────────────')
    print(f'  MACs  per antenna           : {macs/1e6:.2f} M')
    print(f'  FLOPs per antenna           : {macs*2/1e6:.2f} M')
    print(f'  FLOPs per {n_antennas}-ant window      : '
          f'{macs*2*n_antennas/1e6:.2f} M  ({macs*2*n_antennas/1e9:.4f} GFLOPs)')

    # Inference-only mode: load saved finetuned_model_K*.pth and eval
    if args.inference_model:
        ft_ckpt = _torch_load(Path(args.inference_model), map_location=device)
        ft_ch   = ft_ckpt['channels']
        ft_dep  = ft_ckpt.get('depth', (2, 2, 2))
        ft_nc   = ft_ckpt['n_classes']
        ft_na   = ft_ckpt['n_antennas']
        ft_ld   = ft_ckpt['label_dict']
        ft_model = RepDopplerViT(
            n_classes=ft_nc, in_channels=1,
            channels=ft_ch, depth=ft_dep,
            attn_mlp_ratio=ft_ckpt.get('attn_mlp_ratio', 1),
            simple_head=ft_ckpt.get('simple_head', True),
            mobile_stage3=ft_ckpt.get('mobile_stage3', False),
            hybrid_stage3=ft_ckpt.get('hybrid_stage3', False),
        ).to(device)
        # Fused checkpoints store token_mixer._fused.* instead of the br3/br1/
        # br_id branches — fuse the fresh model first so the keys line up.
        if any('._fused.' in k for k in ft_ckpt['model_state']):
            ft_model.reparameterize()
            ft_model._ft_reparameterized = True
        ft_model.load_state_dict(ft_ckpt['model_state'])
        ft_model.eval()
        ft_acts = (
            [a.strip().upper() for a in args.activities.split(',')]
            if args.activities
            else sorted(ft_ld.keys(), key=lambda k: ft_ld[k])
        )
        print('=' * 66)
        print('Inference-Only Mode  (fine-tuned model, no retraining)')
        print('=' * 66)
        print(f'  Model      : {args.inference_model}')
        print(f'  Channels   : {ft_ch}')
        print(f'  K-shot     : {ft_ckpt.get("k_shot")}  '
              f'acc at save={ft_ckpt.get("acc", 0.0):.4f}')
        print(f'  Reparam    : {ft_ckpt.get("reparameterized", False)}  '
              f'(single fused 3×3 conv per RepDWConv block — no multi-branch overhead)')
        ft_macs = count_model_macs(ft_model, (1, 1, WINDOW_FRAMES, 100))
        print(f'  MACs/ant   : {ft_macs/1e6:.2f} M  '
              f'FLOPs/{ft_na}-ant window: {ft_macs*2*ft_na/1e6:.2f} M')
        doppler_dir = Path(args.doppler_dir)
        max_win_inf = args.max_windows_per_class
        all_windows, all_labels, ld_inf = load_scenario_windows(
            doppler_dir, ft_acts, args.scenario,
            window_length=WINDOW_FRAMES, stride_length=WINDOW_STRIDE,
            temporal_stride=args.temporal_stride,
            max_windows=max_win_inf)
        if not all_windows:
            print('error: no windows found — check --doppler_dir, --scenario, --activities')
            raise SystemExit(1)
        # Drop activities with no data; evaluate over the present ones only.
        present_inf = set(all_labels)
        miss_inf    = [a for a in ft_acts if ld_inf[a] not in present_inf]
        if miss_inf:
            print(f'  [warn] no {args.scenario} windows for {miss_inf} — '
                  f'evaluating the remaining activities only.')
            kept_inf   = [a for a in ft_acts if ld_inf[a] in present_inf]
            id_map_inf = {ld_inf[a]: i for i, a in enumerate(kept_inf)}
            all_labels = [id_map_inf[l] for l in all_labels]
            ft_acts    = kept_inf
        ft_class_ids = [ft_ld[a] for a in ft_acts]
        q_ds = MemDataset(all_windows, all_labels, expand=False)
        acc, f1_mac, f1_wt, cm, inf_s, n_win = eval_query(
            ft_model, q_ds, device, ft_na, class_ids=ft_class_ids)
        per_win_ms = inf_s * 1000 / n_win if n_win > 0 else 0.0
        print(f'\n  Accuracy   : {acc:.4f}   ({acc*100:.2f}%)')
        print(f'  F1 macro   : {f1_mac:.4f}')
        print(f'  F1 weighted: {f1_wt:.4f}')
        print(f'  Inf time   : {per_win_ms:.3f} ms/window  '
              f'(total {inf_s:.3f} s for {n_win} windows)')
        sw_lat, sw_rss, sw_gpu, sw_par = profile_single_window(ft_model, device, ft_na)
        print(f'  Single-window latency : {sw_lat:.3f} ms  '
              f'peak RAM {sw_rss:.1f} MB  GPU {sw_gpu:.1f} MB'
              + (f'  |  {ft_na}-core parallel: {sw_par:.3f} ms'
                 if sw_par is not None else ''))
        if HAS_PLOT:
            class_names = [f'{a}:{ACTIVITY_MAP.get(a,a)}' for a in ft_acts]
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_confusion_matrix(cm, class_names,
                                  out_dir / f'cm_inference_{Path(args.inference_model).stem}.png')
        print('\nDone.')
        return

    # Load scenario windows
    doppler_dir = Path(args.doppler_dir)
    scenario    = args.scenario
    max_win = args.max_windows_per_class
    print(f'\n── Loading {scenario} windows from {doppler_dir} ──')
    if max_win is not None:
        print(f'  [max_windows_per_class={max_win}]  '
              f'Loading at most {max_win} windows per class to limit RAM.')
    all_windows, all_labels, ld_data = load_scenario_windows(
        doppler_dir, acts, scenario,
        window_length=WINDOW_FRAMES, stride_length=WINDOW_STRIDE,
        temporal_stride=args.temporal_stride,
        max_windows=max_win)
    if not all_windows:
        print(f'error: no {scenario} windows — check --doppler_dir, --scenario, and --activities')
        raise SystemExit(1)

    # Keep only activities that actually have data: plots, recall tables and
    # fine-tuning cover those; missing ones are dropped with a warning.
    present      = set(all_labels)
    missing_acts = [a for a in acts if ld_data[a] not in present]
    if missing_acts:
        print(f'  [warn] no {scenario} windows for {missing_acts} — dropped '
              f'from fine-tuning, evaluation and plots.')
        kept    = [a for a in acts if ld_data[a] in present]
        id_map  = {ld_data[a]: i for i, a in enumerate(kept)}
        all_labels = [id_map[l] for l in all_labels]
        acts    = kept
    # Model-head output indices for the evaluated activities (also handles
    # --activities subsets whose order differs from the trained label_dict).
    class_ids = [label_dict[a] for a in acts]

    n_cls_loaded = len(set(all_labels))
    per_class    = len(all_labels) // n_cls_loaded
    print(f'  {scenario} total  : {len(all_labels)} windows  '
          f'(per class: ~{per_class}'
          + (f', capped at {max_win}' if max_win is not None else '')
          + f',  classes: {n_cls_loaded})')
    print(f'  Per-antenna  : {len(all_labels)*n_antennas} total samples')

    # Reparameterize before any inference (fuses RepDWConv 3-branch → 1 conv)
    if not getattr(model, '_ft_reparameterized', False):
        model.reparameterize()
        model._ft_reparameterized = True

    # Zero-shot baseline
    zs_acc, zs_f1, zs_recall, zs_inf_stats = zero_shot_eval(
        model, all_windows, all_labels,
        device, n_antennas, acts, out_dir, scenario,
        class_ids=class_ids,
    )

    # Few-shot fine-tuning
    save_meta = {
        'channels':       channels,
        'depth':          depth,
        'n_classes':      n_classes,
        'n_antennas':     n_antennas,
        'label_dict':     label_dict,
        'attn_mlp_ratio': attn_mlp_ratio,
        'simple_head':    simple_head,
        'mobile_stage3':  mobile_stage3,
        'hybrid_stage3':  hybrid_stage3,
        'prune_stats':    prune_stats,
    }
    results = run_fewshot_experiment(
        model, all_windows, all_labels,
        n_antennas=n_antennas,
        device=device,
        macs_per_ant=macs,
        k_shots=k_shots,
        n_trials=args.n_trials,
        acts=acts,
        finetune_epochs=args.finetune_epochs,
        out_dir=out_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        mixup_alpha=args.mixup_alpha,
        supcon_weight=args.supcon_weight,
        supcon_temp=args.supcon_temp,
        unfreeze_k=args.unfreeze_k,
        max_inf_windows=max_win,
        save_meta=save_meta,
        class_ids=class_ids,
    )

    # Console summary
    sep = '=' * 66
    print(f'\n{sep}')
    print('SUMMARY')
    print(sep)
    print(f'  Target HW        : {dev_name}')
    print(f'  Keep ratio       : {keep_ratio:.2f}  (channel pruning fraction)')
    print(f'  Channels         : {channels}  '
          f'(original: (16, 32, 64))')
    if p_before is not None:
        print(f'  Params           : {p_before:,} → {p_after:,}  '
              f'({param_pct:.1f}% kept,  {comp_ratio:.2f}× smaller)')
        print(f'  Model size       : {mb_before:.2f} MB → {mb_after:.3f} MB  (float32)')
    else:
        print(f'  Params           : {n_params:,}  ({n_params*4/1024**2:.2f} MB)')
    print(f'  Inference FLOPs  : {macs*2*n_antennas/1e6:.1f} M / {n_antennas}-ant window')
    print(f'  Zero-shot {scenario:<5}  : acc={zs_acc:.4f}  f1={zs_f1:.4f}')
    print()

    hdr = (f'  {"K":>6}  {"Acc(mean±std)":>17}  {"F1mac":>17}  '
           f'{"Hardest-Recall":>17}  {"Support":>8}  {"Query":>7}  '
           f'{"FT-Compute":>10}  {"FT-CPU":>7}  {"Inf-Model":>9}  {"ms/win":>8}')
    print(hdr)
    print('  ' + '─' * (len(hdr) - 2))
    for K, r in results.items():
        hard_a, (h_rm, h_rs) = min(r['per_class_recall'].items(),
                                   key=lambda kv: kv[1][0])
        nq_  = r["n_query"]
        im_  = r["inf_model_mean"]
        pw_  = (im_ * 1000) / nq_ if nq_ > 0 else 0.0
        print(f'  {K:>6}'
              f'  {r["acc"]:.4f} ± {r["acc_std"]:.4f}'
              f'  {r["f1_mac"]:.4f} ± {r["f1_mac_std"]:.4f}'
              f'  {hard_a}:{h_rm:.4f} ± {h_rs:.4f}'
              f'  {r["n_support"]:>8}'
              f'  {nq_:>7}'
              f'  {r["ft_compute_mean"]:>8.1f}s'
              f'  {r["ft_cpu_mean"]:>5.1f}s'
              f'  {im_:>7.3f}s'
              f'  {pw_:>6.3f}ms')

    # Write comprehensive results file
    res_path = out_dir / 'fewshot_results.txt'
    write_results(
        res_path, ckpt_path, channels, n_params,
        device, macs, n_antennas,
        acts, zs_acc, zs_f1, zs_recall,
        results, args.gamma, args.lr, args.batch_size, args.finetune_epochs,
        prune_stats=prune_stats,
        device_prof=device_prof,
        windows_per_class=max_win,
        total_windows=len(all_labels),
        zs_inf_stats=zs_inf_stats,
    )
    print(f'\n  Results → {res_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
