#!/usr/bin/env python3
"""
sparse_finetune.py  —  Server-Side Structured Sparsity (Phases 1 + 2)

Phase 1: L1 penalty on BN scale γ drives unimportant channels toward 0.
Phase 2: structured channel pruning (Liu et al., ICCV 2017) — per-channel
importance = mean |γ|; physical weight surgery rebuilds a smaller model.
Phase 3: few-shot fine-tuning on the device — see device_finetune.py.
"""

import argparse
import json
import os
import pickle
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# Import model + helpers from training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_doppler_vit import (
    RepDopplerViT, ConvBNAct, RepBlock, RepDWConv, Downsample,
    LinearAttnBlock, MBConvBlock, DropPath,
    DopplerDataset, FocalLoss,
    count_params,
    load_split, filter_activities, set_seed,
    plot_confusion_matrix, ACTIVITY_MAP,
    evaluate,
    torch_load_compat,
)

# Stdout tee — writes every print() to both console and run_log.txt
class _Tee:
    """Redirect sys.stdout so every print goes to the terminal AND a log file."""
    def __init__(self, log_path: Path):
        self._file   = open(log_path, 'w', encoding='utf-8')
        self._stdout = sys.stdout
    def write(self, data: str):
        self._stdout.write(data)
        self._file.write(data)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        sys.stdout = self._stdout
        self._file.close()
    def __enter__(self):
        sys.stdout = self
        return self
    def __exit__(self, *_):
        self.close()

# MACs at channels=(24,48,96), depth (2,2,2), per antenna at keep_ratio=1.0.
_BASE_MACS_M   = 198.0
_FLOPS_EXPONENT = 1.5             # MACs ~ keep_ratio^1.5 (PW convs are C^2, DW are C)

_DEFAULT_SWEEP_RATIOS: List[float] = [0.25, 0.33, 0.50, 0.67, 0.75, 0.85, 1.00]

# TODO: confirm _DOPPLER_FRAME_RATE_HZ against the dataset creation script —
# unverified whether it's 10 fps (CSI 100 Hz) or 100 fps (CSI 1000 Hz) at SLIDING=10.
_DOPPLER_FRAME_RATE_HZ: float = 100.0    # STFT Doppler frames per second — VERIFY
_DEFAULT_WINDOW_FRAMES: int   = 250      # window_length used in train_doppler_vit.py


def _window_collection_ms(window_frames: int   = _DEFAULT_WINDOW_FRAMES,
                           frame_rate_hz: float = _DOPPLER_FRAME_RATE_HZ) -> float:
    """Physical time to collect one inference window — the real-time latency boundary."""
    return (window_frames / frame_rate_hz) * 1000.0

# Effective G-MACs/s (practical PyTorch throughput) per device, calibrated from
# real benchmarks (EPYC 7543) and published MobileNet/SqueezeNet Pi4 numbers.
_DEFAULT_KNOWN_DEVICES: Dict[str, dict] = {
    'pi4_8gb':     dict(arch='aarch64', has_neon=True,  freq_mhz=1800, ram_mb=8192,  gflops=4.0),
    'pi4_4gb':     dict(arch='aarch64', has_neon=True,  freq_mhz=1800, ram_mb=4096,  gflops=4.0),
    'pi4_2gb':     dict(arch='aarch64', has_neon=True,  freq_mhz=1800, ram_mb=2048,  gflops=4.0),
    'pi4_1gb':     dict(arch='aarch64', has_neon=True,  freq_mhz=1800, ram_mb=1024,  gflops=4.0),
    'pi3':         dict(arch='aarch64', has_neon=True,  freq_mhz=1200, ram_mb=1024,  gflops=2.5),
    'pizero2':     dict(arch='aarch64', has_neon=True,  freq_mhz=1000, ram_mb=512,   gflops=1.8),
    'pizero':      dict(arch='armv6l',  has_neon=False, freq_mhz=1000, ram_mb=512,   gflops=0.25),
    'jetson_nano': dict(arch='aarch64', has_neon=True,  freq_mhz=1479, ram_mb=4096,  gflops=3.3),
    'pc':          dict(arch='x86_64',  has_neon=False, freq_mhz=3000, ram_mb=16384, gflops=18.0),
}


def load_known_devices(json_path: Optional[str] = None) -> Dict[str, dict]:
    """Device capability table: built-ins merged with optional custom JSON entries."""
    devices = dict(_DEFAULT_KNOWN_DEVICES)
    if json_path:
        p = Path(json_path)
        if p.exists():
            with open(p) as fh:
                extra = json.load(fh)
            devices.update(extra)
            print(f'  [devices] {len(extra)} custom device(s) loaded from {json_path}  '
                  f'(total known: {len(devices)})')
        else:
            print(f'  [devices] {json_path} not found — using built-in device table.')
    return devices


# Device-Aware Pruning Budget

def probe_device() -> dict:
    """Collect hardware resources from the current machine — run on the deployment device."""
    import psutil

    uname    = platform.uname()
    arch     = uname.machine
    cpu_model = uname.processor or uname.machine

    has_neon = False
    try:
        with open('/proc/cpuinfo') as fh:
            cpuinfo = fh.read()
        has_neon = any(f in cpuinfo.lower() for f in ('asimd', 'neon'))
        for line in cpuinfo.splitlines():
            tag, _, val = line.partition(':')
            if tag.strip().lower() in ('model name', 'hardware', 'cpu model'):
                cpu_model = val.strip()
                break
    except OSError:
        pass

    vm   = psutil.virtual_memory()
    freq = psutil.cpu_freq()

    return {
        'hostname':           uname.node,
        'platform':           uname.system.lower(),
        'arch':               arch,
        'cpu_model':          cpu_model,
        'cpu_cores':          psutil.cpu_count(logical=True)  or 1,
        'cpu_freq_max_mhz':   int(freq.max)     if freq else 0,
        'cpu_freq_current_mhz': int(freq.current) if freq else 0,
        'ram_total_mb':       int(vm.total     / 1024**2),
        'ram_available_mb':   int(vm.available / 1024**2),
        'has_neon':           has_neon,
        'has_cuda':           torch.cuda.is_available(),
    }


def _gflops_for_profile(profile: dict) -> float:
    """Return effective G-MACs/s: prefer calibrated 'gflops', else derive from arch+freq."""
    if 'gflops' in profile:
        return max(float(profile['gflops']), 0.05)

    arch     = profile.get('arch', 'x86_64')
    has_neon = profile.get('has_neon', False)
    freq = (profile.get('cpu_freq_max_mhz')
            or profile.get('freq_mhz')
            or 1500)

    if arch in ('aarch64', 'arm64'):
        ref = 4.0 * (freq / 1800) if has_neon else 1.5 * (freq / 1800)
    elif arch == 'armv7l':
        ref = 2.0 * (freq / 1200) if has_neon else 0.8 * (freq / 1200)
    elif arch == 'armv6l':
        ref = 0.25 * (freq / 1000)
    else:                           # x86_64 / unknown
        ref = 18.0 * (freq / 3000)

    return max(ref, 0.05)


@torch.no_grad()
def benchmark_inference(model: RepDopplerViT,
                        n_antennas: int = 1,
                        window_shape: Tuple[int, int] = (250, 100),
                        n_warmup: int = 10,
                        n_runs: int = 50,
                        device: torch.device = torch.device('cpu')) -> dict:
    """Measure wall-clock inference latency for one single-antenna window on *device*."""
    model.eval().to(device)
    dummy = torch.zeros(1, 1, *window_shape, device=device)

    for _ in range(n_warmup):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        logits = model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000)

    return {
        'mean_ms':   round(float(np.mean(times_ms)),   2),
        'std_ms':    round(float(np.std(times_ms)),    2),
        'min_ms':    round(float(np.min(times_ms)),    2),
        'max_ms':    round(float(np.max(times_ms)),    2),
        'n_runs':    n_runs,
        'device':    str(device),
        'n_antennas': n_antennas,
    }


# FLOPs / MACs reporting

@torch.no_grad()
def count_model_macs(model: nn.Module,
                     input_shape: Tuple[int, ...] = (1, 1, 100, 100)) -> int:
    """Count MACs for one forward pass via Conv2d / Linear forward hooks."""
    total = [0]

    def _conv_hook(m, inp, out):
        B, C_out, H, W = out.shape
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
    model(torch.zeros(*input_shape, device=next(model.parameters()).device))
    if was_training:
        model.train()
    for h in hooks:
        h.remove()
    return total[0]


def report_model_flops(model: nn.Module, n_antennas: int,
                       window_shape: Tuple[int, int] = (100, 100)) -> int:
    """Print MACs/FLOPs for inference and return MACs."""
    macs          = count_model_macs(model, (1, 1, *window_shape))
    flops_per_ant = macs * 2
    flops_total   = flops_per_ant * n_antennas
    sep = '─' * 55
    print(f'\n{sep}')
    print(f'  Model FLOPs (inference)')
    print(sep)
    print(f'  MACs  (1 antenna)             : {macs/1e6:8.2f} M')
    print(f'  FLOPs (1 antenna)             : {flops_per_ant/1e6:8.2f} M')
    print(f'  FLOPs ({n_antennas}-ant late-fusion window): '
          f'{flops_total/1e6:8.2f} M  ({flops_total/1e9:.4f} GFLOPs)')
    print(sep)
    return macs


def _print_model_summary(model: RepDopplerViT) -> None:
    """
    Print a one-page architecture summary after loading best_model.pth.
    Verifies hybrid_stage3 / mobile_stage3 / simple_head were loaded correctly.
    """
    _hybrid = getattr(model, '_hybrid_stage3', False)
    _mobile = getattr(model, '_mobile_stage3', False)
    _simple = getattr(model, '_simple_head',   True)
    _mlp    = getattr(model, '_attn_mlp_ratio', 1)

    if _hybrid:
        arch_str = 'hybrid_stage3  (Downsample×2 → MBConv → LinearAttn)'
    elif _mobile:
        arch_str = 'mobile_stage3  (MBConv only)'
    else:
        arch_str = 'original       (LinearAttn only)'

    c1 = model.stage1[0].channel_mixer[0].num_features
    c2 = model.stage2[0].channel_mixer[0].num_features
    if _hybrid:
        c3 = model.stage3_local[0].dw_mix[0].weight.shape[0]
    elif _mobile:
        c3 = model.stage3[0].dw_mix[0].weight.shape[0]
    else:
        c3 = model.stage3[0].norm_attn.num_features

    d1 = len(model.stage1)
    d2 = len(model.stage2)
    d3 = len(model.stage3_local) if _hybrid else len(model.stage3)

    def _params(m): return sum(p.numel() for p in m.parameters())

    p_stem = _params(model.stem)
    p_s1   = _params(model.stage1) + _params(model.down1)
    p_s2   = _params(model.stage2) + _params(model.down2)
    if _hybrid:
        p_s3 = _params(model.down_pre) + _params(model.stage3_local) + _params(model.stage3_attn)
    else:
        p_s3 = _params(model.stage3)
    p_head  = _params(model.head)
    p_total = _params(model)

    sep = '─' * 60
    print(f'\n{sep}')
    print(f'  Model Summary')
    print(sep)
    print(f'  Architecture  : {arch_str}')
    print(f'  Channels      : C1={c1}  C2={c2}  C3={c3}')
    print(f'  Depth         : d1={d1}  d2={d2}  d3={d3}')
    print(f'  simple_head   : {_simple}')
    print(f'  attn_mlp_ratio: {_mlp}')
    print(sep)
    print(f'  Stem          : {p_stem:>10,} params')
    print(f'  Stage1+Down1  : {p_s1:>10,} params   ({c1} ch, {d1} RepBlocks)')
    print(f'  Stage2+Down2  : {p_s2:>10,} params   ({c2} ch, {d2} RepBlocks)')
    if _hybrid:
        print(f'  Stage3(hybrid): {p_s3:>10,} params   ({c3} ch, down_pre + {d3}×MBConv + {d3}×Attn)')
    elif _mobile:
        print(f'  Stage3(mobile): {p_s3:>10,} params   ({c3} ch, {d3} MBConvBlocks)')
    else:
        print(f'  Stage3(attn)  : {p_s3:>10,} params   ({c3} ch, {d3} LinearAttnBlocks)')
    print(f'  Head          : {p_head:>10,} params   (simple_head={_simple})')
    print(sep)
    print(f'  Total         : {p_total:>10,} params   ({p_total*4/1024**2:.2f} MB float32)')
    print(sep)


# Phase 1 — Sparsity-Inducing Retraining

def l1_bn_penalty(model: nn.Module) -> Tensor:
    """Sum of |γ| over all BN layers — add to loss to drive channels sparse."""
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            penalty = penalty + m.weight.abs().sum()
    return penalty


def sparsity_retrain(model: RepDopplerViT,
                     train_loader: DataLoader,
                     device: torch.device,
                     n_antennas: int,
                     epochs: int = 15,
                     lr: float = 2e-4,
                     l1_lambda: float = 1e-4) -> RepDopplerViT:
    """Retrain with an L1 penalty on BN γ (Network Slimming) to drive weak channels to 0."""
    print(f'\n── Phase 1: Sparsity-Inducing Retraining  ({epochs} epochs, λ={l1_lambda}) ──')
    focal = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = total_n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            B      = x.shape[0]
            logits = model(x)
            loss   = focal(logits, y) + l1_lambda * l1_bn_penalty(model)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * B
            total_n    += B
        if ep % 5 == 0 or ep == epochs:
            print(f'  Epoch {ep:3d}/{epochs}  loss={total_loss/total_n:.4f}')

    # Report γ statistics after sparsity training
    all_gamma = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            all_gamma.append(m.weight.data.abs())
    all_gamma = torch.cat(all_gamma)
    print(f'  BN |γ| stats: mean={all_gamma.mean():.4f}  '
          f'min={all_gamma.min():.4f}  '
          f'<0.05: {(all_gamma < 0.05).float().mean()*100:.1f}%')
    return model


def finetune_pruned(
        model: RepDopplerViT,
        train_loader: DataLoader,
        device: torch.device,
        epochs: int = 10,
        lr: float = 1e-4,
) -> RepDopplerViT:
    """Recover accuracy after pruning: FocalLoss only (no L1 penalty), low LR with cosine decay."""
    if epochs <= 0:
        return model

    print(f'    Fine-tuning pruned model  ({epochs} epochs, lr={lr}) ...')
    focal = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=epochs, eta_min=lr * 0.01)

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = total_n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = focal(logits, y)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * x.shape[0]
            total_n    += x.shape[0]
        sched.step()
        if ep % 5 == 0 or ep == epochs:
            print(f'    Retrain {ep:3d}/{epochs}  loss={total_loss/total_n:.4f}')

    model.eval()
    return model


# Phase 2 — Physical Channel Pruning

def _simd_granularity(device_prof: Optional[dict]) -> int:
    """Return channel granularity matching the target device's SIMD width."""
    if device_prof is None:
        return 8
    arch     = device_prof.get('arch') or device_prof.get('architecture', 'x86_64')
    has_neon = device_prof.get('has_neon', False)
    if arch in ('aarch64', 'arm64'):
        return 8   # NEON 128-bit: 4 fp32/cycle; 8 for dual-issue pipeline efficiency
    if arch == 'armv7l':
        return 8 if has_neon else 4
    if arch == 'armv6l':
        return 4   # no SIMD; 4-byte natural alignment
    return 8       # x86_64: AVX2 = 256-bit = 8 fp32


def _round_granularity(n: int, g: int) -> int:
    """Round n to the nearest multiple of g (device SIMD width), minimum g."""
    return max(g, int(round(n / g)) * g)


# BN γ importance per stage

def _stage_imp_repblock(stage: nn.Sequential) -> Tensor:
    """Per-channel importance = mean |γ| from post-BN of channel_mixer + identity BN."""
    scores = []
    for blk in stage:
        scores.append(blk.channel_mixer[4].weight.data.abs())  # post channel-mixer BN
        scores.append(blk.token_mixer.br_id.weight.data.abs()) # identity branch BN
    return torch.stack(scores).mean(0)


def _stage_imp_linear_attn(stage: nn.Sequential) -> Tensor:
    """Per-channel importance from FFN post-BN + attn_proj BN + norm_attn."""
    scores = []
    for blk in stage:
        scores.append(blk.ffn[4].weight.data.abs())
        scores.append(blk.attn_proj[1].weight.data.abs())
        scores.append(blk.norm_attn.weight.data.abs())
    return torch.stack(scores).mean(0)


def _stage_imp_mbconv(stage: nn.Sequential) -> Tensor:
    """Per-channel importance from dw_mix BN + pw_proj BN of MBConvBlock."""
    scores = []
    for blk in stage:
        scores.append(blk.dw_mix[1].weight.data.abs())   # DW output BN
        scores.append(blk.pw_proj[1].weight.data.abs())  # project output BN
    return torch.stack(scores).mean(0)


# Weight copy helpers

@torch.no_grad()
def _cp_bn(src: nn.BatchNorm2d, dst: nn.BatchNorm2d, idx: Tensor):
    dst.weight.data.copy_(src.weight.data[idx])
    dst.bias.data.copy_(src.bias.data[idx])
    dst.running_mean.data.copy_(src.running_mean.data[idx])
    dst.running_var.data.copy_(src.running_var.data[idx])
    dst.num_batches_tracked.copy_(src.num_batches_tracked)


@torch.no_grad()
def _cp_convbnact(src: ConvBNAct, dst: ConvBNAct,
                  in_idx: Optional[Tensor], out_idx: Tensor):
    """Copy ConvBNAct[Conv, BN, GELU] selecting input/output channels."""
    src_c, src_bn = src[0], src[1]
    dst_c, dst_bn = dst[0], dst[1]
    w = src_c.weight.data   # (out, in/groups, kH, kW)
    if src_c.groups > 1:    # DW conv: groups == in == out
        dst_c.weight.data.copy_(w[out_idx])
    else:                   # standard conv
        w = w[out_idx] if in_idx is None else w[:, in_idx][out_idx]
        dst_c.weight.data.copy_(w)
    _cp_bn(src_bn, dst_bn, out_idx)


@torch.no_grad()
def _top_idx(weight_2d: Tensor, k: int) -> Tensor:
    """Indices of top-k rows by L2 norm, returned sorted."""
    norms = weight_2d.pow(2).sum(dim=(1, 2, 3))
    return norms.argsort(descending=True)[:k].sort().values


# RepBlock weight surgery

@torch.no_grad()
def _cp_repblock(src: RepBlock, dst: RepBlock, idx: Tensor):
    """Copy RepBlock weights, selecting idx channels; channel mixer keeps top-L2 hidden units."""
    K = len(idx)

    # Token mixer: three DW branches, channel-independent
    for br_name in ('br3', 'br1'):
        br = getattr(src.token_mixer, br_name)
        dst_br = getattr(dst.token_mixer, br_name)
        dst_br[0].weight.data.copy_(br[0].weight.data[idx])
        _cp_bn(br[1], dst_br[1], idx)
    _cp_bn(src.token_mixer.br_id, dst.token_mixer.br_id, idx)

    # Channel mixer: BN[0] -> Conv[1] -> ReLU[2] -> Conv[3] -> BN[4]
    _cp_bn(src.channel_mixer[0], dst.channel_mixer[0], idx)

    ew      = src.channel_mixer[1].weight.data
    C_src   = ew.shape[1]
    r       = ew.shape[0] // C_src
    H       = K * r
    ew_s    = ew[:, idx, :, :]
    ei      = _top_idx(ew_s, H)
    dst.channel_mixer[1].weight.data.copy_(ew_s[ei])

    cw   = src.channel_mixer[3].weight.data
    dst.channel_mixer[3].weight.data.copy_(cw[idx][:, ei])

    _cp_bn(src.channel_mixer[4], dst.channel_mixer[4], idx)


# Downsample weight surgery

@torch.no_grad()
def _cp_downsample(src: Downsample, dst: Downsample,
                   in_idx: Tensor, out_idx: Tensor):
    """Downsample = DW stride-2 (in_idx channels) + PW channel-expand (in_idx -> out_idx)."""
    dst.dw[0].weight.data.copy_(src.dw[0].weight.data[in_idx])
    _cp_bn(src.dw[1], dst.dw[1], in_idx)
    _cp_convbnact(src.pw, dst.pw, in_idx=in_idx, out_idx=out_idx)


# LinearAttnBlock weight surgery

@torch.no_grad()
def _cp_linear_attn(src: LinearAttnBlock, dst: LinearAttnBlock, idx: Tensor):
    """Prune LinearAttnBlock channel dim to K=len(idx); attention head dim is not pruned."""
    K = len(idx)

    dst.local_dw[0].weight.data.copy_(src.local_dw[0].weight.data[idx])
    _cp_bn(src.local_dw[1], dst.local_dw[1], idx)
    _cp_bn(src.norm_attn, dst.norm_attn, idx)

    for name in ('q', 'k', 'v'):
        src_c = getattr(src, name)
        dst_c = getattr(dst, name)
        dst_c.weight.data.copy_(src_c.weight.data[:, idx])

    dst.attn_proj[0].weight.data.copy_(src.attn_proj[0].weight.data[idx])
    _cp_bn(src.attn_proj[1], dst.attn_proj[1], idx)

    # FFN hidden size derived from source shape since mlp_ratio can be 1 or 2.
    _cp_bn(src.ffn[0], dst.ffn[0], idx)
    fw      = src.ffn[1].weight.data
    C_src   = fw.shape[1]
    r       = fw.shape[0] // C_src
    H       = K * r
    fw_s    = fw[:, idx, :, :]
    fi      = _top_idx(fw_s, H)
    dst.ffn[1].weight.data.copy_(fw_s[fi])
    cw   = src.ffn[3].weight.data
    dst.ffn[3].weight.data.copy_(cw[idx][:, fi])
    _cp_bn(src.ffn[4], dst.ffn[4], idx)


# MBConvBlock weight surgery

@torch.no_grad()
def _cp_mbconv(src: MBConvBlock, dst: MBConvBlock, idx: Tensor):
    """Prune MBConvBlock channel dim to K=len(idx); pw_exp keeps top-L2 hidden rows."""
    dst.dw_mix[0].weight.data.copy_(src.dw_mix[0].weight.data[idx])
    _cp_bn(src.dw_mix[1], dst.dw_mix[1], idx)

    ew       = src.pw_exp[0].weight.data
    ew_s     = ew[:, idx, :, :]
    hidden_k = dst.pw_exp[0].weight.data.shape[0]
    ei       = _top_idx(ew_s, hidden_k)
    dst.pw_exp[0].weight.data.copy_(ew_s[ei])
    _cp_bn(src.pw_exp[1], dst.pw_exp[1], ei)

    cw = src.pw_proj[0].weight.data
    dst.pw_proj[0].weight.data.copy_(cw[idx][:, ei])
    _cp_bn(src.pw_proj[1], dst.pw_proj[1], idx)


# Head weight surgery

@torch.no_grad()
def _cp_head(src_head: nn.Sequential, dst_head: nn.Sequential, in_idx: Tensor):
    """Copy head weights; layout (simple vs conv head) is detected from head[2]'s type."""
    K = len(in_idx)
    if isinstance(src_head[2], nn.BatchNorm1d):
        _cp_bn(src_head[2], dst_head[2], in_idx)
        dst_head[4].weight.data.copy_(src_head[4].weight.data[:, in_idx])
        dst_head[4].bias.data.copy_(src_head[4].bias.data)
    else:
        hw   = src_head[0][0].weight.data
        hw_s = hw[:, in_idx, :, :]
        hi   = _top_idx(hw_s, 2 * K)
        dst_head[0][0].weight.data.copy_(hw_s[hi])
        _cp_bn(src_head[0][1], dst_head[0][1], hi)
        dst_head[4].weight.data.copy_(src_head[4].weight.data[:, hi])
        dst_head[4].bias.data.copy_(src_head[4].bias.data)


def _infer_depth(state_dict: dict) -> Tuple[int, int, int]:
    """Infer (d1, d2, d3) from a model state_dict by finding the max block index per stage."""
    depths = []
    for i in range(1, 4):
        idxs = [int(k.split('.')[1]) for k in state_dict if k.startswith(f'stage{i}.')]
        if not idxs and i == 3:
            # hybrid_stage3: keys are stage3_local.* / stage3_attn.* instead of stage3.*
            idxs = [int(k.split('.')[1]) for k in state_dict if k.startswith('stage3_local.')]
        depths.append(max(idxs) + 1 if idxs else 2)
    return tuple(depths)


# Master pruning function

def prune_model(model: RepDopplerViT,
                keep_ratio: float,
                n_classes: int,
                device: torch.device,
                granularity: int = 8) -> Tuple[RepDopplerViT, Dict]:
    """Structured channel pruning via BN-γ importance (Network Slimming, Liu et al. 2017)."""
    model.eval()
    _mobile = getattr(model, '_mobile_stage3', False)
    _hybrid = getattr(model, '_hybrid_stage3', False)

    s1_imp = _stage_imp_repblock(model.stage1)
    s2_imp = _stage_imp_repblock(model.stage2)
    if _hybrid:
        imp_local = _stage_imp_mbconv(model.stage3_local)
        imp_attn  = _stage_imp_linear_attn(model.stage3_attn)
        s3_imp    = (imp_local + imp_attn) / 2.0
    elif _mobile:
        s3_imp = _stage_imp_mbconv(model.stage3)
    else:
        s3_imp = _stage_imp_linear_attn(model.stage3)

    c1, c2, c3 = len(s1_imp), len(s2_imp), len(s3_imp)

    K1 = min(c1, _round_granularity(int(c1 * keep_ratio), granularity))
    K2 = min(c2, _round_granularity(int(c2 * keep_ratio), granularity))
    K3 = min(c3, _round_granularity(int(c3 * keep_ratio), granularity))

    s1_idx = s1_imp.argsort(descending=True)[:K1].sort().values
    s2_idx = s2_imp.argsort(descending=True)[:K2].sort().values
    s3_idx = s3_imp.argsort(descending=True)[:K3].sort().values

    print(f'\n── Phase 2: Structured Channel Pruning  (keep_ratio={keep_ratio}, granularity={granularity}) ──')
    print(f'  Stage-1 : {c1}  → {K1}  channels  ({K1/c1*100:.0f}% kept)')
    print(f'  Stage-2 : {c2}  → {K2}  channels  ({K2/c2*100:.0f}% kept)')
    print(f'  Stage-3 : {c3}  → {K3}  channels  ({K3/c3*100:.0f}% kept)')

    # Build pruned model
    _d3 = len(model.stage3_local) if _hybrid else len(model.stage3)
    pruned = RepDopplerViT(
        n_classes=n_classes, in_channels=1,
        channels=(K1, K2, K3),
        depth=(len(model.stage1), len(model.stage2), _d3),
        mlp_ratio=getattr(model, '_mlp_ratio', 2),
        attn_mlp_ratio=getattr(model, '_attn_mlp_ratio', 1),
        simple_head=getattr(model, '_simple_head', True),
        mobile_stage3=_mobile,
        hybrid_stage3=_hybrid,
    ).to(device)
    pruned.eval()

    _cp_convbnact(model.stem, pruned.stem, in_idx=None, out_idx=s1_idx)

    for src_b, dst_b in zip(model.stage1, pruned.stage1):
        _cp_repblock(src_b, dst_b, s1_idx)

    _cp_downsample(model.down1, pruned.down1, s1_idx, s2_idx)

    for src_b, dst_b in zip(model.stage2, pruned.stage2):
        _cp_repblock(src_b, dst_b, s2_idx)

    _cp_downsample(model.down2, pruned.down2, s2_idx, s3_idx)

    if _hybrid:
        for src_ds, dst_ds in zip(model.down_pre, pruned.down_pre):
            _cp_downsample(src_ds, dst_ds, s3_idx, s3_idx)
        for src_b, dst_b in zip(model.stage3_local, pruned.stage3_local):
            _cp_mbconv(src_b, dst_b, s3_idx)
        for src_b, dst_b in zip(model.stage3_attn, pruned.stage3_attn):
            _cp_linear_attn(src_b, dst_b, s3_idx)
    elif _mobile:
        for src_b, dst_b in zip(model.stage3, pruned.stage3):
            _cp_mbconv(src_b, dst_b, s3_idx)
    else:
        for src_b, dst_b in zip(model.stage3, pruned.stage3):
            _cp_linear_attn(src_b, dst_b, s3_idx)

    _cp_head(model.head, pruned.head, s3_idx)

    p_before = count_params(model)
    p_after  = count_params(pruned)
    mb_before = p_before * 4 / 1024**2
    mb_after  = p_after  * 4 / 1024**2
    print(f'  Params  : {p_before:,} → {p_after:,}  ({p_after/p_before*100:.1f}% of original)')
    print(f'  Size    : {mb_before:.2f} MB → {mb_after:.2f} MB  (float32)')

    stats = dict(
        K1=K1, K2=K2, K3=K3,
        params_before=p_before, params_after=p_after,
        mb_before=mb_before, mb_after=mb_after,
        keep_ratio=keep_ratio,
    )
    return pruned, stats



def _plot_pareto_curve(results: list, best_idx: int, out_dir: Path,
                       t_window_ms: Optional[float] = None) -> None:
    """Save latency-accuracy Pareto curve, shading the real-time-feasible zone."""
    lats   = [r['lat_for_knee'] for r in results]
    accs   = [r['acc'] * 100    for r in results]
    ratios = [r['keep_ratio']   for r in results]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Shade zones if T_window is known
    if t_window_ms is not None:
        xlim_right = max(lats) * 1.15
        ax.axvspan(0, t_window_ms, alpha=0.08, color='green',
                   label=f'Real-time zone (< {t_window_ms:.0f} ms)')
        ax.axvspan(t_window_ms, xlim_right, alpha=0.08, color='red',
                   label='Too slow for real-time')
        ax.axvline(t_window_ms, color='green', lw=1.4, ls='--', alpha=0.7)
        ax.set_xlim(0, xlim_right)

    # Classify candidates
    for i, (lat, acc, r) in enumerate(zip(lats, accs, ratios)):
        feasible = (t_window_ms is None) or (lat <= t_window_ms)
        color = 'steelblue' if feasible else 'salmon'
        ax.scatter([lat], [acc], s=55, color=color, zorder=4)
        ax.annotate(f'{r:.2f}', (lat, acc),
                    textcoords='offset points', xytext=(5, 4), fontsize=8)
    ax.plot(lats, accs, '-', color='grey', lw=1.0, alpha=0.5, zorder=2)

    # Knee
    ax.scatter([lats[best_idx]], [accs[best_idx]], s=200, color='crimson',
               marker='*', zorder=6, label=f'Selected  (keep_ratio={ratios[best_idx]:.2f})')

    ax.set_xlabel('Inference Latency (ms)', fontsize=11)
    ax.set_ylabel('S1a Test Accuracy (%)',  fontsize=11)
    ax.set_title('Latency–Accuracy Pareto Curve\n(Device-Aware Automatic Selection)',
                 fontsize=12)
    ax.legend(framealpha=0.88, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / 'pareto_curve.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f'  Pareto curve  → {path}')



# Incremental Pruning Search  (primary auto-sweep path)

def incremental_prune_search(
        sparse_model: RepDopplerViT,
        test_loader: DataLoader,
        device: torch.device,
        n_classes: int,
        n_antennas: int,
        baseline_stats: dict,
        device_prof: Optional[dict] = None,
        do_benchmark: bool = False,
        candidates: Optional[List[float]] = None,
        out_dir: Optional[Path] = None,
        train_loader: Optional[DataLoader] = None,
        retrain_epochs: int = 10,
        max_acc_drop: float = 0.06,
        granularity: int = 8,
        window_frames: int = _DEFAULT_WINDOW_FRAMES,
) -> Tuple[float, 'RepDopplerViT', dict]:
    """Descend keep_ratio from 1.0, pruning + optionally fine-tuning at each step,
    scoring accuracy vs. estimated device latency, stopping once acc_drop exceeds
    max_acc_drop. Picks the highest-scoring candidate within the accuracy budget
    (accuracy-based stopping, not latency-based, so it explores real tradeoffs
    instead of always keeping ratio=1.0 on already-fast devices).
    """
    if candidates is None:
        candidates = list(_DEFAULT_SWEEP_RATIOS)
    candidates = sorted(set(candidates), reverse=True)

    t_window_ms = _window_collection_ms(window_frames)
    gflops      = _gflops_for_profile(device_prof) if device_prof else None
    full_macs_m = _BASE_MACS_M * n_antennas
    ram_avail   = (device_prof or {}).get('ram_available_mb', 0)
    ref_acc     = baseline_stats['acc']   # reference: original model before Phase 1
    bl_lat      = baseline_stats.get('lat_ms')
    bl_cpu      = baseline_stats.get('cpu_pct')

    sep = '═' * 70

    # Print header
    print(f'\n{sep}')
    print(f'  Accuracy-Constrained Incremental Pruning Search')
    print(sep)

    if device_prof:
        print(f'  Target device  : {device_prof.get("cpu_model","?")}')
        print(f'  Architecture   : {device_prof.get("arch","?")}  '
              f'NEON={device_prof.get("has_neon","?")}')
        print(f'  Compute        : {gflops:.2f} G-MACs/s  '
              f'({device_prof.get("cpu_freq_max_mhz","?")} MHz)')
        print(f'  RAM available  : {ram_avail} MB')
    else:
        print(f'  (No device profile — latency will not be estimated)')
    print(f'  T_window       : {t_window_ms:.0f} ms  ({window_frames} frames ÷ {_DOPPLER_FRAME_RATE_HZ:.0f} Hz)')
    print(f'  Max acc drop   : {max_acc_drop*100:.0f}%  (stopping criterion)')
    if train_loader is not None and retrain_epochs > 0:
        print(f'  Post-prune retrain: {retrain_epochs} epochs per ratio step')
    else:
        print(f'  Post-prune retrain: disabled (no train_loader or retrain_epochs=0)')
    print()

    print(f'  Unpruned model  (reference point)')
    print(f'  {"─" * 50}')
    print(f'  Parameters  : {baseline_stats["params"]:,}')
    print(f'  Model size  : {baseline_stats["model_mb"]:.2f} MB  (float32 weights)')
    macs_total_m = baseline_stats['macs_per_ant'] * n_antennas / 1e6
    print(f'  MACs/window : {macs_total_m:.1f} M  ({n_antennas} antennas)')
    if bl_lat is not None:
        print(f'  Est. latency: {bl_lat:.0f} ms  on target device')
        print(f'  CPU budget  : {bl_cpu:.1f}%  ({bl_lat:.0f} ms ÷ {t_window_ms:.0f} ms × 100)')
        print(f'  Real-time?  : {"YES ✓" if bl_lat <= t_window_ms else "NO ✗"}')
    print(f'  S1a accuracy: {ref_acc * 100:.2f}%  ← accuracy budget reference')
    print()

    # Column headers
    if gflops:
        print(f'  {"ratio":>6}  {"channels":>12}  {"params":>8}  '
              f'{"acc%":>7}  {"Δacc":>7}  {"lat_ms":>8}  {"cpu%":>6}  {"score":>7}  {"status":>8}')
        print(f'  {"─"*6}  {"─"*12}  {"─"*8}  '
              f'{"─"*7}  {"─"*7}  {"─"*8}  {"─"*6}  {"─"*7}  {"─"*8}')
    else:
        print(f'  {"ratio":>6}  {"channels":>12}  {"params":>8}  '
              f'{"acc%":>7}  {"Δacc":>7}  {"score":>7}  {"status":>8}')
        print(f'  {"─"*6}  {"─"*12}  {"─"*8}  '
              f'{"─"*7}  {"─"*7}  {"─"*7}  {"─"*8}')

    results:       list      = []
    within_budget: List[int] = []
    model_cache:   dict      = {}   # idx → model (kept until selection)

    for ratio in candidates:
        # Prune
        pruned_m, stats = prune_model(sparse_model, ratio, n_classes, device,
                                      granularity)

        # Optional post-prune fine-tuning
        if train_loader is not None and retrain_epochs > 0:
            pruned_m = finetune_pruned(pruned_m, train_loader, device,
                                       epochs=retrain_epochs)

        # Accuracy on S1a test set  (server-side, real labels)
        acc, f1m, _, _, _, _, _ = evaluate(
            pruned_m, test_loader, device)

        # Device latency from MACs
        macs_m  = full_macs_m * ratio ** _FLOPS_EXPONENT
        est_lat = (macs_m / (gflops * 1e3) * 1000) if gflops else None

        # Optional actual wall-clock benchmark on this server
        act_lat: Optional[float] = None
        if do_benchmark:
            bm      = benchmark_inference(pruned_m, n_antennas=n_antennas,
                                          window_shape=(window_frames, 100),
                                          device=torch.device('cpu'))
            act_lat = bm['mean_ms']

        lat_ms   = act_lat if act_lat is not None else est_lat
        cpu_pct  = (lat_ms / t_window_ms * 100) if lat_ms is not None else None
        model_mb = stats['params_after'] * 4 / 1024**2

        # Tradeoff score: accuracy dominates within budget (0.05 tiebreaker);
        # a strong 3.0 penalty per excess T_window kicks in once over budget.
        lat_ratio = (lat_ms / t_window_ms) if lat_ms is not None else 0.0
        excess    = max(0.0, lat_ratio - 1.0)
        score     = acc * 100.0 - 3.0 * excess - 0.05 * lat_ratio

        acc_drop  = ref_acc - acc
        in_budget = acc_drop <= max_acc_drop

        # Format and print row
        drop_str = f'{acc_drop * 100:+.2f}pp'
        status   = 'OK ✓' if in_budget else f'>{max_acc_drop*100:.0f}% ✗'
        if gflops:
            lat_str = f'{lat_ms:.0f}' if lat_ms is not None else 'n/a'
            cpu_str = f'{cpu_pct:.1f}' if cpu_pct is not None else 'n/a'
            print(f'  {ratio:>6.2f}  ({stats["K1"]:2d},{stats["K2"]:2d},{stats["K3"]:2d})'
                  f'  {stats["params_after"]:>8,}  '
                  f'{acc * 100:>6.2f}%  {drop_str:>7}  '
                  f'{lat_str:>7}ms  {cpu_str:>5}%  {score:>7.2f}  {status:>8}')
        else:
            print(f'  {ratio:>6.2f}  ({stats["K1"]:2d},{stats["K2"]:2d},{stats["K3"]:2d})'
                  f'  {stats["params_after"]:>8,}  '
                  f'{acc * 100:>6.2f}%  {drop_str:>7}  '
                  f'{score:>7.2f}  {status:>8}')

        entry = dict(
            keep_ratio  = ratio,
            channels    = (stats['K1'], stats['K2'], stats['K3']),
            params      = stats['params_after'],
            model_mb    = round(model_mb, 3),
            acc         = round(acc,  4),
            f1          = round(f1m,  4),
            acc_drop    = round(float(acc_drop), 4),
            lat_ms      = round(lat_ms, 1)  if lat_ms  is not None else None,
            lat_for_knee= lat_ms,
            cpu_pct     = round(cpu_pct, 1) if cpu_pct is not None else None,
            score       = round(score,  4),
            in_budget   = in_budget,
        )
        results.append(entry)
        idx = len(results) - 1
        model_cache[idx] = pruned_m

        if in_budget:
            within_budget.append(idx)
        else:
            print(f'\n  Accuracy dropped {acc_drop*100:.2f}% > {max_acc_drop*100:.0f}% budget'
                  f' — stopping descent.')
            break   # do not evaluate smaller ratios

    # Pick the smallest keep_ratio within budget: max compression, lowest
    # latency/energy. The accuracy budget is a hard constraint, not a target.
    if within_budget:
        best_idx = min(within_budget, key=lambda i: results[i]['keep_ratio'])
        sel_drop = results[best_idx]['acc_drop'] * 100
        reason   = (f'smallest keep_ratio within {max_acc_drop*100:.0f}% accuracy budget'
                    f'  (acc_drop={sel_drop:.2f}pp, '
                    f'{len(within_budget)} candidate(s) evaluated)')
    else:
        # All candidates exceeded the budget — fall back to least-damaged model
        best_idx = min(range(len(results)), key=lambda i: results[i]['acc_drop'])
        reason   = (f'all candidates exceeded {max_acc_drop*100:.0f}% budget — '
                    f'selected smallest acc_drop')

    sel   = results[best_idx]
    sel_m = model_cache[best_idx]

    # Summary
    acc_delta = (sel['acc'] - ref_acc) * 100
    skipped   = len(candidates) - len(results)

    print()
    print(f'  {"─" * 60}')
    print(f'  Selected keep_ratio : {sel["keep_ratio"]:.2f}')
    print(f'  Channels            : {sel["channels"]}')
    print(f'  Accuracy            : {sel["acc"]*100:.2f}%  '
          f'(original: {ref_acc*100:.2f}%,  Δ = {acc_delta:+.2f} pp)')
    if sel.get('lat_ms') and bl_lat:
        speedup = bl_lat / sel['lat_ms']
        print(f'  Latency on device   : {sel["lat_ms"]:.0f} ms  '
              f'(original: {bl_lat:.0f} ms,  {speedup:.1f}× faster)')
    elif sel.get('lat_ms'):
        print(f'  Latency on device   : {sel["lat_ms"]:.0f} ms')
    if sel.get('cpu_pct') is not None:
        rt_str = 'real-time ✓' if sel['cpu_pct'] <= 100 else 'over budget ✗'
        print(f'  CPU budget used     : {sel["cpu_pct"]:.1f}%  of T_window '
              f'({t_window_ms:.0f} ms)  [{rt_str}]')
    print(f'  Selection criterion : smallest keep_ratio within {max_acc_drop*100:.0f}% accuracy budget')
    print(f'  Model size          : {sel["model_mb"]:.3f} MB  '
          f'(original: {baseline_stats["model_mb"]:.2f} MB)')
    print(f'  Reason              : {reason}')
    if skipped > 0:
        print(f'  Accuracy-stopped    : {skipped} smaller ratio(s) skipped '
              f'(would exceed {max_acc_drop*100:.0f}% drop budget)')
    print(sep)

    # Plots and JSON
    if out_dir is not None:
        if HAS_PLOT:
            _plot_pareto_curve(results, best_idx, out_dir, t_window_ms=t_window_ms)
        clean = []
        for r in results:
            fixed = {}
            for k, v in r.items():
                if isinstance(v, np.bool_):
                    fixed[k] = bool(v)
                elif isinstance(v, np.integer):
                    fixed[k] = int(v)
                elif isinstance(v, np.floating):
                    fixed[k] = float(v)
                else:
                    fixed[k] = v
            clean.append(fixed)
        with open(out_dir / 'sweep_results.json', 'w') as fp:
            json.dump({
                't_window_ms':         t_window_ms,
                'max_acc_drop':        max_acc_drop,
                'retrain_epochs':      retrain_epochs,
                'baseline':            {k: v for k, v in baseline_stats.items()},
                'sweep':               clean,
                'selected_idx':        best_idx,
                'selected_keep_ratio': sel['keep_ratio'],
            }, fp, indent=2)
        print(f'  Sweep results → {out_dir / "sweep_results.json"}')

    return sel['keep_ratio'], sel_m, {'sweep': results, 'selected_idx': best_idx,
                                      't_window_ms': t_window_ms}


# CLI  (server-side: Phases 1 + 2 only)

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            'Device-aware structured pruning: Phase 1 (L1 sparsity) + '
            'Phase 2 (accuracy-constrained incremental channel pruning).\n'
            'Output: pruned_model.pth  →  run device_finetune.py on the target device.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('dataset_dir', nargs='?', default=None,
        help='phase1 dataset folder (e.g. output/doppler_train/phase1). '
             'Optional when using --export_device_profile.')
    p.add_argument('--activities', default=None,
        help='Comma-separated gesture codes, e.g. FW,ND,SH. '
             'Default: read from best_model.pth label_dict (matches training automatically).')

    dev_group = p.add_argument_group(
        'Target Device',
        'Provide the deployment device so the script estimates inference latency\n'
        'and finds the best accuracy–speed tradeoff for that specific hardware.\n\n'
        'Workflow:\n'
        '  (A) Built-in device name (no hardware needed):\n'
        '        python3 sparse_finetune.py <dataset_dir> --target_device pi4_4gb\n'
        '  (B) Real hardware profile (most accurate):\n'
        '        # On the Pi:  python3 sparse_finetune.py --export_device_profile pi4.json\n'
        '        # On server:  python3 sparse_finetune.py <dataset_dir> --device_profile pi4.json\n'
    )
    dev_group.add_argument('--target_device', default=None,
        help='Built-in device name or "auto" to probe the current machine.  '
             'Built-in: ' + ', '.join(sorted(_DEFAULT_KNOWN_DEVICES)))
    dev_group.add_argument('--device_profile', default=None, metavar='PATH',
        help='Path to device_profile.json from --export_device_profile.')
    dev_group.add_argument('--devices_file', default=None, metavar='PATH',
        help='JSON of custom device definitions (merged into the built-in table).')
    dev_group.add_argument('--export_device_profile', default=None, metavar='PATH',
        help='Probe this machine, save JSON, then exit. Run ON the target device.')
    dev_group.add_argument('--benchmark', action='store_true',
        help='Measure actual CPU latency after pruning (on this server).')

    search_group = p.add_argument_group(
        'Incremental Pruning Search',
        'The script descends from keep_ratio=1.0, pruning and fine-tuning at each step,\n'
        'stopping when accuracy drops > --max_acc_drop.  The candidate with the best\n'
        'accuracy–latency tradeoff score is saved as pruned_model.pth.'
    )
    search_group.add_argument('--sweep_ratios', default=None,
        help='Comma-separated keep_ratio values to search, e.g. "0.33,0.50,0.67,0.75,1.00". '
             f'Default: {",".join(str(r) for r in _DEFAULT_SWEEP_RATIOS)}')
    search_group.add_argument('--retrain_epochs', type=int, default=20,
        help='Fine-tune epochs after each pruning step (0 = no fine-tuning). '
             '10 epochs is insufficient to recover after pruning active channels — use 20+.')
    search_group.add_argument('--max_acc_drop', type=float, default=0.06,
        help='Stop descending when accuracy drops more than this fraction from original '
             '(default 0.06 = 6 pp).')

    p.add_argument('--no_cache_ram', dest='cache_ram', action='store_false',
        help='Disable preloading all windows into RAM (use on low-memory '
             'machines; falls back to unpickling from disk per sample).')
    p.set_defaults(cache_ram=True)
    p.add_argument('--sparsity_epochs', type=int, default=30,
        help='Epochs for Phase 1 L1-BN sparsity retraining. '
             '15 epochs with λ=1e-4 is too weak — no channels go sparse. '
             'Use 30+ epochs with λ=5e-4 to drive channels toward 0.')
    p.add_argument('--l1_lambda', type=float, default=5e-4,
        help='L1 penalty weight on BN γ (Phase 1). '
             'λ=1e-4 contributes <2%% of total loss and fails to induce sparsity. '
             'λ=5e-4 is the minimum effective value for this model size.')
    p.add_argument('--model_dir', default=None, metavar='PATH',
        help='Folder containing best_model.pth (default: same as dataset_dir). '
             'Use this when the training script saves to a different directory '
             'than the dataset.')
    p.add_argument('--output_dir', default=None,
        help='Base output dir (default: <dataset_dir>/sparse_fewshot). '
             'A subfolder named by the device hostname is always appended when '
             'a device profile is provided, e.g. sparse_fewshot/raspberrypi/.')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# Main  (server-side)

def main():
    args = parse_args()

    known_devices = load_known_devices(args.devices_file)

    # --export_device_profile: probe this machine, save JSON, then exit
    if args.export_device_profile:
        profile  = probe_device()
        out_path = Path(args.export_device_profile)
        out_path.write_text(json.dumps(profile, indent=2))
        print('Device profile saved to:', out_path)
        print(json.dumps(profile, indent=2))
        print('\nNow on the training server:')
        print(f'  python3 sparse_finetune.py <dataset_dir> --device_profile {out_path.name}')
        return

    if args.dataset_dir is None:
        print('error: dataset_dir is required (or use --export_device_profile)')
        raise SystemExit(1)

    # Resolve device profile
    device_prof: Optional[dict] = None
    if args.device_profile:
        device_prof = json.loads(Path(args.device_profile).read_text())
    elif args.target_device:
        if args.target_device == 'auto':
            device_prof = probe_device()
        else:
            if args.target_device not in known_devices:
                print(f'error: unknown --target_device {args.target_device!r}.  '
                      f'Valid: auto, {", ".join(sorted(known_devices))}')
                raise SystemExit(1)
            device_prof = dict(known_devices[args.target_device])
            device_prof.setdefault('cpu_model',        args.target_device)
            device_prof.setdefault('cpu_cores',        4)
            device_prof.setdefault('ram_available_mb', device_prof.get('ram_mb', 2048))
            device_prof.setdefault('ram_total_mb',     device_prof.get('ram_mb', 2048))
            device_prof.setdefault('has_cuda',         False)

    # Setup
    set_seed(args.seed)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = Path(args.dataset_dir)
    out_dir  = Path(args.output_dir) if args.output_dir else base_dir / 'sparse_fewshot'
    hostname = (device_prof or {}).get('hostname', None)
    if hostname:
        out_dir = out_dir / hostname
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / 'run_log.txt'
    _tee = _Tee(log_path)
    _tee.__enter__()

    # Load metadata
    with open(base_dir / 'label_dict.pkl', 'rb') as f:
        label_dict = pickle.load(f)
    n_antennas = 1   # HeadGest is always single-antenna

    # Default: read activities from best_model.pth to match the trained model.
    if args.activities is not None:
        acts = [a.strip().upper() for a in args.activities.split(',')]
    else:
        ckpt_peek = torch_load_compat(base_dir / 'best_model.pth',
                                      map_location='cpu')
        ckpt_ld   = ckpt_peek.get('label_dict')
        if ckpt_ld:
            acts = sorted(ckpt_ld.keys(), key=lambda k: ckpt_ld[k])
            print(f'  [activities] auto-detected from best_model.pth: {acts}')
        else:
            acts = None
        del ckpt_peek

    train_files, train_labels = [], []
    for split in ('train', 'val'):
        fs, ls = load_split(base_dir, split)
        fs, ls, ld = filter_activities(fs, ls, label_dict, acts)
        train_files.extend(fs); train_labels.extend(ls)
    n_classes = len(ld)

    import pickle as _pkl
    with open(train_files[0], 'rb') as _f:
        _peek = _pkl.load(_f)
    actual_window_frames = int(_peek.shape[1])  # (n_ant, T, V) → T
    del _peek
    print(f'  Window frames  : {actual_window_frames}  '
          f'({actual_window_frames}/{_DOPPLER_FRAME_RATE_HZ:.0f} Hz '
          f'= {actual_window_frames/_DOPPLER_FRAME_RATE_HZ:.3f} s per window)')

    print('=' * 65)
    print('Device-Aware Structured Pruning  (Phases 1 + 2)')
    print('=' * 65)
    print(f'  Compute device : {device}')
    print(f'  Activities     : {ld}')
    granularity = _simd_granularity(device_prof)
    if device_prof:
        gflops = _gflops_for_profile(device_prof)
        arch_s = device_prof.get('arch') or device_prof.get('architecture', '?')
        print(f'  Target device  : {device_prof.get("cpu_model", "?")}  '
              f'({gflops:.2f} G-MACs/s,  {device_prof.get("ram_available_mb", "?")} MB RAM)')
        print(f'  SIMD granularity: {granularity}  (arch={arch_s})')
    else:
        print(f'  SIMD granularity: {granularity}  (default — no device profile)')
    print(f'  Accuracy budget: max {args.max_acc_drop*100:.0f}% drop from original')
    print(f'  Post-prune retrain: {args.retrain_epochs} epochs per step')

    # Load trained model
    model_dir      = Path(args.model_dir) if args.model_dir else base_dir
    ckpt_path      = model_dir / 'best_model.pth'
    ckpt           = torch_load_compat(ckpt_path, map_location=device)
    depth          = ckpt.get('depth', _infer_depth(ckpt['model_state']))
    n_classes_ckpt = ckpt.get('n_classes', n_classes)
    mlp_ratio      = ckpt.get('mlp_ratio', 2)
    attn_mlp_ratio = ckpt.get('attn_mlp_ratio', 1)
    simple_head    = ckpt.get('simple_head', True)
    mobile_stage3  = ckpt.get('mobile_stage3', False)
    hybrid_stage3  = ckpt.get('hybrid_stage3', False)
    print(f'  Model depth    : {depth}  (from checkpoint)')
    print(f'  mlp_ratio      : {mlp_ratio}  (RepBlock FFN expand, from checkpoint)')
    print(f'  attn_mlp_ratio : {attn_mlp_ratio}  (from checkpoint)')
    if n_classes_ckpt != n_classes:
        print(f'  [warn] --activities filter gives {n_classes} classes but checkpoint '
              f'has {n_classes_ckpt} — using checkpoint n_classes to load model.')
        n_classes = n_classes_ckpt
    # Channel widths: stored by newer checkpoints; for older ones read from weight shapes.
    if 'channels' in ckpt:
        channels = tuple(ckpt['channels'])
    else:
        _sd = ckpt['model_state']
        channels = (_sd['stem.0.weight'].shape[0],
                    _sd['down1.pw.0.weight'].shape[0],
                    _sd['down2.pw.0.weight'].shape[0])
        print(f'  [compat] channels auto-detected from state dict: {channels}')
    model = RepDopplerViT(n_classes=n_classes, in_channels=1,
                          channels=channels, depth=depth,
                          mlp_ratio=mlp_ratio,
                          attn_mlp_ratio=attn_mlp_ratio,
                          simple_head=simple_head,
                          mobile_stage3=mobile_stage3,
                          hybrid_stage3=hybrid_stage3).to(device)
    model.load_state_dict(ckpt['model_state'])
    print(f'\n  Loaded {ckpt_path}  (epoch {ckpt["epoch"]},  val_acc={ckpt["val_acc"]:.4f})')
    _print_model_summary(model)

    # Test loader
    test_files, test_labels = load_split(base_dir, 'test')
    test_files, test_labels, _ = filter_activities(test_files, test_labels, label_dict, acts)
    # Workers only pay off on CUDA; CPU-only devices do better with cache_ram.
    if device.type == 'cuda':
        n_workers = min(4, os.cpu_count() or 1)
    else:
        n_workers = 0 if args.cache_ram else min(2, os.cpu_count() or 1)
    test_ds     = DopplerDataset(test_files, test_labels, augment=False,
                                 expand=False, cache_ram=args.cache_ram)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=n_workers,
                             persistent_workers=(n_workers > 0))

    # Baseline profiling (before Phase 1)
    gflops_dev      = _gflops_for_profile(device_prof) if device_prof else None
    bl_params       = count_params(model)
    bl_macs_per_ant = count_model_macs(model, (1, 1, actual_window_frames, 100))
    bl_model_mb     = bl_params * 4 / 1024**2
    bl_full_macs_m  = bl_macs_per_ant / 1e6 * n_antennas
    bl_lat_ms: Optional[float] = (
        bl_full_macs_m / (gflops_dev * 1e3) * 1000 if gflops_dev else None)
    bl_t_window = _window_collection_ms(actual_window_frames)
    bl_cpu_pct: Optional[float] = (
        bl_lat_ms / bl_t_window * 100 if bl_lat_ms else None)
    bl_acc, bl_f1, _, _, _, _, _ = evaluate(
        model, test_loader, device)

    baseline_stats: dict = dict(
        acc          = bl_acc,
        f1           = bl_f1,
        macs_per_ant = bl_macs_per_ant,
        params       = bl_params,
        model_mb     = round(bl_model_mb, 3),
        lat_ms       = round(bl_lat_ms, 1)  if bl_lat_ms  else None,
        cpu_pct      = round(bl_cpu_pct, 1) if bl_cpu_pct else None,
        ram_available_mb = (device_prof or {}).get('ram_available_mb', 0),
    )

    # Phase 1: L1-BN sparsity retraining
    train_ds     = DopplerDataset(train_files, train_labels, augment=True,
                                  expand=True, cache_ram=args.cache_ram)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=n_workers,
                              pin_memory=(device.type == 'cuda'),
                              persistent_workers=(n_workers > 0))
    model = sparsity_retrain(model, train_loader, device, n_antennas,
                             epochs=args.sparsity_epochs, l1_lambda=args.l1_lambda)

    # Phase 2: Device-Aware Incremental Pruning Search
    candidates: Optional[List[float]] = None
    if args.sweep_ratios:
        try:
            candidates = sorted(float(r.strip()) for r in args.sweep_ratios.split(','))
        except ValueError:
            print(f'error: --sweep_ratios must be comma-separated floats, '
                  f'got: {args.sweep_ratios!r}')
            raise SystemExit(1)

    selected_ratio, pruned, sweep_data = incremental_prune_search(
        sparse_model   = model,
        test_loader    = test_loader,
        device         = device,
        n_classes      = n_classes,
        n_antennas     = n_antennas,
        baseline_stats = baseline_stats,
        device_prof    = device_prof,
        do_benchmark   = args.benchmark,
        candidates     = candidates,
        out_dir        = out_dir,
        train_loader   = train_loader,
        retrain_epochs = args.retrain_epochs,
        max_acc_drop   = args.max_acc_drop,
        granularity    = granularity,
        window_frames  = actual_window_frames,
    )

    best_idx   = sweep_data['selected_idx']
    pruned_acc = sweep_data['sweep'][best_idx]['acc']
    pruned_f1m = sweep_data['sweep'][best_idx]['f1']
    p_before   = baseline_stats['params']
    p_after    = count_params(pruned)
    # Use the channels stored by incremental_prune_search, not a re-computation.
    K1, K2, K3 = sweep_data['sweep'][best_idx]['channels']
    prune_stats = dict(
        K1=K1, K2=K2, K3=K3,
        params_before=p_before, params_after=p_after,
        mb_before=p_before * 4 / 1024**2,
        mb_after=p_after   * 4 / 1024**2,
        keep_ratio=selected_ratio,
    )

    # FLOPs of the final pruned model
    macs_pruned = report_model_flops(pruned, n_antennas, window_shape=(actual_window_frames, 100))

    # Optional post-hoc benchmark
    if args.benchmark:
        print(f'\n── Inference Benchmark  (CPU, {n_antennas} antennas) ──')
        bm = benchmark_inference(pruned, n_antennas=n_antennas,
                                 window_shape=(actual_window_frames, 100),
                                 device=torch.device('cpu'))
        print(f'  Latency: {bm["mean_ms"]:.1f} ± {bm["std_ms"]:.1f} ms  '
              f'(n={bm["n_runs"]} runs)')
        torch.save(bm, out_dir / 'benchmark.json')

    # Save pruned model
    save_path    = out_dir / 'pruned_model.pth'
    _hybrid_save = getattr(pruned, '_hybrid_stage3', False)
    _d3_save     = len(pruned.stage3_local) if _hybrid_save else len(pruned.stage3)
    torch.save({
        'model_state':      pruned.state_dict(),
        'channels':         (K1, K2, K3),
        'depth':            (len(pruned.stage1), len(pruned.stage2), _d3_save),
        'mlp_ratio':        getattr(pruned, '_mlp_ratio', 2),
        'attn_mlp_ratio':   getattr(pruned, '_attn_mlp_ratio', 1),
        'simple_head':      getattr(pruned, '_simple_head', True),
        'mobile_stage3':    getattr(pruned, '_mobile_stage3', False),
        'hybrid_stage3':    _hybrid_save,
        'n_classes':        n_classes,
        'n_antennas':       n_antennas,
        'label_dict':       ld,
        'prune_stats':      prune_stats,
        'device_prof':      device_prof,
        'macs_per_antenna': macs_pruned,
        'window_frames':    actual_window_frames,
    }, save_path)
    print(f'\n  Pruned model → {save_path}')

    # Final summary
    print('\n── Summary ──────────────────────────────────────────────────')
    print(f'  Original  : {p_before:,} params  ({p_before*4/1024**2:.2f} MB)')
    print(f'  Pruned    : {p_after:,} params  ({p_after*4/1024**2:.2f} MB)  '
          f'channels=({K1},{K2},{K3})')
    print(f'  S1a test  : acc={pruned_acc:.4f}  f1={pruned_f1m:.4f}')
    print(f'  Keep ratio: {selected_ratio:.2f}  '
          f'(Δacc = {(pruned_acc - bl_acc)*100:+.2f} pp from original)')
    print(f'  MACs/ant  : {macs_pruned/1e6:.1f} M  →  '
          f'{macs_pruned*2*n_antennas/1e6:.1f} M FLOPs per {n_antennas}-ant window')
    # Theoretical latency estimates, to compare later with device_finetune.py.
    if device_prof and gflops_dev:
        n_cls   = len(acts)
        t_win_s = actual_window_frames / _DOPPLER_FRAME_RATE_HZ

        t_inf_ms = macs_pruned * n_antennas / (gflops_dev * 1e9) * 1000

        # Forward + backward ~= 3x forward FLOPs; support samples are per-antenna.
        ft_flops_per_sample = 3 * 2 * macs_pruned
        default_epochs = 60

        print('\n── Theoretical Latency Estimates (target device) ──────────────')
        print(f'  Device          : {device_prof.get("cpu_model","?")} @ '
              f'{device_prof.get("cpu_freq_max_mhz","?")} MHz  '
              f'cores={device_prof.get("cpu_cores","?")}  '
              f'NEON={device_prof.get("has_neon","?")}')
        print(f'  Compute         : {gflops_dev:.2f} G-MACs/s')
        print(f'  Pruned model    : {macs_pruned/1e6:.1f} M MACs/ant  '
              f'({macs_pruned*n_antennas/1e6:.1f} M MACs/{n_antennas}-ant window)')
        print()
        print(f'  Inference (per window = {actual_window_frames}/{_DOPPLER_FRAME_RATE_HZ:.0f}Hz'
              f' = {t_win_s:.3f}s of radar data):')
        print(f'    Estimated wall time : {t_inf_ms:.2f} ms/window')
        print(f'    Throughput          : {1000/t_inf_ms:.0f} windows/s'
              f'  ({1000/t_inf_ms*t_win_s:.0f} s radar-data/s)')
        print(f'    Real-time factor    : {t_win_s*1000/t_inf_ms:.1f}×')
        print(f'    NOTE: single-core estimate; actual may differ ±2-3× due to')
        print(f'          cache effects, memory bandwidth, OS scheduling.')
        print()
        print(f'  Fine-tuning per trial ({default_epochs} epochs, head+stage3 for K≥10):')
        print(f'  {"K":>5}  {"support":>9}  '
              f'{"FLOPs/epoch":>12}  {"est/epoch":>10}  {"est/{:d}ep":>10}'.format(default_epochs))
        print(f'  ' + '─' * 54)
        for K in [5, 10, 20, 30, 40, 50]:
            n_sup      = K * n_cls * n_antennas          # expand mode (per-antenna samples)
            ft_ep      = n_sup * ft_flops_per_sample     # FLOPs
            t_ep_s     = ft_ep / (gflops_dev * 1e9)
            t_total_s  = t_ep_s * default_epochs
            print(f'  {K:>5}  {n_sup:>9}  {ft_ep/1e9:>10.2f}G'
                  f'  {t_ep_s:>8.2f}s  {t_total_s:>8.1f}s')
        print(f'  NOTE: compare these with Inf-Wall / FT-Wall in fewshot_results.txt')

    print(f'\n  Log → {log_path}')
    print(f'\nNext step:')
    print(f'  python3 device_finetune.py {save_path} \\')
    print(f'      --doppler_dir <doppler_dir> --scenario phase2 --activities {args.activities}')

    _tee.__exit__(None, None, None)


if __name__ == '__main__':
    main()
