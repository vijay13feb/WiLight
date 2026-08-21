#!/usr/bin/env python3
"""
train_doppler_vit_simwisense.py — RepDopplerViT for Doppler spectrogram HAR.
Processes save_classroom and save_office automatically. n_antennas=1, so each
window (1,T,V) is processed directly with no antenna expansion or fusion.

Loss is FocalLoss + supcon_weight * SupConLoss (Khosla et al. 2020) on the GAP
embeddings. Backbone: RepViT + EfficientViT (CVPR 2024); reparameterize()
fuses to a single 3x3 DW conv per block for ONNX export. 
"""

import argparse
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report,
)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[warn] psutil not found — RAM/CPU monitoring disabled.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[warn] matplotlib/seaborn not found — plots skipped.")

# Paths

SCRIPT_DIR  = Path(__file__).resolve().parent
INPUT_BASE  = SCRIPT_DIR / 'output_data' / 'doppler_train'
OUTPUT_BASE = SCRIPT_DIR / 'output_data' / 'doppler_test'
SCENARIOS   = ['save_classroom', 'save_office']

# Activity map

ACTIVITY_MAP = {
    'A': 'Push_forward',
    'B': 'Rotate',
    'C': 'Hands_up_and_down',
    'D': 'Waive',
    'E': 'Brush',
    'F': 'Clap',
    'G': 'Sit',
    'H': 'Eat',
    'I': 'Drink',
    'J': 'Kick',
    'K': 'Bend_forward',
    'L': 'Wash_hands',
    'M': 'Call',
    'N': 'Browsing_phone',
    'O': 'Check_wrist',
    'P': 'Read',
    'Q': 'Waive_while_sitting',
    'R': 'Writing',
    'S': 'Side_bend',
    'T': 'Standing'
}

# Fixed alphabetical groups of 5 for confusion matrix splitting
CM_GROUPS = [
    ['A', 'B', 'C', 'D', 'E'],
    ['F', 'G', 'H', 'I', 'J'],
    ['K', 'L', 'M', 'N', 'O'],
    ['P', 'Q', 'R', 'S', 'T'],
]

# n_antennas is always 1 for this dataset
N_ANTENNAS = 1


# Reproducibility

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_compat(path, map_location):
    """torch.load compat: weights_only kwarg needs torch >=1.13."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# Dataset

class DopplerDataset(Dataset):
    """Loads (1, T, V) windows saved by create_train_simwisense.py; n_antennas=1
    so expand/no-expand modes both return (1, T, V) with no antenna expansion."""

    def __init__(self, file_list: List[str], label_list: List[int],
                 augment: bool = False, expand: bool = True,
                 cache_ram: bool = True):
        self.files   = file_list
        self.labels  = label_list
        self.augment = augment
        self.expand  = expand
        # Preload once in the main process so forked workers share via copy-on-write.
        self._cache: Optional[List[np.ndarray]] = None
        if cache_ram:
            self._cache = [self._load_file(i) for i in range(len(file_list))]
            n_bytes = sum(a.nbytes for a in self._cache)
            print(f'[dataset] cached {len(self._cache)} windows '
                  f'in RAM ({n_bytes / 1024**2:.0f} MB)')

    def _load_file(self, idx: int) -> np.ndarray:
        with open(self.files[idx], 'rb') as fp:
            return pickle.load(fp).astype(np.float32)   # (1, T, V)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self._cache is not None:
            x = self._cache[idx]
        else:
            x = self._load_file(idx)
        x = torch.from_numpy(x)   # zero-copy view; cached array must not be mutated in place

        if self.augment:
            x = torch.roll(x, random.randint(-8, 8), dims=1)
            v0 = random.randint(0, 85)
            x[:, :, v0 : v0 + random.randint(0, 15)] = 0.0
            t0 = random.randint(0, max(0, x.shape[1] - 8))
            x[:, t0 : t0 + random.randint(0, 8), :] = 0.0
            x = x * random.uniform(0.85, 1.15)
            x = (x + 0.005 * torch.randn_like(x)).clamp(0.0, 1.0)

        return x, self.labels[idx]


def _load_pkl(path: Path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def load_split(base: Path, split: str) -> Tuple[List, List]:
    return (_load_pkl(base / f'files_{split}.pkl'),
            _load_pkl(base / f'labels_{split}.pkl'))


def filter_activities(
    files: List[str], labels: List[int],
    label_dict: dict, activities: Optional[List[str]],
) -> Tuple[List[str], List[int], dict]:
    if activities is None:
        return files, labels, label_dict

    keep = sorted(
        [(letter, label_dict[letter]) for letter in activities if letter in label_dict],
        key=lambda t: t[1],
    )
    if not keep:
        raise ValueError(f"None of {activities} found in {label_dict}.")

    missing = [a for a in activities if a not in label_dict]
    if missing:
        print(f"[warn] activities {missing} not found — ignored.")

    old_to_new = {old: new for new, (_, old) in enumerate(keep)}
    new_ld     = {letter: old_to_new[old] for (letter, old) in keep}

    pairs = [(f, old_to_new[l]) for f, l in zip(files, labels) if l in old_to_new]
    if not pairs:
        raise RuntimeError("Filter left zero samples.")
    new_files, new_labels = zip(*pairs)
    return list(new_files), list(new_labels), new_ld


# Loss 1 — Focal Loss

def _cross_entropy_smooth(logits: Tensor, targets: Tensor,
                           smoothing: float, reduction: str = 'mean') -> Tensor:
    """Label-smoothed cross-entropy, compatible with PyTorch < 1.10."""
    log_p = F.log_softmax(logits, dim=-1)
    nll   = -log_p.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    smooth_loss = -log_p.mean(dim=-1)
    loss = (1.0 - smoothing) * nll + smoothing * smooth_loss
    if reduction == 'none':
        return loss
    return loss.mean()


class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., ICCV 2017)."""
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce  = _cross_entropy_smooth(logits, targets,
                                    self.label_smoothing, reduction='none')
        pt  = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# Loss 2 — Supervised Contrastive Loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020) on GAP features."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        B  = features.shape[0]
        f  = F.normalize(features, dim=1)          # (B, D)
        sim = (f @ f.T) / self.temperature          # (B, B)

        lbl = labels.view(-1, 1)
        same_class = (lbl == lbl.T).float()         # (B, B)
        eye        = torch.eye(B, device=f.device)
        pos_mask   = same_class - eye               # exclude diagonal

        sim = sim - sim.max(dim=1, keepdim=True).values
        exp_sim = torch.exp(sim) * (1.0 - eye)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        n_pos = pos_mask.sum(dim=1).clamp(min=1.0)
        loss  = -(pos_mask * log_prob).sum(dim=1) / n_pos
        return loss.mean()


# Model components  (RepViT + EfficientViT, CVPR 2024)

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep  = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.floor(
            torch.rand(shape, dtype=x.dtype, device=x.device) + keep) / keep
        return x * noise


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    W     = conv.weight.data.clone()
    scale = bn.weight.data / (bn.running_var + bn.eps).sqrt()
    W_f   = W * scale.view(-1, 1, 1, 1)
    b_f   = bn.bias.data - bn.running_mean * scale
    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        conv.stride, conv.padding, groups=conv.groups, bias=True,
    )
    fused.weight.data = W_f
    fused.bias.data   = b_f
    return fused


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3,
                 stride: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),   # ReLU: XNNPACK fuses Conv2d+BN+ReLU into one kernel on ARM
        )


class RepDWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim    = dim
        self._fused: Optional[nn.Conv2d] = None
        self.br3    = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.br1    = nn.Sequential(
            nn.Conv2d(dim, dim, 1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.br_id  = nn.BatchNorm2d(dim)

    def _branch_kernel(self, conv, bn):
        fused = _fuse_conv_bn(conv, bn)
        W     = fused.weight.data
        if W.shape[-1] == 1:
            W = F.pad(W, [1, 1, 1, 1])
        return W, fused.bias.data

    def _identity_kernel(self):
        bn  = self.br_id
        dev = bn.weight.device
        W_id = torch.zeros(self.dim, 1, 3, 3, device=dev)
        W_id[:, 0, 1, 1] = 1.0
        id_conv = nn.Conv2d(
            self.dim, self.dim, 3, padding=1, groups=self.dim, bias=False).to(dev)
        id_conv.weight.data = W_id
        return self._branch_kernel(id_conv, bn)

    def reparameterize(self):
        if self._fused is not None:
            return
        W3, b3 = self._branch_kernel(self.br3[0], self.br3[1])
        W1, b1 = self._branch_kernel(self.br1[0], self.br1[1])
        Wi, bi = self._identity_kernel()
        fused  = nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim, bias=True)
        fused.weight.data = W3 + W1 + Wi
        fused.bias.data   = b3 + b1 + bi
        self._fused = fused.to(W3.device)
        del self.br3, self.br1, self.br_id

    def forward(self, x: Tensor) -> Tensor:
        if self._fused is not None:
            return self._fused(x)
        return self.br3(x) + self.br1(x) + self.br_id(x)


class RepBlock(nn.Module):
    def __init__(self, dim: int, expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.token_mixer   = RepDWConv(dim)
        self.channel_mixer = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim * expand, 1, bias=False),
            nn.ReLU(),   # XNNPACK-fused with the preceding Conv2d on ARM
            nn.Conv2d(dim * expand, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.drop_path = DropPath(drop_path)

    def reparameterize(self):
        self.token_mixer.reparameterize()

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_mixer(x)
        x = x + self.drop_path(self.channel_mixer(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch,  k=3, stride=2, groups=in_ch)
        self.pw = ConvBNAct(in_ch, out_ch, k=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pw(self.dw(x))


def _relu_linear_attn(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    q = F.relu(q) + 1e-6
    k = F.relu(k) + 1e-6
    kv   = torch.einsum('bhnk, bhnv -> bhkv', k, v)
    out  = torch.einsum('bhnk, bhkv -> bhnv', q, kv)
    norm = (q * k.sum(dim=2, keepdim=True)).sum(-1, keepdim=True).clamp(1e-6)
    return out / norm


class LinearAttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 16,
                 mlp_ratio: int = 2, dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        inner         = heads * dim_head
        self.heads    = heads
        self.dim_head = dim_head
        self.local_dw = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),   # XNNPACK-fused
        )
        self.norm_attn = nn.BatchNorm2d(dim)
        self.q         = nn.Conv2d(dim, inner, 1, bias=False)
        self.k         = nn.Conv2d(dim, inner, 1, bias=False)
        self.v         = nn.Conv2d(dim, inner, 1, bias=False)
        self.attn_proj = nn.Sequential(
            nn.Conv2d(inner, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.attn_drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.ffn = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim * mlp_ratio, 1, bias=False),
            nn.ReLU(),   # XNNPACK-fused
            nn.Conv2d(dim * mlp_ratio, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        N = H * W
        x = x + self.drop_path(self.local_dw(x))
        xn = self.norm_attn(x)

        def _to_seq(t: Tensor) -> Tensor:
            return t.reshape(B, self.heads, self.dim_head, N).permute(0, 1, 3, 2)

        q, k, v  = _to_seq(self.q(xn)), _to_seq(self.k(xn)), _to_seq(self.v(xn))
        attn_out = _relu_linear_attn(q, k, v)
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        x = x + self.drop_path(self.attn_drop(self.attn_proj(attn_out)))
        x = x + self.drop_path(self.ffn(x))
        return x


# XNNPACK-friendly stage3 block  (replaces LinearAttnBlock in mobile_stage3 mode)

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck — pure XNNPACK path for stage3."""
    def __init__(self, dim: int, expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        mid = dim * expand
        self.dw_mix = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.pw_exp = nn.Sequential(
            nn.Conv2d(dim, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(),
        )
        self.pw_proj = nn.Sequential(
            nn.Conv2d(mid, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.dw_mix(x))
        shortcut = x
        x = self.pw_proj(self.pw_exp(x))
        return shortcut + self.drop_path(x)


# Full model  — in_channels=1

class RepDopplerViT(nn.Module):
    """
    RepDopplerViT.

    Input  : (B, 1, T, V)
    Output : (B, n_classes) logits

    ≈ 182 K parameters (in_channels=1).
    """

    def __init__(self, n_classes: int,
                 in_channels: int = 1,
                 channels: Tuple[int, int, int] = (16, 32, 64),
                 depth: Tuple[int, int, int] = (1, 1, 1),
                 heads: int = 4, dim_head: int = 16,
                 mlp_ratio: int = 2,
                 attn_mlp_ratio: int = 1, dropout: float = 0.2,
                 drop_path_rate: float = 0.1,
                 simple_head: bool = False,
                 mobile_stage3: bool = False,
                 hybrid_stage3: bool = True):
        # hybrid_stage3 downsamples to 22x6 (132 tokens) before MBConv+LinearAttn,
        # and takes precedence over mobile_stage3 when both are True.
        super().__init__()
        self._simple_head    = simple_head
        self._mlp_ratio      = mlp_ratio        # RepBlock FFN expand (stage1/stage2)
        self._attn_mlp_ratio = attn_mlp_ratio   # LinearAttnBlock FFN expand (stage3)
        self._mobile_stage3  = mobile_stage3
        self._hybrid_stage3  = hybrid_stage3
        c1, c2, c3 = channels
        d1, d2, d3 = depth
        total = d1 + d2 + d3
        dp    = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        idx   = 0

        self.stem   = ConvBNAct(in_channels, c1, k=3)

        self.stage1 = nn.Sequential(*[
            RepBlock(c1, mlp_ratio, drop_path=dp[idx + i]) for i in range(d1)
        ]); idx += d1
        self.down1  = Downsample(c1, c2)

        self.stage2 = nn.Sequential(*[
            RepBlock(c2, mlp_ratio, drop_path=dp[idx + i]) for i in range(d2)
        ]); idx += d2
        self.down2  = Downsample(c2, c3)

        if hybrid_stage3:
            self.down_pre = nn.Sequential(
                Downsample(c3, c3),
                Downsample(c3, c3),
                Downsample(c3, c3),
            )
            self.stage3_local = nn.Sequential(*[
                MBConvBlock(c3, expand=max(attn_mlp_ratio, 2),
                            drop_path=dp[idx + i])
                for i in range(d3)
            ])
            self.stage3_attn = nn.Sequential(*[
                LinearAttnBlock(c3, heads, dim_head, attn_mlp_ratio, dropout,
                                drop_path=dp[idx + i])
                for i in range(d3)
            ])
        elif mobile_stage3:
            self.stage3 = nn.Sequential(*[
                MBConvBlock(c3, expand=max(attn_mlp_ratio, 2),
                            drop_path=dp[idx + i])
                for i in range(d3)
            ])
        else:
            self.stage3 = nn.Sequential(*[
                LinearAttnBlock(c3, heads, dim_head, attn_mlp_ratio, dropout,
                                drop_path=dp[idx + i])
                for i in range(d3)
            ])

        if simple_head:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.BatchNorm1d(c3),
                nn.Dropout(dropout),
                nn.Linear(c3, n_classes),
            )
        else:
            self.head = nn.Sequential(
                ConvBNAct(c3, c3 * 2, k=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(c3 * 2, n_classes),
            )

    def reparameterize(self) -> 'RepDopplerViT':
        for m in self.modules():
            if isinstance(m, RepBlock):
                m.reparameterize()
        return self

    def forward(self, x: Tensor,
                return_features: bool = False) -> Tensor:
        x = self.stem(x)
        x = self.down1(self.stage1(x))
        x = self.down2(self.stage2(x))
        if self._hybrid_stage3:
            x = self.down_pre(x)
            x = self.stage3_local(x)
            x = self.stage3_attn(x)
        else:
            x = self.stage3(x)
        # Features captured at head[2]: BN1d output (simple_head) or Flatten output.
        feat_idx = 2
        for i, layer in enumerate(self.head):
            x = layer(x)
            if return_features and i == feat_idx:
                feat = x   # GAP embedding
        if return_features:
            return x, feat   # x = logits
        return x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_reparameterize(model: RepDopplerViT, sample_input: Tensor,
                          tol: float = 1e-3) -> float:
    model.eval()
    with torch.no_grad():
        out_before = model(sample_input).clone()
        model.reparameterize()
        out_after  = model(sample_input)
    diff = (out_before - out_after).abs().max().item()
    status = "OK" if diff < tol else "MISMATCH"
    print(f"  Reparameterize check: max Δ = {diff:.2e}  [{status}]")
    return diff


# System / inference profiling

def _mb(n): return n / 1024 ** 2


def print_system_info(model, device):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes   = sum(b.numel() * b.element_size() for b in model.buffers())
    print('\n' + '─' * 65)
    print('SYSTEM INFO')
    print('─' * 65)
    if HAS_PSUTIL:
        proc      = psutil.Process()
        ram_used  = _mb(proc.memory_info().rss)
        ram_total = _mb(psutil.virtual_memory().total)
        print(f'  RAM total     : {ram_total:.1f} MB')
        print(f'  RAM used now  : {ram_used:.1f} MB')
    else:
        print('  RAM           : psutil not available')
    if device.type == 'cuda':
        vram_total = _mb(torch.cuda.get_device_properties(device).total_memory)
        vram_free  = (_mb(torch.cuda.mem_get_info(device.index or 0)[0])
                      if hasattr(torch.cuda, 'mem_get_info') else 0.0)  # ≥1.10
        print(f'  GPU           : {torch.cuda.get_device_name(device)}')
        print(f'  VRAM total    : {vram_total:.0f} MB  |  free: {vram_free:.0f} MB')
    print(f'  Model params  : {count_params(model):,}')
    print(f'  Model size    : {_mb(param_bytes + buf_bytes):.2f} MB')
    print('─' * 65)


# Training helpers

def run_epoch(model, loader, focal_crit, supcon_crit,
              supcon_w, optimizer, device, train: bool):
    """Single training or validation epoch. n_antennas=1, no antenna merging."""
    model.train(train)
    total_loss = total_correct = total_n = 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            B    = x.shape[0]

            logits, feats = model(x, return_features=True)
            focal  = focal_crit(logits, y)
            supcon = supcon_crit(feats, y)
            loss   = focal + supcon_w * supcon

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_correct += (logits.argmax(1) == y).sum().item()
            total_loss    += loss.item() * B
            total_n       += B

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (accuracy, macro-F1, weighted-F1, per-class-F1, conf-mat, preds, targets)."""
    model.eval()
    all_preds, all_targets = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)                         # (B, C)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_targets.extend(y.numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1m = f1_score(all_targets, all_preds, average='macro',    zero_division=0)
    f1w = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1c = f1_score(all_targets, all_preds, average=None,       zero_division=0)
    cm  = confusion_matrix(all_targets, all_preds)
    return acc, f1m, f1w, f1c, cm, all_preds, all_targets


# Plots

def plot_curves(history, path):
    if not HAS_PLOT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes, ['loss', 'acc'], ['Loss', 'Accuracy']):
        ax.plot(history[f'train_{key}'], label='Train')
        ax.plot(history[f'val_{key}'],   label='Val')
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Curves      → {path}")


def plot_confusion_matrix_group(cm_group, class_names_group, group_label, path):
    """Plot one confusion matrix for a subset of activities."""
    if not HAS_PLOT:
        return
    n = len(class_names_group)
    cm_norm = cm_group.astype(float) / cm_group.sum(axis=1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n)))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names_group, yticklabels=class_names_group,
                ax=ax, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j + 0.5, i + 0.72, f'({cm_group[i,j]})',
                    ha='center', va='center', fontsize=8, color='grey')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {group_label} (row-normalised, raw count in grey)')
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Confusion [{group_label}] → {path}")


def plot_all_confusion_matrices(cm, final_ld, idx_to_act, out_dir, scenario):
    """Split the full confusion matrix into the 4 fixed groups in CM_GROUPS,
    showing only activities present in this run."""
    if not HAS_PLOT:
        return

    for g_idx, group_letters in enumerate(CM_GROUPS):
        present = [(letter, final_ld[letter])
                   for letter in group_letters if letter in final_ld]

        if not present:
            continue

        group_label = f'{group_letters[0]}-{group_letters[-1]}'
        label_indices = [idx for (_, idx) in present]
        class_names_group = [
            f"{letter}\n{ACTIVITY_MAP.get(letter, letter)}"
            for (letter, _) in present
        ]

        cm_group = cm[np.ix_(label_indices, label_indices)]

        fname = f'confusion_matrix_group{g_idx+1}_{group_label}_{scenario}.png'
        plot_confusion_matrix_group(
            cm_group, class_names_group, group_label,
            out_dir / fname
        )


# Per-scenario training

def train_scenario(scenario: str, args):
    base_dir = INPUT_BASE / scenario
    out_dir  = OUTPUT_BASE / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '=' * 65)
    print(f'SCENARIO: {scenario}')
    print('=' * 65)

    if not base_dir.exists():
        print(f'  [warn] Dataset folder not found: {base_dir} — skipping.')
        return

    label_dict = _load_pkl(base_dir / 'label_dict.pkl')
    act_filter = (
        [a.strip().upper() for a in args.activities.split(',')]
        if args.activities else None
    )

    splits = {}
    ld_filtered = None
    for split in ('train', 'val', 'test'):
        files, labels = load_split(base_dir, split)
        files, labels, ld_filtered = filter_activities(
            files, labels, label_dict, act_filter)
        aug    = (split == 'train') and args.augment
        expand = (split == 'train')
        splits[split] = DopplerDataset(files, labels, augment=aug,
                                       expand=expand, cache_ram=args.cache_ram)

    final_ld    = ld_filtered
    n_classes   = len(final_ld)
    idx_to_act  = {v: k for k, v in final_ld.items()}
    class_names = [
        f"{idx_to_act[i]} ({ACTIVITY_MAP.get(idx_to_act[i], '?')})"
        for i in range(n_classes)
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'  Dataset     : {base_dir}')
    print(f'  Device      : {device}')
    print(f'  n_antennas  : {N_ANTENNAS}  (hardcoded, windows are (1, T, V))')
    print(f'  Classes     : {n_classes}  →  {final_ld}')
    n_tr = len(splits['train'].labels)
    n_va = len(splits['val'].labels)
    n_te = len(splits['test'].labels)
    print(f'  Windows     : train={n_tr}  val={n_va}  test={n_te}')
    print(f'  SupCon w    : {args.supcon_weight}  |  temp: {args.supcon_temp}')

    channels       = (16, 32, 64)
    depth          = (1, 1, 1)
    mlp_ratio      = 2
    attn_mlp_ratio = 1
    model = RepDopplerViT(
        n_classes=n_classes, in_channels=1,
        channels=channels, depth=depth,
        mlp_ratio=mlp_ratio,
        attn_mlp_ratio=attn_mlp_ratio,
        simple_head=args.simple_head,
        mobile_stage3=args.mobile_stage3,
        hybrid_stage3=args.hybrid_stage3,
    ).to(device)
    print(f'  Parameters  : {count_params(model):,}  (in_channels=1)')
    print_system_info(model, device)

    # Workers only pay off on CUDA; CPU-only devices do better with cache_ram.
    if device.type == 'cuda':
        n_workers = min(4, os.cpu_count() or 1)
    else:
        n_workers = 0 if args.cache_ram else min(2, os.cpu_count() or 1)
    loaders = {
        s: DataLoader(
            splits[s],
            batch_size         = args.batch_size,
            shuffle            = (s == 'train'),
            num_workers        = n_workers,
            pin_memory         = (device.type == 'cuda'),
            persistent_workers = (n_workers > 0),
        )
        for s in ('train', 'val', 'test')
    }

    focal_crit  = FocalLoss(gamma=args.gamma,
                            label_smoothing=args.label_smoothing)
    supcon_crit = SupConLoss(temperature=args.supcon_temp)
    optimizer   = torch.optim.AdamW(model.parameters(),
                                    lr=args.lr, weight_decay=args.weight_decay)
    # LambdaLR implements warmup→cosine without LinearLR/SequentialLR (≥1.11),
    # so the script also runs on PyTorch 1.8 (Jetson Nano JetPack 4).
    _warm = args.warmup_epochs
    _eta_ratio = 1e-6 / max(args.lr, 1e-12)
    def _lr_lambda(epoch: int) -> float:
        if epoch < _warm:
            return 0.1 + 0.9 * epoch / max(_warm, 1)
        t = (epoch - _warm) / max(args.epochs - _warm, 1)
        return _eta_ratio + (1.0 - _eta_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    print('\n' + '-' * 65)
    print(f'{"Ep":>4}  {"TrLoss":>8}  {"TrAcc":>7}  '
          f'{"VaLoss":>8}  {"VaAcc":>7}  {"LR":>9}')
    print('-' * 65)

    history      = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
    best_val_acc  = 0.0
    best_val_loss = float('inf')
    best_epoch    = 0
    no_improve    = 0
    ckpt_path     = out_dir / 'best_model.pth'

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(
            model, loaders['train'], focal_crit, supcon_crit,
            args.supcon_weight, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(
            model, loaders['val'], focal_crit, supcon_crit,
            args.supcon_weight, optimizer, device, train=False)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        lr_now = scheduler.get_last_lr()[0]
        # Tiebreak on val_loss so a saturated val_acc keeps tracking convergence.
        improved = (va_acc > best_val_acc) or (
            va_acc == best_val_acc and va_loss < best_val_loss)
        star = '*' if improved else ' '
        print(f'{ep:4d}  {tr_loss:8.4f}  {tr_acc:7.4f}  '
              f'{va_loss:8.4f}  {va_acc:7.4f}  {lr_now:9.2e}  '
              f'({time.time()-t0:.1f}s) {star}')

        if improved:
            best_val_acc  = va_acc
            best_val_loss = va_loss
            best_epoch    = ep
            no_improve    = 0
            _hybrid = getattr(model, '_hybrid_stage3', False)
            _d3_len = len(model.stage3_local if _hybrid else model.stage3)
            torch.save({
                'epoch':          ep,
                'model_state':    model.state_dict(),
                'val_acc':        va_acc,
                'label_dict':     final_ld,
                'n_classes':      n_classes,
                'n_antennas':     N_ANTENNAS,
                'channels':       channels,
                'depth':          (len(model.stage1), len(model.stage2), _d3_len),
                'mlp_ratio':      mlp_ratio,
                'attn_mlp_ratio': attn_mlp_ratio,
                'simple_head':    getattr(model, '_simple_head', True),
                'mobile_stage3':  getattr(model, '_mobile_stage3', False),
                'hybrid_stage3':  getattr(model, '_hybrid_stage3', False),
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f'\n  Early stop — no val improvement for {args.patience} epochs.')
                break

    print('-' * 65)
    print(f'  Best val acc : {best_val_acc:.4f}  (epoch {best_epoch})')

    # test evaluation
    ckpt = torch_load_compat(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f'\n  Loaded best checkpoint (epoch {ckpt["epoch"]})')

    acc, f1m, f1w, f1c, cm, preds, targets = evaluate(
        model, loaders['test'], device)

    print('\n' + '=' * 65)
    print(f'TEST RESULTS  [{scenario}]')
    print('=' * 65)
    print(f'  Accuracy       : {acc:.4f}   ({acc*100:.2f} %)')
    print(f'  F1  macro      : {f1m:.4f}')
    print(f'  F1  weighted   : {f1w:.4f}')
    print('\nPer-class F1:')
    for i, name in enumerate(class_names):
        print(f'    [{name:35s}]  F1 = {f1c[i]:.4f}')
    print('\nClassification Report:')
    print(classification_report(targets, preds,
                                 target_names=class_names, zero_division=0))
    print('Confusion Matrix (raw counts):')
    print(cm)

    # reparameterize + save fused model
    print('\n  Verifying reparameterization …')
    sample     = next(iter(loaders['test']))[0][:2].to(device)
    verify_reparameterize(model, sample)

    fused_path = out_dir / 'best_model_fused.pth'
    _hybrid = getattr(model, '_hybrid_stage3', False)
    _d3_len = len(model.stage3_local if _hybrid else model.stage3)
    torch.save({
        'epoch':           ckpt['epoch'],
        'model_state':     model.state_dict(),
        'val_acc':         ckpt['val_acc'],
        'label_dict':      final_ld,
        'n_classes':       n_classes,
        'n_antennas':      N_ANTENNAS,
        'channels':        channels,
        'depth':           (len(model.stage1), len(model.stage2), _d3_len),
        'mlp_ratio':       mlp_ratio,
        'attn_mlp_ratio':  attn_mlp_ratio,
        'simple_head':     getattr(model, '_simple_head', True),
        'mobile_stage3':   getattr(model, '_mobile_stage3', False),
        'hybrid_stage3':   getattr(model, '_hybrid_stage3', False),
        'reparameterized': True,
        'scenario':        scenario,
    }, fused_path)
    print(f'  Fused model → {fused_path}')

    # save results
    res_path = out_dir / 'test_results.txt'
    with open(res_path, 'w') as fp:
        fp.write(f'Model             : RepDopplerViT\n')
        fp.write(f'Scenario          : {scenario}\n')
        fp.write(f'Dataset           : {base_dir}\n')
        fp.write(f'Activities        : {final_ld}\n')
        fp.write(f'Parameters        : {count_params(model):,}\n')
        fp.write(f'Best epoch        : {ckpt["epoch"]}\n')
        fp.write(f'Accuracy          : {acc:.4f}\n')
        fp.write(f'F1 macro          : {f1m:.4f}\n')
        fp.write(f'F1 weighted       : {f1w:.4f}\n\n')
        fp.write(classification_report(targets, preds,
                                        target_names=class_names, zero_division=0))
        fp.write('\nConfusion matrix (raw counts):\n')
        fp.write(str(cm) + '\n')
    print(f'  Results     → {res_path}')

    # plots
    plot_curves(history, out_dir / 'training_curves.png')

    # 4 separate confusion matrices — fixed groups A–E, F–J, K–O, P–T
    plot_all_confusion_matrices(cm, final_ld, idx_to_act, out_dir, scenario)

    print(f'\n  Done — {scenario}')


# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description='Train RepDopplerViT on Doppler spectrogram dataset. '
                    'Automatically processes save_classroom and save_office.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--activities', default=None,
        help='Comma-separated activity letters, e.g. A,C,D,F,G. '
             'Default: all activities found in label_dict.')
    p.add_argument('--scenarios', default='all',
        help='Comma-separated scenarios, e.g. save_classroom,save_office. '
             'Use "all" for both.')
    p.add_argument('--epochs',          type=int,   default=100)
    p.add_argument('--batch_size',      type=int,   default=64)
    p.add_argument('--lr',              type=float, default=5e-4)
    p.add_argument('--warmup_epochs',   type=int,   default=5)
    p.add_argument('--weight_decay',    type=float, default=1e-4)
    p.add_argument('--gamma',           type=float, default=2.0,
        help='Focal loss γ.')
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--supcon_weight',   type=float, default=0.1,
        help='Weight of SupCon loss relative to Focal loss.')
    p.add_argument('--supcon_temp',     type=float, default=0.07,
        help='SupCon temperature.')
    p.add_argument('--patience',        type=int,   default=15)
    p.add_argument('--augment',         action='store_true')
    p.add_argument('--mobile_stage3',   action='store_true',
        help='Replace stage3 LinearAttnBlock with MBConvBlock (pure XNNPACK path). Requires retraining.')
    p.add_argument('--no_hybrid_stage3', dest='hybrid_stage3', action='store_false',
        help='Disable hybrid stage3; use plain LinearAttnBlock. Default: hybrid_stage3=True (Downsample→MBConv→LinearAttn).')
    p.set_defaults(hybrid_stage3=True)
    p.add_argument('--no_simple_head',  dest='simple_head', action='store_false',
        help='Revert to original ConvBNAct head. Default: simple_head=True (GAP→BN→Dropout→Linear, enables feature cache).')
    p.set_defaults(simple_head=True)
    p.add_argument('--no_cache_ram', dest='cache_ram', action='store_false',
        help='Disable preloading all windows into RAM (use on low-memory '
             'machines; falls back to unpickling from disk per sample).')
    p.set_defaults(cache_ram=True)
    p.add_argument('--seed',            type=int,   default=42)
    return p.parse_args()


# Main

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.scenarios.lower() == 'all':
        scenarios = SCENARIOS
    else:
        scenarios = [s.strip() for s in args.scenarios.split(',')]

    print('=' * 65)
    print('RepDopplerViT  —  (RepViT + EfficientViT, CVPR 2024)')
    print('=' * 65)
    print(f'  Input base  : {INPUT_BASE}')
    print(f'  Output base : {OUTPUT_BASE}')
    print(f'  Scenarios   : {scenarios}')
    print(f'  n_antennas  : {N_ANTENNAS}  (hardcoded)')

    for scenario in scenarios:
        train_scenario(scenario, args)

    print('\n' + '=' * 65)
    print('All scenarios complete.')
    print('=' * 65)


if __name__ == '__main__':
    main()