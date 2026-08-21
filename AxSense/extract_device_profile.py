#!/usr/bin/env python3
"""
extract_device_profile.py  —  Run on the TARGET DEVICE to collect hardware info.

Usage
-----
  python3 extract_device_profile.py                      # saves device_profile.json
  python3 extract_device_profile.py my_pi4.json          # saves to custom name

Then copy the JSON back to your training server and use:
  python3 sparse_finetune.py output/doppler/S1a \\
      --device_profile my_pi4.json \\
      --activities A,C,D,F,G

Requirements
------------
  pip install psutil        # needed for CPU freq + RAM info
  pip install torch         # needed for CUDA detection

The script works WITHOUT psutil (freq/RAM fields will be 0/null).
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print('[warn] psutil not found — install with: pip install psutil')
    print('       CPU frequency and RAM info will be missing.\n')

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print('[warn] torch not found — CUDA detection skipped.\n')


def _detect_neon() -> bool:
    """True if this ARM CPU has NEON / Advanced SIMD support."""
    try:
        with open('/proc/cpuinfo') as fh:
            info = fh.read().lower()
        return any(f in info for f in ('asimd', 'neon'))
    except OSError:
        return False


def _cpu_model() -> str:
    """Best-effort CPU name from /proc/cpuinfo or platform.uname()."""
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                tag, _, val = line.partition(':')
                if tag.strip().lower() in ('model name', 'hardware', 'cpu model'):
                    return val.strip()
    except OSError:
        pass
    return platform.uname().processor or platform.uname().machine


def collect_profile() -> dict:
    uname = platform.uname()
    arch  = uname.machine   # 'aarch64', 'armv6l', 'armv7l', 'x86_64', …

    profile = {
        'hostname':             uname.node,
        'platform':             uname.system.lower(),
        'arch':                 arch,
        'cpu_model':            _cpu_model(),
        'has_neon':             _detect_neon(),
        'has_cuda':             (torch.cuda.is_available() if HAS_TORCH else False),
    }

    # CPU core count
    try:
        profile['cpu_cores'] = os.cpu_count() or 1
    except Exception:
        profile['cpu_cores'] = 1

    # psutil: frequency + RAM
    if HAS_PSUTIL:
        freq = psutil.cpu_freq()
        vm   = psutil.virtual_memory()
        profile['cpu_freq_max_mhz']     = int(freq.max)     if freq else 0
        profile['cpu_freq_current_mhz'] = int(freq.current) if freq else 0
        profile['ram_total_mb']         = int(vm.total     / 1024**2)
        profile['ram_available_mb']     = int(vm.available / 1024**2)
    else:
        profile['cpu_freq_max_mhz']     = 0
        profile['cpu_freq_current_mhz'] = 0
        profile['ram_total_mb']         = 0
        profile['ram_available_mb']     = 0

    return profile


def print_profile(p: dict) -> None:
    print('\nDevice Profile')
    print(f'  Hostname    : {p["hostname"]}')
    print(f'  Platform    : {p["platform"]}')
    print(f'  Arch        : {p["arch"]}')
    print(f'  CPU model   : {p["cpu_model"]}')
    print(f'  CPU cores   : {p["cpu_cores"]}')
    freq_max = p.get("cpu_freq_max_mhz", 0)
    if freq_max:
        print(f'  Freq (max)  : {freq_max} MHz')
    ram_mb = p.get("ram_total_mb", 0)
    if ram_mb:
        print(f'  RAM total   : {ram_mb} MB  ({ram_mb/1024:.1f} GB)')
        print(f'  RAM avail   : {p["ram_available_mb"]} MB')
    print(f'  NEON        : {p["has_neon"]}  (ARM SIMD)')
    print(f'  CUDA        : {p["has_cuda"]}')

    _KNOWN = {
        'pi4_8gb':     dict(gflops=4.0,   note='Raspberry Pi 4 (8 GB)'),
        'pi4_4gb':     dict(gflops=4.0,   note='Raspberry Pi 4 (4 GB)'),
        'pi4_2gb':     dict(gflops=4.0,   note='Raspberry Pi 4 (2 GB)'),
        'pi4_1gb':     dict(gflops=4.0,   note='Raspberry Pi 4 (1 GB)'),
        'pi3':         dict(gflops=2.5,   note='Raspberry Pi 3'),
        'pizero2':     dict(gflops=1.8,   note='Pi Zero 2 W'),
        'pizero':      dict(gflops=0.25,  note='Pi Zero / Zero W'),
        'jetson_nano': dict(gflops=3.3,   note='Jetson Nano'),
        'pc':          dict(gflops=18.0,  note='Desktop / Server'),
    }

    arch  = p['arch']
    freq  = p.get('cpu_freq_max_mhz', 0) or 0
    ram   = p.get('ram_total_mb', 0)      or 0
    neon  = p['has_neon']

    if arch == 'armv6l':
        match, gflops = 'pizero', 0.25
    elif arch in ('aarch64', 'arm64') and neon:
        if freq >= 1600:                    match, gflops = 'pi4_4gb', 4.0
        elif freq >= 1400:                  match, gflops = 'jetson_nano', 3.3
        elif freq >= 1100:                  match, gflops = 'pi3', 2.5
        else:                               match, gflops = 'pizero2', 1.8
    elif 'x86' in arch:
        match, gflops = 'pc', 18.0
    else:
        match, gflops = None, None

    if match:
        note = _KNOWN[match]['note']
        print(f'  Best match  : --target_device {match}  ({note})')
        print(f'  Est. GFLOPS : {gflops:.2f} G-MACs/s')
    else:
        print('  Best match  : (no match — use --device_profile on the server)')

    print('\nNext step on the TRAINING SERVER:')
    print('  python3 sparse_finetune.py output/doppler/S1a \\')
    print('      --device_profile <this_file.json> \\')
    print('      --activities A,C,D,F,G')


if __name__ == '__main__':
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('device_profile.json')

    profile = collect_profile()
    print_profile(profile)

    out_path.write_text(json.dumps(profile, indent=2))
    print(f'Profile saved → {out_path}')
    print()
    print('Copy this file to your training server, then run:')
    print(f'  python3 sparse_finetune.py output/doppler/S1a \\')
    print(f'      --device_profile {out_path.name} \\')
    print(f'      --activities A,C,D,F,G')
