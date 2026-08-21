"""
CSI Preprocessing Pipeline - Batch Processing All Files & All Antennas
Input:  input_data/*.txt (pickled files)
Output: output/double_ratio/, output/doppler/, output/csv/
"""

import numpy as np
import pickle
import os
from pathlib import Path
from scipy.signal import savgol_filter as _savgol
from scipy.fftpack import fft, fftshift
from scipy.signal.windows import hann
from scipy.ndimage import gaussian_filter
import pandas as pd
import time
import math as mt

import sys
sys.modules['numpy._core'] = np.core
sys.modules['numpy._core.multiarray'] = np.core.multiarray

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / 'input_data'
OUTPUT_DIR = SCRIPT_DIR / 'output'

# Bad subcarriers to remove (242 raw -> 228 kept -> interpolated to TARGET_N_SUB)
DELETE_IDX = np.asarray(
    [0, 1, 2, 3, 4, 5,
     127, 128, 129,
     251, 252, 253, 254, 255],
    dtype=int
)

TARGET_N_SUB = 233
NUM_ANTENNAS = 4

# Spatial (per-packet) parameters
THRESHOLD     = 2
VOTE_PACKETS  = 100
VOTE_FRACTION = 2 / 3
CLIP_MULT     = 5.0
SG_WINDOW     = 11
SG_POLY       = 3

# Temporal (per-column) parameter
T_CLIP_MULT = 5.0

# STFT parameters
NUM_SYMBOLS = 51
SLIDING     = 10
N_FFT       = 100
NOISE_LEVEL = -2

# Post-Doppler 2-D Gaussian smooth
SMOOTH_SIGMA_T = 0.5
SMOOTH_SIGMA_V = 0.5

Tc      = 6e-3   # 802.11ac 80 MHz / 5 GHz: one channel estimate every 6 ms
fc      = 5.18e9
v_light = 3e8

delta_v = round(v_light / (Tc * fc * NUM_SYMBOLS), 4)


def _interp_missing_subcarriers(arr, target_n_sub):
    """arr: (n_packets, n_subcarriers). Interpolate missing subcarriers up to target_n_sub."""
    n_pkt, n_sub = arr.shape

    if n_sub == target_n_sub:
        return arr

    if n_sub > target_n_sub:
        return arr[:, :target_n_sub]

    missing_idx  = np.arange(n_sub, target_n_sub)
    existing_idx = np.arange(n_sub)

    out = np.empty((n_pkt, target_n_sub), dtype=arr.dtype)
    out[:, :n_sub] = arr

    for p in range(n_pkt):
        row = arr[p]
        out[p, missing_idx] = (np.interp(missing_idx, existing_idx, row.real) +
                                1j * np.interp(missing_idx, existing_idx, row.imag))

    return out


def _ratio_row(row):
    n   = len(row) // 2
    num = row[0::2][:n]
    den = row[1::2][:n]
    out = np.full(n, np.nan + 0j, dtype=complex)
    nz  = den != 0
    out[nz] = num[nz] / den[nz]
    return out


def _unwrap_phase_safe(z):
    valid = np.isfinite(z)
    phase = np.unwrap(np.angle(np.where(valid, z, 1 + 0j)))
    phase[~valid] = np.nan
    return phase


def _complex_interp_indices(z, x, bad_indices):
    if len(bad_indices) == 0:
        return z
    z = z.copy()
    valid_mask = np.ones(len(z), dtype=bool)
    valid_mask[bad_indices] = False
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return np.zeros_like(z)
    z.real[bad_indices] = np.interp(bad_indices, valid_indices, z.real[valid_indices])
    z.imag[bad_indices] = np.interp(bad_indices, valid_indices, z.imag[valid_indices])
    return z


def _complex_clip_1d(z, clip_mult):
    med = np.median(np.abs(z))
    if med == 0:
        return z
    cap  = clip_mult * med
    mask = np.abs(z) > cap
    if mask.any():
        z = z.copy()
        z[mask] = cap * z[mask] / np.abs(z[mask])
    return z


def _complex_savgol_1d(z, window, poly):
    n = len(z)
    w = window if window % 2 == 1 else window - 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w <= poly or w < 3:
        return z
    return (_savgol(z.real, w, poly).astype(float) +
            1j * _savgol(z.imag, w, poly).astype(float))


def ratio_stage(data_2d, threshold, vote_packets, vote_fraction,
                clip_mult, sg_window, sg_poly):
    n_packets, N = data_2d.shape
    n_pair       = N // 2
    check_upto   = min(vote_packets, n_packets)
    x            = np.arange(n_pair, dtype=float)

    # A. Compute all ratios in one vectorised broadcast
    num = data_2d[:, 0::2][:, :n_pair]
    den = data_2d[:, 1::2][:, :n_pair]
    nz  = den != 0
    raw_all = np.where(nz, num / np.where(nz, den, 1+0j), np.nan + 0j)

    # B. Vote — vectorised over all check_upto packets at once
    subset  = raw_all[:check_upto]
    valid_m = np.isfinite(subset.real)
    phase   = np.unwrap(np.angle(np.where(valid_m, subset, 1+0j)), axis=1)
    phase[~valid_m] = np.nan

    row_valid = valid_m.sum(axis=1)
    row_avg   = np.where(valid_m, np.abs(phase), 0.0).sum(axis=1) / row_valid.clip(1)

    bad_m = valid_m & (np.abs(phase) > threshold * row_avg[:, None])
    bad_m[row_valid == 0] = False
    bad_count = bad_m.sum(axis=0)
    del subset, valid_m, phase, bad_m

    min_votes     = int(np.ceil(vote_fraction * check_upto))
    confirmed_bad = np.where(bad_count >= min_votes)[0]

    # C. Set confirmed_bad to NaN for all packets at once, then interpolate
    result = raw_all.copy()
    result[:, confirmed_bad] = np.nan

    if len(confirmed_bad) > 0:
        valid_idx = np.where(~np.isin(np.arange(n_pair), confirmed_bad))[0]
        if len(valid_idx) > 0:
            for bad_j in confirmed_bad:
                left  = valid_idx[valid_idx < bad_j]
                right = valid_idx[valid_idx > bad_j]
                if len(left) == 0:
                    result[:, bad_j] = result[:, right[0]]
                elif len(right) == 0:
                    result[:, bad_j] = result[:, left[-1]]
                else:
                    lx, rx = int(left[-1]), int(right[0])
                    alpha  = float(bad_j - lx) / (rx - lx)
                    result[:, bad_j] = (1.0 - alpha) * result[:, lx] + alpha * result[:, rx]

    # Handle residual per-row NaNs (den==0 subcarriers)
    still_nan = ~np.isfinite(result.real).all(axis=1)
    for p in np.where(still_nan)[0]:
        z = result[p]; valid = np.isfinite(z.real)
        if valid.sum() == 0:
            result[p] = 0
        elif not valid.all():
            result[p] = (np.interp(x, x[valid], z.real[valid]) +
                         1j * np.interp(x, x[valid], z.imag[valid]))

    # D. Clip — vectorised per-row median broadcast
    mags = np.abs(result)
    med  = np.median(mags, axis=1, keepdims=True)
    cap  = np.where(med != 0, clip_mult * med, np.inf)
    bad_clip = mags > cap
    if bad_clip.any():
        scale  = np.where(bad_clip, cap / np.maximum(mags, 1e-300), 1.0)
        result = result * scale
    del mags, med, cap, bad_clip

    # E. Savitzky-Golay — single 2-D call along subcarrier axis
    w = sg_window if sg_window % 2 == 1 else sg_window - 1
    w = min(w, n_pair if n_pair % 2 == 1 else n_pair - 1)
    if w > sg_poly and w >= 3:
        result = (_savgol(result.real, w, sg_poly, axis=1).astype(float) +
                  1j * _savgol(result.imag, w, sg_poly, axis=1).astype(float))

    return result, confirmed_bad, bad_count, min_votes


def temporal_clean(dr, t_clip_mult):
    n_packets, n_groups = dr.shape
    t   = np.arange(n_packets, dtype=float)
    out = dr.copy()

    mags    = np.abs(out)
    col_med = np.median(mags, axis=0, keepdims=True)
    col_med = np.where(col_med == 0, 1.0, col_med)
    bad_mask = mags > t_clip_mult * col_med
    del mags

    out[bad_mask] = np.nan

    for j in np.where(bad_mask.any(axis=0))[0]:
        col   = out[:, j]
        valid = np.isfinite(col.real)
        if valid.sum() == 0:
            out[:, j] = 0
        elif not valid.all():
            out[:, j] = (np.interp(t, t[valid], col.real[valid]) +
                         1j * np.interp(t, t[valid], col.imag[valid]))

    n_flagged = bad_mask.sum()
    return out, n_flagged


def compute_doppler_raw(csi_complex, num_symbols, sliding, n_fft):
    n_packets = csi_complex.shape[0]
    hann_win  = np.expand_dims(hann(num_symbols), axis=-1)

    csi_clean = np.nan_to_num(csi_complex)
    n_windows = len(range(0, n_packets - num_symbols, sliding))
    profiles  = np.empty((n_windows, n_fft))

    for k, i in enumerate(range(0, n_packets - num_symbols, sliding)):
        win = csi_clean[i:i + num_symbols, :]
        win = win - win.mean(axis=0, keepdims=True)

        spec = fft(win * hann_win, n=n_fft, axis=0)
        spec = fftshift(spec, axes=0)

        power = np.abs(spec * np.conj(spec))
        profiles[k] = power.sum(axis=1)

    return profiles


def process_one_file_one_antenna(file_path, antenna_idx):
    filename  = os.path.basename(file_path)
    base_name = filename.replace('.txt', '')

    print(f'\n  Antenna {antenna_idx}...', end=' ', flush=True)
    total_start = time.time()

    with open(file_path, 'rb') as f:
        _raw = pickle.load(f)   # (subcarriers, packets, antennas)

    _slice = _raw[:, :, antenna_idx]
    keep_idx = np.array([i for i in range(_slice.shape[0]) if i not in DELETE_IDX])
    data = _slice[keep_idx, :]
    data = data.T   # (packets, subcarriers)
    data = _interp_missing_subcarriers(data, TARGET_N_SUB)

    timings = {'Filename': filename, 'Antenna': antenna_idx}

    stage1_start = time.time()
    single_ratio, _, _, _ = ratio_stage(
        data, THRESHOLD, VOTE_PACKETS, VOTE_FRACTION, CLIP_MULT, SG_WINDOW, SG_POLY)
    timings['Stage1_SingleRatio_sec'] = round(time.time() - stage1_start, 2)

    stage2_start = time.time()
    double_ratio, _, _, _ = ratio_stage(
        single_ratio, THRESHOLD, VOTE_PACKETS, VOTE_FRACTION, CLIP_MULT, SG_WINDOW, SG_POLY)
    timings['Stage2_DoubleRatio_sec'] = round(time.time() - stage2_start, 2)

    stage3_start = time.time()
    double_ratio, _ = temporal_clean(double_ratio, T_CLIP_MULT)
    timings['Stage3_Temporal_sec'] = round(time.time() - stage3_start, 2)

    dr_folder = Path(OUTPUT_DIR) / 'double_ratio'
    dr_folder.mkdir(parents=True, exist_ok=True)
    dr_path = dr_folder / f'{base_name}_ant{antenna_idx}.npz'
    np.savez_compressed(dr_path, double_ratio=double_ratio)

    stage4_start = time.time()
    raw_power = compute_doppler_raw(double_ratio, NUM_SYMBOLS, SLIDING, N_FFT)

    # Per-window normalisation avoids baking in scene-specific absolute power.
    noise_floor = mt.pow(10, NOISE_LEVEL)
    win_max = raw_power.max(axis=1, keepdims=True)
    win_max = np.where(win_max == 0, 1.0, win_max)
    doppler = np.clip(raw_power / win_max, noise_floor, 1.0)

    if SMOOTH_SIGMA_T > 0 or SMOOTH_SIGMA_V > 0:
        doppler = gaussian_filter(doppler, sigma=(SMOOTH_SIGMA_T, SMOOTH_SIGMA_V))
        doppler = np.clip(doppler, noise_floor, 1.0)

    timings['Stage4_Doppler_sec'] = round(time.time() - stage4_start, 2)

    dop_folder = Path(OUTPUT_DIR) / 'doppler'
    dop_folder.mkdir(parents=True, exist_ok=True)
    dop_path = dop_folder / f'{base_name}_ant{antenna_idx}.npz'
    np.savez_compressed(
        dop_path,
        doppler=doppler,
        delta_v=delta_v,
        Tc=Tc,
        num_symbols=NUM_SYMBOLS,
        sliding=SLIDING,
        n_fft=N_FFT
    )

    timings['Total_Time_sec'] = round(time.time() - total_start, 2)
    print(f"done ({timings['Total_Time_sec']}s)", flush=True)

    return timings


def main():
    print(f'Input folder:  {DATA_DIR}')
    print(f'Output folder: {OUTPUT_DIR}')
    print(f'Target subcarriers: {TARGET_N_SUB}')

    all_files = sorted(Path(DATA_DIR).glob('*.txt'))
    print(f'Found {len(all_files)} files')

    all_timings = []

    for file_idx, file_path in enumerate(all_files, 1):
        print(f'\nFILE {file_idx}/{len(all_files)}: {file_path.name}')

        try:
            for antenna_idx in range(NUM_ANTENNAS):
                timing_data = process_one_file_one_antenna(file_path, antenna_idx)
                all_timings.append(timing_data)
        except Exception as e:
            print(f'ERROR processing {file_path.name}: {e}')
            import traceback
            traceback.print_exc()

    csv_folder = Path(OUTPUT_DIR) / 'csv'
    csv_folder.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_timings)
    csv_path = csv_folder / 'processing_timing.csv'
    df.to_csv(csv_path, index=False)

    dr_files  = len(list((Path(OUTPUT_DIR) / 'double_ratio').glob('*.npz')))
    dop_files = len(list((Path(OUTPUT_DIR) / 'doppler').glob('*.npz')))
    total_times = df['Total_Time_sec'].values

    print(f'\nTiming CSV saved: {csv_path}')
    print(f'Records: {len(all_timings)}  double_ratio: {dr_files}  doppler: {dop_files}')
    print(f'Time avg={np.mean(total_times):.2f}s  min={np.min(total_times):.2f}s  max={np.max(total_times):.2f}s')


if __name__ == '__main__':
    main()
