"""
CSI Preprocessing Pipeline - HeadGest Dataset
Input:  input_files/phase{1,2,3}/*.txt  (pickled DataFrames)
Output: output/double_ratio/{phase}/, output/doppler/{phase}/, output/csv/

WiFi standard : 802.11n  (11n), 40 MHz bandwidth
Phase 1       : sampling rate 100 Hz  (training data)
Phase 2, 3    : sampling rate 200 Hz  (test scenarios)
Antennas      : 1  (single-link CSI)
Subcarriers   : 114 complex  (228 floats stored as interleaved real+imaginary)
"""

import numpy as np
import pickle
from pathlib import Path
from scipy.signal import savgol_filter as _savgol
from scipy.fftpack import fft, fftshift
from scipy.signal.windows import hann
from scipy.ndimage import gaussian_filter
import pandas as pd
import time
import math as mt

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_BASE = SCRIPT_DIR / 'input_files'
OUTPUT_DIR = SCRIPT_DIR / 'output'

PHASE_RATES = {
    'phase1': 100,   # Hz
    'phase2': 200,   # Hz
    'phase3': 200,   # Hz
}

# Carrier frequency (802.11n at 2.4 GHz band — adjust to your channel,
# e.g. 2.412e9 for ch1, 2.437e9 for ch6, 2.462e9 for ch11)
fc      = 2.4e9
v_light = 3e8

GESTURE_CODE = {
    'Forward':       'FW',
    'Looking Down':  'LD',
    'Looking Left':  'LL',
    'Looking Right': 'LR',
    'Looking Up':    'LU',
    'Nodding':       'ND',
    'Shaking':       'SH',
}

NUM_ANTENNAS  = 1    # single-link CSI
N_SUBCARRIERS = 114  # 228 floats -> 114 complex subcarriers per packet

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
NUM_SYMBOLS = 31
SLIDING     = 1
N_FFT       = 100
NOISE_LEVEL = -1.5

# Post-Doppler 2-D Gaussian smooth
SMOOTH_SIGMA_T = 0.5
SMOOTH_SIGMA_V = 0.5


def _unwrap_phase_safe(z):
    """Unwrap phase of a 1-D complex vector, marking NaN as NaN."""
    valid = np.isfinite(z)
    phase = np.unwrap(np.angle(np.where(valid, z, 1 + 0j)))
    phase[~valid] = np.nan
    return phase


def _interp_bad_cols(arr2d, bad_cols, valid_cols):
    """Linearly interpolate bad_cols from valid_cols for every row at once."""
    if len(bad_cols) == 0 or len(valid_cols) == 0:
        return arr2d
    arr = arr2d.copy()
    for j in bad_cols:
        left  = valid_cols[valid_cols < j]
        right = valid_cols[valid_cols > j]
        if len(left) > 0 and len(right) > 0:
            j_l, j_r = int(left[-1]), int(right[0])
            w = (j - j_l) / (j_r - j_l)
            arr[:, j] = (1.0 - w) * arr2d[:, j_l] + w * arr2d[:, j_r]
        elif len(left) > 0:
            arr[:, j] = arr2d[:, int(left[-1])]
        else:
            arr[:, j] = arr2d[:, int(right[0])]
    return arr


def _interp_row_complex(z, bad_j):
    """Interpolate bad_j positions in a single complex row vector."""
    if len(bad_j) == 0:
        return z
    z    = z.copy()
    good = np.where(np.isfinite(z.real))[0]
    if len(good) == 0:
        return np.zeros_like(z)
    z.real[bad_j] = np.interp(bad_j, good, z.real[good])
    z.imag[bad_j] = np.interp(bad_j, good, z.imag[good])
    return z


def ratio_stage(
    data_2d,
    threshold,
    vote_packets,
    vote_fraction,
    clip_mult,
    sg_window,
    sg_poly
):
    """One ratio stage: (n_packets, N) complex -> (n_packets, N//2) complex. Fully vectorised."""
    n_packets, N = data_2d.shape
    n_pair       = N // 2
    check_upto   = min(vote_packets, n_packets)

    # (1) Vectorised ratio computation
    num = data_2d[:, 0::2][:, :n_pair]
    den = data_2d[:, 1::2][:, :n_pair]

    nz_mask   = den != 0
    all_ratios = np.full((n_packets, n_pair), np.nan + 0j, dtype=complex)
    all_ratios[nz_mask] = num[nz_mask] / den[nz_mask]
    zero_mask = ~nz_mask   # positions where denominator was 0

    # (2) Vote — vectorised over all check_upto packets at once
    subset    = all_ratios[:check_upto]
    valid_m   = np.isfinite(subset.real)
    phase     = np.unwrap(np.angle(np.where(valid_m, subset, 1+0j)), axis=1)
    phase[~valid_m] = np.nan
    row_valid = valid_m.sum(axis=1)
    row_avg   = np.where(valid_m, np.abs(phase), 0.0).sum(axis=1) / row_valid.clip(1)
    bad_m     = valid_m & (np.abs(phase) > threshold * row_avg[:, None])
    bad_m[row_valid == 0] = False
    bad_count = bad_m.sum(axis=0)
    del subset, valid_m, phase, bad_m

    min_votes     = int(np.ceil(vote_fraction * check_upto))
    confirmed_bad = np.where(bad_count >= min_votes)[0]

    # (3) Mark confirmed_bad as NaN (all packets at once)
    if len(confirmed_bad) > 0:
        all_ratios[:, confirmed_bad] = np.nan + 0j

    # (4a) Interpolate confirmed_bad columns (vectorised across all packets)
    all_nan_cols = np.unique(
        np.concatenate([confirmed_bad,
                        np.where(zero_mask.any(axis=0))[0]])
    )
    valid_cols = np.setdiff1d(np.arange(n_pair), all_nan_cols)

    if len(all_nan_cols) > 0 and len(valid_cols) > 0:
        real_fixed = _interp_bad_cols(all_ratios.real, all_nan_cols, valid_cols)
        imag_fixed = _interp_bad_cols(all_ratios.imag, all_nan_cols, valid_cols)
        all_ratios[:, all_nan_cols] = (
            real_fixed[:, all_nan_cols] + 1j * imag_fixed[:, all_nan_cols]
        )

    # (4b) Fix any remaining per-packet NaN (sparse, from zero denominators)
    remaining_nan = ~np.isfinite(all_ratios.real)
    rows_with_nan = np.where(remaining_nan.any(axis=1))[0]
    for p in rows_with_nan:
        bad_j = np.where(remaining_nan[p])[0]
        all_ratios[p] = _interp_row_complex(all_ratios[p], bad_j)

    # (5) Vectorised per-packet amplitude clipping
    mags     = np.abs(all_ratios)
    med      = np.median(mags, axis=1, keepdims=True)
    med_safe = np.where(med == 0, 1.0, med)
    cap      = clip_mult * med_safe
    over     = mags > cap
    if over.any():
        phases     = np.angle(all_ratios)
        all_ratios = np.where(over, cap * np.exp(1j * phases), all_ratios)

    # (6) Vectorised Savitzky-Golay over subcarrier axis (axis=1)
    n = n_pair
    w = sg_window if sg_window % 2 == 1 else sg_window - 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w > sg_poly and w >= 3:
        r = _savgol(all_ratios.real, w, sg_poly, axis=1).astype(np.float64)
        i = _savgol(all_ratios.imag, w, sg_poly, axis=1).astype(np.float64)
        all_ratios = (r + 1j * i).astype(complex)

    return all_ratios, confirmed_bad, bad_count, min_votes


def temporal_clean(dr, t_clip_mult):
    """Temporal outlier removal along the packet axis (vectorised)."""
    n_packets = dr.shape[0]
    t   = np.arange(n_packets, dtype=float)
    out = dr.copy()

    mags     = np.abs(out)
    col_med  = np.median(mags, axis=0, keepdims=True)
    col_med  = np.where(col_med == 0, 1.0, col_med)
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

    return out, bad_mask.sum()


def compute_doppler_raw(csi_complex, num_symbols, sliding, n_fft):
    """Returns raw (unnormalised) power array shape (n_windows, n_fft)."""
    n_packets = csi_complex.shape[0]
    hann_win  = np.expand_dims(hann(num_symbols), axis=-1)

    csi_clean = np.nan_to_num(csi_complex)
    n_windows = len(range(0, n_packets - num_symbols, sliding))
    profiles  = np.empty((n_windows, n_fft))

    for k, i in enumerate(range(0, n_packets - num_symbols, sliding)):
        win = csi_clean[i:i + num_symbols, :]
        win = win - win.mean(axis=0, keepdims=True)

        spec  = fft(win * hann_win, n=n_fft, axis=0)
        spec  = fftshift(spec, axes=0)
        power = np.abs(spec * np.conj(spec))
        profiles[k] = power.sum(axis=1)

    return profiles


def load_csi_complex(file_path):
    """
    Load pickled DataFrame and return complex CSI matrix.

    Input DataFrame: (n_packets, 229) — columns 0..227 are floats, 'headlabel' is str.
    Output: (n_packets, 114)  complex64
      even columns (0,2,4,...) = real parts
      odd  columns (1,3,5,...) = imaginary parts
    """
    with open(file_path, 'rb') as f:
        df = pickle.load(f)

    numeric_cols = [c for c in df.columns if isinstance(c, int)]
    arr = df[numeric_cols].values.astype(np.float32)   # (n_packets, 228)

    real = arr[:, 0::2]   # (n_packets, 114)
    imag = arr[:, 1::2]   # (n_packets, 114)
    return (real + 1j * imag).astype(np.complex64)


def process_one_file(file_path, phase, gesture_name):
    """Process one gesture file through the 4-stage pipeline."""
    fs      = PHASE_RATES[phase]
    Tc      = 1.0 / fs
    delta_v = round(v_light / (Tc * fc * NUM_SYMBOLS), 4)

    gesture_code = GESTURE_CODE.get(gesture_name)
    if gesture_code is None:
        raise ValueError(f"Unknown gesture: '{gesture_name}'")

    base_name = f'signal_{phase}_{gesture_code}_ant0'

    print(f'  {gesture_name} ({gesture_code}) @ {fs} Hz...', end=' ', flush=True)
    total_start = time.time()

    csi = load_csi_complex(file_path)
    timings = {'File': file_path.name, 'Phase': phase, 'Gesture': gesture_name}

    t1 = time.time()
    single_ratio, _, _, _ = ratio_stage(
        csi, THRESHOLD, VOTE_PACKETS, VOTE_FRACTION, CLIP_MULT, SG_WINDOW, SG_POLY)
    timings['Stage1_sec'] = round(time.time() - t1, 2)

    t2 = time.time()
    double_ratio, _, _, _ = ratio_stage(
        single_ratio, THRESHOLD, VOTE_PACKETS, VOTE_FRACTION, CLIP_MULT, SG_WINDOW, SG_POLY)
    timings['Stage2_sec'] = round(time.time() - t2, 2)

    t3 = time.time()
    double_ratio, _ = temporal_clean(double_ratio, T_CLIP_MULT)
    timings['Stage3_sec'] = round(time.time() - t3, 2)

    dr_folder = OUTPUT_DIR / 'double_ratio' / phase
    dr_folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dr_folder / f'{base_name}.npz', double_ratio=double_ratio)

    t4 = time.time()
    raw_power   = compute_doppler_raw(double_ratio, NUM_SYMBOLS, SLIDING, N_FFT)
    noise_floor = mt.pow(10, NOISE_LEVEL)

    win_max = raw_power.max(axis=1, keepdims=True)
    win_max = np.where(win_max == 0, 1.0, win_max)
    doppler = np.clip(raw_power / win_max, noise_floor, 1.0)

    if SMOOTH_SIGMA_T > 0 or SMOOTH_SIGMA_V > 0:
        doppler = gaussian_filter(doppler, sigma=(SMOOTH_SIGMA_T, SMOOTH_SIGMA_V))
        doppler = np.clip(doppler, noise_floor, 1.0)

    timings['Stage4_sec'] = round(time.time() - t4, 2)

    dop_folder = OUTPUT_DIR / 'doppler' / phase
    dop_folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dop_folder / f'{base_name}.npz',
        doppler=doppler,
        delta_v=delta_v,
        Tc=Tc,
        num_symbols=NUM_SYMBOLS,
        sliding=SLIDING,
        n_fft=N_FFT
    )

    timings['Total_sec'] = round(time.time() - total_start, 2)
    print(f"done ({timings['Total_sec']}s) -> {doppler.shape[0]} frames", flush=True)
    return timings


def main():
    print(f'Input base : {INPUT_BASE}')
    print(f'Output dir : {OUTPUT_DIR}')
    print(f'Carrier    : {fc/1e9:.3f} GHz  Bandwidth: 40 MHz  Standard: 802.11n')
    print(f"Phase 1    : {PHASE_RATES['phase1']} Hz  Phase 2/3: {PHASE_RATES['phase2']} Hz")
    print(f'Subcarriers: {N_SUBCARRIERS} complex  (228 floats interleaved real+imag)')

    phases = ['phase1', 'phase2', 'phase3']
    all_timings = []

    for phase in phases:
        phase_dir = INPUT_BASE / phase
        if not phase_dir.exists():
            print(f'[skip] {phase}: folder not found')
            continue

        txt_files = sorted(phase_dir.glob('*.txt'))
        if not txt_files:
            print(f'[skip] {phase}: no .txt files')
            continue

        print(f'\nPHASE: {phase}  ({PHASE_RATES[phase]} Hz) — {len(txt_files)} files')

        for file_path in txt_files:
            gesture_name = file_path.stem
            if gesture_name not in GESTURE_CODE:
                print(f"  [warn] Unknown gesture '{gesture_name}', skipping.")
                continue

            try:
                timing_data = process_one_file(file_path, phase, gesture_name)
                all_timings.append(timing_data)
            except Exception as e:
                print(f'  ERROR: {file_path.name}: {e}')
                import traceback
                traceback.print_exc()

    csv_folder = OUTPUT_DIR / 'csv'
    csv_folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_timings)
    df.to_csv(csv_folder / 'processing_timing.csv', index=False)
    print(f'\nTiming CSV saved: {csv_folder / "processing_timing.csv"}')

    print(f'\nTotal processed: {len(all_timings)}')
    for phase in phases:
        dr_n  = len(list((OUTPUT_DIR / 'double_ratio' / phase).glob('*.npz'))) \
                if (OUTPUT_DIR / 'double_ratio' / phase).exists() else 0
        dop_n = len(list((OUTPUT_DIR / 'doppler' / phase).glob('*.npz'))) \
                if (OUTPUT_DIR / 'doppler' / phase).exists() else 0
        print(f'  {phase}: double_ratio={dr_n}  doppler={dop_n}')

    if all_timings:
        totals = [t['Total_sec'] for t in all_timings]
        print(f'\nTiming: avg={np.mean(totals):.1f}s  '
              f'min={np.min(totals):.1f}s  max={np.max(totals):.1f}s')


if __name__ == '__main__':
    main()
