"""
CSI Preprocessing Pipeline - Batch Processing with Per-Stage Timing
Input: input/S1/ and input/S2/ folders
Output: Flat structure with original filenames in double_ratio/ and doppler/
CSV: Separate timing files for S1 and S2
"""

import gc
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as _savgol
from scipy.fftpack import fft, fftshift
from scipy.signal.windows import hann
from scipy.ndimage import gaussian_filter
from pathlib import Path
import re
import math as mt
import pandas as pd
import time



SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / 'preprocessing'  # parent folder containing S1/ and S2/ subfolders
OUTPUT_BASE = SCRIPT_DIR / 'output'

# Set True to process only the first file (quick correctness / timing check)
TEST_MODE = False

ACTIVITY_MAP = {
    'A': 'Walk', 'B': 'Run', 'C': 'Jump', 'D': 'Sitting',
    'E': 'Empty_room', 'F': 'Standing', 'G': 'Wave_hands', 'H': 'Clapping',
    'I': 'Lay_down', 'J': 'Wiping', 'K': 'Squat', 'L': 'Stretching'
}

# Spatial (per-packet) parameters
THRESHOLD     = 2
VOTE_PACKETS  = 100
VOTE_FRACTION = 2 / 3
CLIP_MULT     = 5.0
SG_WINDOW     = 11
SG_POLY       = 3

# Temporal (per-column) parameters
T_CLIP_MULT   = 5.0   # flag packet if |dr| > T_CLIP_MULT x median(|column|)

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
    phase = np.unwrap(np.angle(np.where(valid, z, 1+0j)))
    phase[~valid] = np.nan
    return phase


def _complex_interp_1d(z, x):
    """Interpolate NaNs in a 1-D complex array."""
    valid = np.isfinite(z.real)
    if valid.sum() == 0:
        return np.zeros_like(z)
    if valid.all():
        return z
    return (np.interp(x, x[valid], z.real[valid]) +
            1j * np.interp(x, x[valid], z.imag[valid]))


def _complex_clip_1d(z, clip_mult):
    """Clip large-magnitude outliers in a 1-D complex array."""
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
    """Savitzky-Golay on a 1-D complex array (real+imag independently)."""
    n = len(z)
    w = window if window % 2 == 1 else window - 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w <= poly or w < 3:
        return z
    return (_savgol(z.real, w, poly).astype(float) +
            1j * _savgol(z.imag, w, poly).astype(float))


def ratio_stage(data_2d, threshold, vote_packets, vote_fraction,
                clip_mult, sg_window, sg_poly):
    """
    One ratio stage: (n_packets, N) complex -> (n_packets, N//2) complex.
    Spatial cleaning only: A. Vote  B. Interpolate  C. Clip  D. Savitzky-Golay
    """
    n_packets, N = data_2d.shape
    n_pair       = N // 2
    check_upto   = min(vote_packets, n_packets)
    x            = np.arange(n_pair, dtype=float)

    num = data_2d[:, 0::2][:, :n_pair]
    den = data_2d[:, 1::2][:, :n_pair]
    nz  = den != 0
    raw_all = np.where(nz, num / np.where(nz, den, 1+0j), np.nan + 0j)

    # A. Vote
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

    # B. Set confirmed_bad to NaN, then interpolate
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

    # Handle any residual per-row NaNs (e.g. from den==0 subcarriers)
    still_nan = ~np.isfinite(result.real).all(axis=1)
    for p in np.where(still_nan)[0]:
        z     = result[p]; valid = np.isfinite(z.real)
        if valid.sum() == 0:
            result[p] = 0
        elif not valid.all():
            result[p] = (np.interp(x, x[valid], z.real[valid]) +
                         1j * np.interp(x, x[valid], z.imag[valid]))

    # C. Clip — per-row median and cap
    mags = np.abs(result)
    med  = np.median(mags, axis=1, keepdims=True)
    cap  = np.where(med != 0, clip_mult * med, np.inf)   # inf → no clip when med==0
    bad_clip = mags > cap
    if bad_clip.any():
        scale  = np.where(bad_clip, cap / np.maximum(mags, 1e-300), 1.0)
        result = result * scale
    del mags, med, cap, bad_clip

    # D. Savitzky-Golay on the full 2-D array along axis=1
    w = sg_window if sg_window % 2 == 1 else sg_window - 1
    w = min(w, n_pair if n_pair % 2 == 1 else n_pair - 1)
    if w > sg_poly and w >= 3:
        result = (_savgol(result.real, w, sg_poly, axis=1).astype(float) +
                  1j * _savgol(result.imag, w, sg_poly, axis=1).astype(float))

    return result, confirmed_bad, bad_count, min_votes


def temporal_clean(dr, t_clip_mult):
    """Flag packets where |dr[:,j]| > t_clip_mult * median per column, then interpolate."""
    n_packets, n_groups = dr.shape
    t = np.arange(n_packets, dtype=float)
    out = dr.copy()

    # E. Temporal clip + interp
    mags = np.abs(out)
    col_med = np.median(mags, axis=0, keepdims=True)
    col_med = np.where(col_med == 0, 1.0, col_med)  # avoid /0
    bad_mask = mags > t_clip_mult * col_med
    del mags

    out[bad_mask] = np.nan

    # Only iterate over columns that actually have flagged packets
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


# STFT parameters
NUM_SYMBOLS  = 51    # packets per Doppler window
SLIDING      = 1     # sliding step in packets
N_FFT        = 100   # Doppler bins
NOISE_LEVEL  = -2    # log10 noise floor

# Post-Doppler 2-D Gaussian smooth
SMOOTH_SIGMA_T = 0.5
SMOOTH_SIGMA_V = 0.5

Tc      = 1/150         # packet interval [s]
fc      = 5.18e9        # carrier frequency [Hz]
v_light = 3e8

delta_v = round(v_light / (Tc * fc * NUM_SYMBOLS), 4)


def compute_doppler_raw(csi_complex, num_symbols, sliding, n_fft):
    """Returns raw (unnormalised) power array shape (n_windows, n_fft)."""
    n_packets = csi_complex.shape[0]
    hann_win  = np.expand_dims(hann(num_symbols), axis=-1)

    csi_clean = np.nan_to_num(csi_complex)
    n_windows = len(range(0, n_packets - num_symbols, sliding))
    profiles  = np.empty((n_windows, n_fft))

    for k, i in enumerate(range(0, n_packets - num_symbols, sliding)):
        win = csi_clean[i:i + num_symbols, :]
        win = win - win.mean(axis=0, keepdims=True)  # per-window static removal

        spec  = fft(win * hann_win, n=n_fft, axis=0)
        spec  = fftshift(spec, axes=0)

        power = np.abs(spec * np.conj(spec))
        profiles[k] = power.sum(axis=1)

    return profiles          # (n_windows, n_fft)


def get_source_folder(filepath):
    if '/S1/' in str(filepath) or '\\S1\\' in str(filepath):
        return 'S1'
    elif '/S2/' in str(filepath) or '\\S2\\' in str(filepath):
        return 'S2'
    else:
        return 'Unknown'


def parse_activity(filename):
    activity_match = re.search(r'_([A-L])_', filename)
    if activity_match:
        activity_letter = activity_match.group(1)
        return ACTIVITY_MAP.get(activity_letter, 'Unknown')
    return 'Unknown'


def process_one_file(file_path):
    filename = file_path.name
    source_folder = get_source_folder(file_path)
    activity_name = parse_activity(filename)

    data = np.load(file_path)['data']

    timings = {
        'File': filename,
        'Source_Folder': source_folder,
        'Activity': activity_name
    }

    total_start = time.time()

    remove_idx = [994, 1492, 1493]  # null subcarriers
    keep_idx   = [i for i in range(data.shape[1]) if i not in remove_idx]
    data = data[:, keep_idx]

    stage1_start = time.time()
    single_ratio, bad_s, cnt_s, _ = ratio_stage(
        data, THRESHOLD, VOTE_PACKETS, VOTE_FRACTION, CLIP_MULT, SG_WINDOW, SG_POLY)
    del data; gc.collect()
    stage1_time = time.time() - stage1_start
    timings['Stage1_SingleRatio_seconds'] = round(stage1_time, 2)

    stage2_start = time.time()
    double_ratio, bad_d, cnt_d, _ = ratio_stage(
        single_ratio, THRESHOLD, VOTE_PACKETS, VOTE_FRACTION, CLIP_MULT, SG_WINDOW, SG_POLY)
    del single_ratio; gc.collect()
    stage2_time = time.time() - stage2_start
    timings['Stage2_DoubleRatio_seconds'] = round(stage2_time, 2)

    stage3_start = time.time()
    double_ratio, n_flagged = temporal_clean(double_ratio, T_CLIP_MULT)
    stage3_time = time.time() - stage3_start
    timings['Stage3_Temporal_seconds'] = round(stage3_time, 2)

    dr_folder = OUTPUT_BASE / 'double_ratio'
    dr_folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dr_folder / filename, double_ratio=double_ratio)

    stage4_start = time.time()
    raw_power = compute_doppler_raw(double_ratio, NUM_SYMBOLS, SLIDING, N_FFT)

    # Per-window peak normalisation avoids baking in scene-specific absolute power.
    noise_floor = mt.pow(10, NOISE_LEVEL)
    win_max = raw_power.max(axis=1, keepdims=True)
    win_max = np.where(win_max == 0, 1.0, win_max)
    doppler = np.clip(raw_power / win_max, noise_floor, 1.0)

    if SMOOTH_SIGMA_T > 0 or SMOOTH_SIGMA_V > 0:
        doppler = gaussian_filter(doppler, sigma=(SMOOTH_SIGMA_T, SMOOTH_SIGMA_V))
        doppler = np.clip(doppler, noise_floor, 1.0)

    stage4_time = time.time() - stage4_start
    timings['Stage4_Doppler_seconds'] = round(stage4_time, 2)

    dop_folder = OUTPUT_BASE / 'doppler'
    dop_folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dop_folder / filename,
        doppler=doppler,
        delta_v=delta_v,
        Tc=Tc,
        num_symbols=NUM_SYMBOLS,
        sliding=SLIDING,
        n_fft=N_FFT,
        source_folder=source_folder,
        activity=activity_name
    )

    total_time = time.time() - total_start
    timings['Total_Time_seconds'] = round(total_time, 2)

    print(f"{filename}: {total_time:.2f}s")

    return timings


def main():
    (OUTPUT_BASE / 'double_ratio').mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / 'doppler').mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / 'csv').mkdir(parents=True, exist_ok=True)

    s1_files = sorted(list((BASE_PATH / 'S1').glob('*.npz'))) if (BASE_PATH / 'S1').exists() else []
    s2_files = sorted(list((BASE_PATH / 'S2').glob('*.npz'))) if (BASE_PATH / 'S2').exists() else []
    all_files = s1_files + s2_files

    if len(all_files) == 0:
        print(f"No .npz files found in {BASE_PATH}/S1/ or {BASE_PATH}/S2/")
        return

    print(f"Found {len(s1_files)} files in S1/, {len(s2_files)} files in S2/")

    if TEST_MODE:
        all_files = all_files[:1]

    all_timings = []
    success_count = 0
    error_count = 0

    for file_path in all_files:
        try:
            timing_data = process_one_file(file_path)
            all_timings.append(timing_data)
            success_count += 1
        except Exception as e:
            print(f"ERROR processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    csv_folder = OUTPUT_BASE / 'csv'

    s1_timings = [t for t in all_timings if t['Source_Folder'] == 'S1']
    s2_timings = [t for t in all_timings if t['Source_Folder'] == 'S2']

    column_order = [
        'File', 'Activity',
        'Stage1_SingleRatio_seconds',
        'Stage2_DoubleRatio_seconds',
        'Stage3_Temporal_seconds',
        'Stage4_Doppler_seconds',
        'Total_Time_seconds'
    ]

    if s1_timings:
        s1_df = pd.DataFrame(s1_timings)[column_order]
        s1_df.to_csv(csv_folder / 'S1_timing.csv', index=False)

    if s2_timings:
        s2_df = pd.DataFrame(s2_timings)[column_order]
        s2_df.to_csv(csv_folder / 'S2_timing.csv', index=False)

    print(f"Processed {success_count} files ({error_count} errors)")


if __name__ == '__main__':
    main()
