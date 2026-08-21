"""
CSI Preprocessing Pipeline - Batch Processing with Per-Stage Timing
Input: input_data/save_classroom/ and input_data/save_office/ folders
Output: Mirrored subfolder structure with original filenames in double_ratio/ and doppler/
CSV: Separate timing files for save_classroom and save_office
"""

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

SCRIPT_DIR  = Path(__file__).resolve().parent
BASE_PATH   = SCRIPT_DIR / 'input_data'    # parent folder containing save_classroom/ and save_office/
OUTPUT_BASE = SCRIPT_DIR / 'output_data'

ACTIVITY_MAP = {
    'A': 'Push_forward', 'B': 'Rotate', 'C': 'Hands_up_and_down', 'D': 'Waive',
    'E': 'Brush', 'F': 'Clap', 'G': 'Sit', 'H': 'Eat', 'I': 'Drink', 'J': 'Kick',
    'K': 'Bend_forward', 'L': 'Wash_hands', 'M': 'Call', 'N': 'Browsing_phone',
    'O': 'Check_wrist', 'P': 'Read', 'Q': 'Waive_while_sitting', 'R': 'Writing',
    'S': 'Side_bend', 'T': 'Standing',
}

# Spatial (per-packet) parameters
THRESHOLD     = 2
VOTE_PACKETS  = 500
VOTE_FRACTION = 2 / 3
CLIP_MULT     = 5.0
SG_WINDOW     = 11
SG_POLY       = 3

# Temporal (per-column) parameter
T_CLIP_MULT = 5.0   # flag packet if |dr| > T_CLIP_MULT x median(|column|)


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
    bad_m     = valid_m & (np.abs(phase) > threshold * row_avg[:, None])
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
    """Temporal outlier removal along the packet (time) axis."""
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

    n_flagged = bad_mask.sum()
    return out, n_flagged


# STFT parameters
NUM_SYMBOLS  = 100   # packets per Doppler window
SLIDING      = 1     # sliding step; =10 would give 50 fps / 2s windows (SharpAC scale)
N_FFT        = 100   # Doppler bins
NOISE_LEVEL  = -2    # log10 noise floor

# Post-Doppler 2-D Gaussian smooth
SMOOTH_SIGMA_T = 0.5
SMOOTH_SIGMA_V = 0.0   # do not blur velocity axis — preserves spectral resolution

Tc      = 1/500    # packet interval [s]
fc      = 5.18e9   # carrier frequency [Hz]
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
        win = win - win.mean(axis=0, keepdims=True)

        spec  = fft(win * hann_win, n=n_fft, axis=0)
        spec  = fftshift(spec, axes=0)

        power = np.abs(spec * np.conj(spec))
        profiles[k] = power.sum(axis=1)

    return profiles


def get_source_folder(filepath):
    """Determine if file came from save_classroom or save_office folder."""
    if '/save_classroom/' in str(filepath) or '\\save_classroom\\' in str(filepath):
        return 'save_classroom'
    elif '/save_office/' in str(filepath) or '\\save_office\\' in str(filepath):
        return 'save_office'
    else:
        return 'Unknown'


def parse_activity(filename):
    """Extract activity name from filename for display."""
    activity_letter = Path(filename).stem
    return ACTIVITY_MAP.get(activity_letter, 'Unknown')


def process_one_file(file_path):
    """Process ONE file through the pipeline. Returns detailed timing for each stage."""
    filename = file_path.name
    filestem = file_path.stem   # e.g. 'A' from 'A.npy' — used for output filename
    source_folder = get_source_folder(file_path)
    activity_name = parse_activity(filename)

    print(f'Processing: {filename}  ({source_folder} / {activity_name})')

    data = np.load(file_path)
    print(f'  Raw shape: {data.shape}')

    timings = {
        'File': filename,
        'Source_Folder': source_folder,
        'Activity': activity_name
    }

    total_start = time.time()

    stage1_start = time.time()
    single_ratio, bad_s, cnt_s, _ = ratio_stage(
        data, THRESHOLD, VOTE_PACKETS, VOTE_FRACTION, CLIP_MULT, SG_WINDOW, SG_POLY)
    stage1_time = time.time() - stage1_start
    timings['Stage1_SingleRatio_seconds'] = round(stage1_time, 2)

    stage2_start = time.time()
    double_ratio, bad_d, cnt_d, _ = ratio_stage(
        single_ratio, THRESHOLD, VOTE_PACKETS, VOTE_FRACTION, CLIP_MULT, SG_WINDOW, SG_POLY)
    stage2_time = time.time() - stage2_start
    timings['Stage2_DoubleRatio_seconds'] = round(stage2_time, 2)

    stage3_start = time.time()
    double_ratio, n_flagged = temporal_clean(double_ratio, T_CLIP_MULT)
    stage3_time = time.time() - stage3_start
    timings['Stage3_Temporal_seconds'] = round(stage3_time, 2)

    dr_folder = OUTPUT_BASE / source_folder / 'double_ratio'
    dr_folder.mkdir(parents=True, exist_ok=True)
    dr_path = dr_folder / (filestem + '.npz')
    np.savez_compressed(dr_path, double_ratio=double_ratio)

    stage4_start = time.time()
    raw_power = compute_doppler_raw(double_ratio, NUM_SYMBOLS, SLIDING, N_FFT)

    # Per-window normalisation avoids baking in scene-specific absolute power
    # (classroom vs office), so the model generalises across environments.
    noise_floor = mt.pow(10, NOISE_LEVEL)
    win_max = raw_power.max(axis=1, keepdims=True)
    win_max = np.where(win_max == 0, 1.0, win_max)
    doppler = np.clip(raw_power / win_max, noise_floor, 1.0)

    if SMOOTH_SIGMA_T > 0 or SMOOTH_SIGMA_V > 0:
        doppler = gaussian_filter(doppler, sigma=(SMOOTH_SIGMA_T, SMOOTH_SIGMA_V))
        doppler = np.clip(doppler, noise_floor, 1.0)

    stage4_time = time.time() - stage4_start
    timings['Stage4_Doppler_seconds'] = round(stage4_time, 2)

    dop_folder = OUTPUT_BASE / source_folder / 'doppler'
    dop_folder.mkdir(parents=True, exist_ok=True)
    dop_path = dop_folder / (filestem + '.npz')
    np.savez_compressed(
        dop_path,
        doppler=doppler,
        delta_v=delta_v,
        Tc=Tc,
        num_symbols=NUM_SYMBOLS,
        sliding=SLIDING,
        n_fft=N_FFT,
        source_folder=source_folder,
        activity=activity_name
    )

    timings['Total_Time_seconds'] = round(time.time() - total_start, 2)
    print(f'  Done ({timings["Total_Time_seconds"]:.2f}s)')

    return timings


def main():
    print(f'Input folder:  {BASE_PATH.absolute()}')
    print(f'Output folder: {OUTPUT_BASE.absolute()}')

    (OUTPUT_BASE / 'save_classroom' / 'double_ratio').mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / 'save_classroom' / 'doppler').mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / 'save_office' / 'double_ratio').mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / 'save_office' / 'doppler').mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / 'csv').mkdir(parents=True, exist_ok=True)

    classroom_files = sorted((BASE_PATH / 'save_classroom').glob('*.npy')) if (BASE_PATH / 'save_classroom').exists() else []
    office_files    = sorted((BASE_PATH / 'save_office').glob('*.npy'))    if (BASE_PATH / 'save_office').exists() else []
    all_files = classroom_files + office_files

    if not all_files:
        print(f'No .npy files found in {BASE_PATH}/save_classroom/ or {BASE_PATH}/save_office/')
        return

    print(f'Found {len(classroom_files)} files in save_classroom/, {len(office_files)} in save_office/')
    print(f'Velocity resolution: {delta_v} m/s per bin  Doppler range: ±{round(N_FFT/2 * delta_v, 3)} m/s')

    all_timings = []
    success_count = error_count = 0

    for idx, file_path in enumerate(all_files, 1):
        print(f'\nFILE {idx}/{len(all_files)}')
        try:
            timing_data = process_one_file(file_path)
            all_timings.append(timing_data)
            success_count += 1
        except Exception as e:
            print(f'ERROR processing {file_path.name}: {e}')
            import traceback
            traceback.print_exc()
            error_count += 1

    csv_folder = OUTPUT_BASE / 'csv'

    classroom_timings = [t for t in all_timings if t['Source_Folder'] == 'save_classroom']
    office_timings    = [t for t in all_timings if t['Source_Folder'] == 'save_office']

    column_order = [
        'File', 'Activity',
        'Stage1_SingleRatio_seconds',
        'Stage2_DoubleRatio_seconds',
        'Stage3_Temporal_seconds',
        'Stage4_Doppler_seconds',
        'Total_Time_seconds'
    ]

    if classroom_timings:
        classroom_df = pd.DataFrame(classroom_timings)[column_order]
        classroom_df.to_csv(csv_folder / 'save_classroom_timing.csv', index=False)

    if office_timings:
        office_df = pd.DataFrame(office_timings)[column_order]
        office_df.to_csv(csv_folder / 'save_office_timing.csv', index=False)

    dr_files = (len(list((OUTPUT_BASE / 'save_classroom' / 'double_ratio').glob('*.npz'))) +
                len(list((OUTPUT_BASE / 'save_office' / 'double_ratio').glob('*.npz'))))
    dop_files = (len(list((OUTPUT_BASE / 'save_classroom' / 'doppler').glob('*.npz'))) +
                 len(list((OUTPUT_BASE / 'save_office' / 'doppler').glob('*.npz'))))

    print(f'\nProcessed {success_count} files ({error_count} errors)')
    print(f'double_ratio: {dr_files} files  doppler: {dop_files} files')

    if classroom_timings:
        ct = [t['Total_Time_seconds'] for t in classroom_timings]
        print(f'save_classroom: avg={np.mean(ct):.2f}s min={np.min(ct):.2f}s max={np.max(ct):.2f}s')

    if office_timings:
        ot = [t['Total_Time_seconds'] for t in office_timings]
        print(f'save_office: avg={np.mean(ot):.2f}s min={np.min(ot):.2f}s max={np.max(ot):.2f}s')


if __name__ == '__main__':
    main()
