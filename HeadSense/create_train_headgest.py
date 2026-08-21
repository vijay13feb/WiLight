"""
Create Train / Val / Test dataset from HeadGest Doppler spectrograms.

Input  : Doppler .npz files named  signal_{phase}_{gesture_code}_ant0.npz
         e.g.  signal_phase1_FW_ant0.npz,  signal_phase1_ND_ant0.npz

Output : Dataset directory with train/val/test splits + label / file-list pickles

Each saved window has shape (1, window_length, 100):
    dim-0 - antenna channel (always 1 for HeadGest)
    dim-1 - Doppler frames  (window_length)
    dim-2 - 100 velocity bins

Windows are pooled across all gestures and split randomly (stratified), not
temporally.

Gesture codes
  FW = Forward       LD = Looking Down   LL = Looking Left
  LR = Looking Right LU = Looking Up     ND = Nodding
  SH = Shaking

Usage examples
  python3 create_train_headgest.py phase1 all 100 10
  python3 create_train_headgest.py phase1 FW,ND,SH 100 10
  python3 create_train_headgest.py phase1 all 100 10 --val_ratio 0.15 --test_ratio 0.15
"""

import argparse
import re
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent

GESTURE_MAP = {
    'FW': 'Forward',
    'LD': 'Looking_Down',
    'LL': 'Looking_Left',
    'LR': 'Looking_Right',
    'LU': 'Looking_Up',
    'ND': 'Nodding',
    'SH': 'Shaking',
}


def parse_filename(name):
    """signal_{phase}_{gesture_code}_ant{antenna}.npz -> (phase, gesture_code, antenna)."""
    m = re.match(r'^signal_(phase\d+)_([A-Z]+)_ant(\d+)\.npz$', name)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None


def save_pickle(obj, path):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp)


def main():
    parser = argparse.ArgumentParser(
        description='Build train/val/test splits from HeadGest Doppler .npz files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('phase',
        help='Phase name, e.g. phase1 or phase2.')
    parser.add_argument('gestures',
        help='Comma-separated gesture codes (FW,LD,LL,LR,LU,ND,SH). Use "all" for all.')
    parser.add_argument('window_length', type=int,
        help='Doppler frames per window (e.g. 100).')
    parser.add_argument('stride_length', type=int,
        help='Stride between windows in Doppler frames (e.g. 10).')
    parser.add_argument('--input_dir', default='output/doppler',
        help='Folder containing input .npz Doppler files (phase subfolders).')
    parser.add_argument('--output_dir', default='output/doppler_train',
        help='Root output folder; a sub-folder named after the phase is created here.')
    parser.add_argument('--val_ratio', type=float, default=0.15,
        help='Fraction of all windows for validation.')
    parser.add_argument('--test_ratio', type=float, default=0.15,
        help='Fraction of all windows for test.')
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed.')
    args = parser.parse_args()

    input_dir = Path(args.input_dir) / args.phase
    phase_out = Path(args.output_dir) / args.phase

    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')

    if args.gestures.lower() == 'all':
        requested_gestures = set()
        for f in input_dir.glob(f'signal_{args.phase}_*_ant*.npz'):
            parsed = parse_filename(f.name)
            if parsed:
                _, gesture_code, _ = parsed
                if gesture_code in GESTURE_MAP:
                    requested_gestures.add(gesture_code)
        requested_gestures = sorted(requested_gestures)
    else:
        requested_gestures = [g.strip().upper() for g in args.gestures.split(',')]

    if not requested_gestures:
        raise ValueError(f'No gestures found for phase {args.phase}')

    print(f'Phase: {args.phase}  Gestures: {requested_gestures}')
    print(f'Window: {args.window_length} frames  Stride: {args.stride_length} frames')

    label_dict = {g: idx for idx, g in enumerate(requested_gestures)}

    for split in ('train', 'val', 'test'):
        (phase_out / split).mkdir(parents=True, exist_ok=True)

    # Group by (phase, gesture_code). With single antenna (ant0), each
    # recording has exactly one file.
    recordings = defaultdict(lambda: defaultdict(dict))

    for f in sorted(input_dir.glob(f'signal_{args.phase}_*_ant*.npz')):
        parsed = parse_filename(f.name)
        if not parsed:
            print(f'  [warn] skipping malformed filename: {f.name}')
            continue

        phase, gesture_code, antenna_idx = parsed
        if phase != args.phase:
            continue
        if gesture_code not in requested_gestures:
            continue

        recordings[gesture_code][(phase, gesture_code)][antenna_idx] = f

    all_windows = []
    all_labels  = []
    global_n_antennas = None

    for gesture_code in requested_gestures:
        gesture_name = GESTURE_MAP.get(gesture_code, gesture_code)
        label_num    = label_dict[gesture_code]

        if gesture_code not in recordings:
            print(f'  [{gesture_code}] {gesture_name}: no files found, skipping')
            continue

        recording_groups = recordings[gesture_code]

        for recording_key, antenna_files in recording_groups.items():
            _, gesture_k = recording_key
            available_antennas = sorted(antenna_files.keys())
            n_ant = len(available_antennas)

            if global_n_antennas is None:
                global_n_antennas = n_ant
            elif global_n_antennas != n_ant:
                print(f'  [warn] antenna count mismatch ({n_ant} vs {global_n_antennas})')

            if n_ant == 0:
                print(f'  [warn] {gesture_k}: no antenna files, skipping')
                continue

            try:
                raw_doppler = []
                for ant_idx in available_antennas:
                    data = np.load(antenna_files[ant_idx])
                    if 'doppler' not in data:
                        raise KeyError(f"'doppler' key not found in {antenna_files[ant_idx].name}")
                    raw_doppler.append(data['doppler'].astype(np.float32))

                min_frames = min(d.shape[0] for d in raw_doppler)
                doppler_mc = np.stack([d[:min_frames] for d in raw_doppler], axis=0)
            except Exception as e:
                print(f'  [error] {gesture_k}: failed to load - {e}')
                continue

            n_frames  = min_frames
            n_windows = (n_frames - args.window_length) // args.stride_length + 1

            if n_windows <= 0:
                print(f'  [warn] {gesture_k}: not enough frames ({n_frames}) '
                      f'for window {args.window_length}, skipping')
                continue

            for k in range(n_windows):
                start = k * args.stride_length
                w = doppler_mc[:, start : start + args.window_length, :]
                all_windows.append(w)
                all_labels.append(label_num)

            print(f'  [{gesture_code}] {gesture_k}: {n_ant} antenna(s), '
                  f'{n_frames} frames, {n_windows} windows')

    if not all_labels:
        raise RuntimeError(
            f'No windows collected for phase {args.phase}. '
            f'Check input directory and file format.')

    all_labels = np.array(all_labels)
    total = len(all_labels)
    print(f'Total windows collected: {total}')

    # Hold out val+test, then split that remainder into val / test.
    temp_ratio = args.val_ratio + args.test_ratio
    train_idx, temp_idx = train_test_split(
        np.arange(total),
        test_size=temp_ratio,
        stratify=all_labels,
        random_state=args.seed,
    )
    val_frac = args.val_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - val_frac,
        stratify=all_labels[temp_idx],
        random_state=args.seed,
    )

    split_indices = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    split_files  = {s: [] for s in ('train', 'val', 'test')}
    split_labels = {s: [] for s in ('train', 'val', 'test')}

    for split_name, idxs in split_indices.items():
        for cnt, i in enumerate(idxs):
            out_path = phase_out / split_name / f'{cnt}.pkl'
            save_pickle(all_windows[i], out_path)
            split_files[split_name].append(str(out_path))
            split_labels[split_name].append(int(all_labels[i]))

    n_antennas_final = global_n_antennas or 1
    save_pickle(label_dict,       phase_out / 'label_dict.pkl')
    save_pickle(n_antennas_final, phase_out / 'n_antennas.pkl')

    for split_name in ('train', 'val', 'test'):
        save_pickle(split_labels[split_name], phase_out / f'labels_{split_name}.pkl')
        save_pickle(split_files[split_name],  phase_out / f'files_{split_name}.pkl')

    print(f'Dataset saved to: {phase_out}')
    import collections
    for split_name in ('train', 'val', 'test'):
        n = len(split_labels[split_name])
        dist = collections.Counter(split_labels[split_name])
        g_dist = {k: dist[v] for k, v in label_dict.items() if v in dist}
        print(f'  {split_name:5s}: {n:6d} windows  {g_dist}')


if __name__ == '__main__':
    main()
