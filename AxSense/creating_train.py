"""
Create Train / Val / Test dataset from Doppler spectrograms.

Input  : output/doppler/  —  .npz files named  signal_{scenario}_{activity}_{antenna}.npz
         where antenna ∈ {0, 1, 2, 3}  (4 synchronised WiFi receiver antennas)

Output : output/doppler/{scenario}/train|val|test/  +  label / file-list pickles

Each saved window has shape  (n_antennas, window_length, 100):
    dim-0  — antenna channel (0..3)
    dim-1  — Doppler frames  (window_length)
    dim-2  — 100 velocity bins
"""

import argparse
import collections
import re
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


ACTIVITY_MAP = {
    'A': 'Walk', 'B': 'Run',  'C': 'Jump',       'D': 'Sitting',
    'E': 'Empty_room',        'F': 'Standing',    'G': 'Wave_hands',
    'H': 'Clapping',          'I': 'Lay_down',    'J': 'Wiping',
    'K': 'Squat',             'L': 'Stretching',
}


def _activity_letter(name: str):
    m = re.match(r'^signal_([A-Za-z0-9]+)_([A-Z])_\d+\.npz$', name)
    return m.group(2) if m else None


def save_pickle(obj, path):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp)


def main():
    parser = argparse.ArgumentParser(
        description='Build multi-antenna train/val/test splits from Doppler .npz files '
                    'using random stratified split.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('scenario',
        help='Scenario name, e.g. S1a or S2a.')
    parser.add_argument('activities',
        help='Comma-separated activity letters, e.g. A,C,D,F,G. Use "all" for all.')
    parser.add_argument('window_length', type=int,
        help='Doppler frames per window (e.g. 100).  '
             'With SLIDING=1 files, 100 frames = 100/150 ≈ 0.67 s of activity.')
    parser.add_argument('stride_length', type=int,
        help='Stride between windows in Doppler frames (e.g. 10).  '
             'Reduces inter-window correlation without changing window size.')
    parser.add_argument('--input_dir', default='output/doppler',
        help='Folder containing input .npz Doppler files.')
    parser.add_argument('--output_dir', default='output/doppler',
        help='Root output folder; a sub-folder named after the scenario is created here.')
    parser.add_argument('--val_ratio', type=float, default=0.15,
        help='Fraction of all windows for validation.')
    parser.add_argument('--test_ratio', type=float, default=0.15,
        help='Fraction of all windows for test.')
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed for reproducible splits.')
    args = parser.parse_args()

    input_dir    = Path(args.input_dir)
    scenario_dir = Path(args.output_dir) / args.scenario

    if args.activities.lower() == 'all':
        requested_activities = sorted({
            _activity_letter(f.name)
            for f in input_dir.glob(f'signal_{args.scenario}_*.npz')
            if _activity_letter(f.name)
        })
    else:
        requested_activities = [a.strip().upper() for a in args.activities.split(',')]

    print(f'Scenario: {args.scenario}  Activities: {requested_activities}')
    print(f'Window: {args.window_length} frames  Stride: {args.stride_length} frames')

    label_dict = {act: idx for idx, act in enumerate(requested_activities)}

    for split in ('train', 'val', 'test'):
        (scenario_dir / split).mkdir(parents=True, exist_ok=True)

    all_windows = []   # each entry: numpy (n_antennas, window_length, 100)
    all_labels  = []   # integer label

    global_n_antennas = None

    for act_letter in requested_activities:
        act_name  = ACTIVITY_MAP.get(act_letter, act_letter)
        label_num = label_dict[act_letter]

        # Group .npz files by antenna index
        antenna_files = {}
        for f in sorted(input_dir.glob(f'signal_{args.scenario}_{act_letter}_*.npz')):
            m = re.match(r'^signal_([A-Za-z0-9]+)_([A-Z])_(\d+)\.npz$', f.name)
            if m:
                antenna_files[int(m.group(3))] = f

        if not antenna_files:
            print(f'  [{act_letter}] No files found — skipping.')
            continue

        n_ant    = len(antenna_files)
        ant_idxs = sorted(antenna_files.keys())

        if global_n_antennas is None:
            global_n_antennas = n_ant
        elif global_n_antennas != n_ant:
            print(f'    [warn] antenna count mismatch ({n_ant} vs {global_n_antennas})')

        # Load all antennas; trim to shortest
        raw        = [np.load(antenna_files[i])['doppler'].astype(np.float32) for i in ant_idxs]
        min_frames = min(d.shape[0] for d in raw)
        doppler_mc = np.stack([d[:min_frames] for d in raw], axis=0)
        # doppler_mc: (n_antennas, min_frames, 100)

        n_frames = min_frames
        n_windows = (n_frames - args.window_length) // args.stride_length + 1

        if n_windows <= 0:
            print(f'    [warn] Not enough frames ({n_frames}) for window '
                  f'{args.window_length} — skipped.')
            continue

        for k in range(n_windows):
            start = k * args.stride_length
            w = doppler_mc[:, start : start + args.window_length, :]
            all_windows.append(w)
            all_labels.append(label_num)

        print(f'  [{act_letter}] {act_name}: {n_ant} antenna(s), {n_frames} frames, {n_windows} windows')

    if not all_labels:
        raise RuntimeError('No windows collected. Check input_dir and scenario name.')

    all_labels = np.array(all_labels)
    total = len(all_labels)
    print(f'Total windows collected: {total}')

    # hold out val+test, then split that remainder into val / test
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
            out_path = scenario_dir / split_name / f'{cnt}.pkl'
            save_pickle(all_windows[i], out_path)
            split_files[split_name].append(str(out_path))
            split_labels[split_name].append(int(all_labels[i]))

    n_antennas_final = global_n_antennas or 1
    save_pickle(label_dict,        scenario_dir / 'label_dict.pkl')
    save_pickle(n_antennas_final,  scenario_dir / 'n_antennas.pkl')

    for split_name in ('train', 'val', 'test'):
        save_pickle(split_labels[split_name], scenario_dir / f'labels_{split_name}.pkl')
        save_pickle(split_files[split_name],  scenario_dir / f'files_{split_name}.pkl')

    print(f'Dataset saved to: {scenario_dir}')
    for split_name in ('train', 'val', 'test'):
        n = len(split_labels[split_name])
        dist = collections.Counter(split_labels[split_name])
        act_dist = {k: dist[v] for k, v in label_dict.items() if v in dist}
        print(f'  {split_name:5s}: {n:5d} windows  {act_dist}')


if __name__ == '__main__':
    main()
