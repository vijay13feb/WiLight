"""
Create Train / Val / Test dataset from Doppler spectrograms.

Input  : Doppler .npz files named  signal_{scenario}_{activity}_ant{antenna}.npz
         where antenna in {0, 1, 2, 3}  (4 synchronised WiFi receiver antennas)
         Activity variants (H1, H2, J1, J2) are automatically normalized to (H, J)

Output : Dataset directory with train/val/test splits + label / file-list pickles

Each saved window has shape (n_antennas, window_length, 100):
    dim-0 - antenna channel (0..3)
    dim-1 - Doppler frames  (window_length)
    dim-2 - 100 velocity bins

Windows are pooled across all activities/recordings and split randomly
(stratified), not temporally, to avoid drift between the start and end of
a recording. Multiple recordings per activity (e.g. H1, H2) are grouped
separately by (scenario, raw_activity) so antennas from the same recording
stay synchronized, then all windows contribute to one pooled dataset.

Recommended: window_length=100, stride_length=10
"""

import argparse
import collections
import re
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent

ACTIVITY_MAP = {
    'E': 'Empty_room',
    'W': 'Walking',
    'R': 'Running',
    'J': 'Jumping',
    'S': 'Sitting',
    'L': 'Standing_still',
    'H': 'Arm_gymnastics',
    'C': 'SitStand_transition',
}


def normalize_activity(act):
    """Map activity variants (H1, H2, J1, J2, ...) to their base class (H, J)."""
    act = act.upper()
    if act.startswith('H'):
        return 'H'
    if act.startswith('J'):
        return 'J'
    return act


def parse_filename(name):
    """signal_{scenario}_{activity}_ant{antenna}.npz -> (scenario, activity, antenna_idx)."""
    m = re.match(r'^signal_([A-Za-z0-9]+)_([A-Za-z0-9]+)_ant(\d+)\.npz$', name)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None


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
        help='Comma-separated activity letters (E,W,R,J,S,L,H,C). Use "all" for all. '
             'Activity variants (H1,H2,J1,J2) are automatically included and normalized.')
    parser.add_argument('window_length', type=int,
        help='Doppler frames per window (e.g. 100).')
    parser.add_argument('stride_length', type=int,
        help='Stride between windows in Doppler frames (e.g. 10).')
    parser.add_argument('--input_dir', default='output/doppler',
        help='Folder containing input .npz Doppler files.')
    parser.add_argument('--output_dir', default='output/doppler_train',
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

    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')

    if args.activities.lower() == 'all':
        requested_activities = set()
        for f in input_dir.glob(f'signal_{args.scenario}_*_ant*.npz'):
            parsed = parse_filename(f.name)
            if parsed:
                _, raw_activity, _ = parsed
                requested_activities.add(normalize_activity(raw_activity))
        requested_activities = sorted(requested_activities)
    else:
        requested_activities = [a.strip().upper() for a in args.activities.split(',')]

    if not requested_activities:
        raise ValueError(f'No activities found for scenario {args.scenario}')

    print(f'Scenario: {args.scenario}  Activities: {requested_activities}')
    print(f'Window: {args.window_length} frames  Stride: {args.stride_length} frames')

    label_dict = {act: idx for idx, act in enumerate(requested_activities)}

    for split in ('train', 'val', 'test'):
        (scenario_dir / split).mkdir(parents=True, exist_ok=True)

    # Group by (scenario, raw_activity) so multi-recording variants like H1/H2
    # stay separate recordings instead of overwriting each other.
    recordings = defaultdict(lambda: defaultdict(dict))

    for f in sorted(input_dir.glob(f'signal_{args.scenario}_*_ant*.npz')):
        parsed = parse_filename(f.name)
        if not parsed:
            print(f'  [warn] skipping malformed filename: {f.name}')
            continue

        scenario, raw_activity, antenna_idx = parsed
        if scenario != args.scenario:
            continue

        normalized = normalize_activity(raw_activity)
        if normalized not in requested_activities:
            continue

        recordings[normalized][(scenario, raw_activity)][antenna_idx] = f

    all_windows = []   # each entry: numpy (n_antennas, window_length, 100)
    all_labels  = []   # integer label
    global_n_antennas = None

    for act_letter in requested_activities:
        act_name  = ACTIVITY_MAP.get(act_letter, act_letter)
        label_num = label_dict[act_letter]

        if act_letter not in recordings:
            print(f'  [{act_letter}] {act_name}: no recordings found, skipping')
            continue

        recording_groups = recordings[act_letter]

        for recording_key, antenna_files in recording_groups.items():
            _, raw_activity = recording_key
            available_antennas = sorted(antenna_files.keys())
            n_ant = len(available_antennas)

            missing = {0, 1, 2, 3} - set(available_antennas)
            if missing:
                print(f'  [warn] {raw_activity}: missing antennas {sorted(missing)}')
            if n_ant == 0:
                print(f'  [warn] {raw_activity}: no antenna files found, skipping')
                continue

            if global_n_antennas is None:
                global_n_antennas = n_ant
            elif global_n_antennas != n_ant:
                print(f'  [warn] {raw_activity}: antenna count mismatch '
                      f'({n_ant} vs expected {global_n_antennas})')

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
                print(f'  [error] {raw_activity}: failed to load - {e}')
                continue

            n_frames = min_frames
            n_windows = (n_frames - args.window_length) // args.stride_length + 1

            if n_windows <= 0:
                print(f'  [warn] {raw_activity}: not enough frames ({n_frames}) '
                      f'for window {args.window_length}, skipping')
                continue

            for k in range(n_windows):
                start = k * args.stride_length
                w = doppler_mc[:, start : start + args.window_length, :]
                all_windows.append(w)
                all_labels.append(label_num)

            print(f'  [{act_letter}] {raw_activity}: {n_ant} antenna(s), '
                  f'{n_frames} frames, {n_windows} windows')

    if not all_labels:
        raise RuntimeError(
            f'No windows collected for scenario {args.scenario}. '
            f'Check input directory and filename format.')

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
            out_path = scenario_dir / split_name / f'{cnt}.pkl'
            save_pickle(all_windows[i], out_path)
            split_files[split_name].append(str(out_path))
            split_labels[split_name].append(int(all_labels[i]))

    n_antennas_final = global_n_antennas or 4
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
