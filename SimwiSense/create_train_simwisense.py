"""
Create Train / Val / Test dataset from Doppler spectrograms.

Input  : output_data/save_classroom/doppler/, output_data/save_office/doppler/
         Files named A.npz, B.npz ... T.npz (one per activity per scenario)

Output : output_data/doppler_train/{scenario}/train|val|test/ + label / file-list pickles

Each saved window has shape (1, window_length, 100):
    dim-0 - antenna channel (always 1)
    dim-1 - Doppler frames  (window_length)
    dim-2 - 100 velocity bins

Windows are pooled across all activities and split randomly (stratified),
not temporally, to avoid drift between the start and end of a recording.
Raw Doppler files are at 500 Hz (SLIDING=1 in batch_processing_simwisense.py);
temporal_stride subsamples before windowing (default 10 -> 50 Hz, so a
100-frame window covers 2 s).
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

SCRIPT_DIR  = Path(__file__).resolve().parent
INPUT_BASE  = SCRIPT_DIR / 'output_data'
OUTPUT_BASE = SCRIPT_DIR / 'output_data' / 'doppler_train'

SCENARIOS = ['save_classroom', 'save_office']

ACTIVITY_MAP = {
    'A': 'Push_forward', 'B': 'Rotate', 'C': 'Hands_up_and_down', 'D': 'Waive',
    'E': 'Brush', 'F': 'Clap', 'G': 'Sit', 'H': 'Eat', 'I': 'Drink', 'J': 'Kick',
    'K': 'Bend_forward', 'L': 'Wash_hands', 'M': 'Call', 'N': 'Browsing_phone',
    'O': 'Check_wrist', 'P': 'Read', 'Q': 'Waive_while_sitting', 'R': 'Writing',
    'S': 'Side_bend', 'T': 'Standing',
}


def save_pickle(obj, path):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp)


def process_scenario(scenario, activities, window_length, stride_length,
                     val_ratio, test_ratio, seed, temporal_stride=10):
    """Process one scenario folder and write train/val/test splits."""
    doppler_dir  = INPUT_BASE / scenario / 'doppler'
    scenario_dir = OUTPUT_BASE / scenario

    print(f'\nScenario: {scenario}  Activities: {activities}')
    print(f'Window: {window_length} frames  Stride: {stride_length} frames')

    if not doppler_dir.exists():
        print(f'  [warn] doppler dir not found: {doppler_dir} — skipping scenario.')
        return

    label_dict = {act: idx for idx, act in enumerate(activities)}

    for split in ('train', 'val', 'test'):
        (scenario_dir / split).mkdir(parents=True, exist_ok=True)

    all_windows = []   # each entry: numpy (1, window_length, 100)
    all_labels  = []   # integer label

    for act_letter in activities:
        act_name  = ACTIVITY_MAP.get(act_letter, act_letter)
        label_num = label_dict[act_letter]

        npz_path = doppler_dir / f'{act_letter}.npz'
        if not npz_path.exists():
            print(f'  [{act_letter}] {act_name}: file not found: {npz_path}, skipping')
            continue

        doppler_2d = np.load(npz_path)['doppler'].astype(np.float32)   # (n_frames, 100)

        if temporal_stride > 1:
            doppler_2d = doppler_2d[::temporal_stride, :]

        n_frames  = doppler_2d.shape[0]
        n_windows = (n_frames - window_length) // stride_length + 1

        if n_windows <= 0:
            print(f'  [{act_letter}] {act_name}: not enough frames ({n_frames}) '
                  f'for window {window_length}, skipping')
            continue

        for k in range(n_windows):
            start = k * stride_length
            w = doppler_2d[start : start + window_length, :]
            w = np.expand_dims(w, axis=0)   # (1, window_length, 100)
            all_windows.append(w)
            all_labels.append(label_num)

        print(f'  [{act_letter}] {act_name}: {n_frames} frames, {n_windows} windows')

    if not all_labels:
        print(f'  [error] No windows collected for scenario {scenario}. '
              f'Check input dir and activity letters.')
        return

    all_labels = np.array(all_labels)
    total = len(all_labels)
    print(f'Total windows collected: {total}')

    # Hold out val+test, then split that remainder into val / test.
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx = train_test_split(
        np.arange(total),
        test_size=temp_ratio,
        stratify=all_labels,
        random_state=seed,
    )
    val_frac = val_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - val_frac,
        stratify=all_labels[temp_idx],
        random_state=seed,
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

    save_pickle(label_dict, scenario_dir / 'label_dict.pkl')
    save_pickle(1, scenario_dir / 'n_antennas.pkl')   # always 1 for SimWiSense

    for split_name in ('train', 'val', 'test'):
        save_pickle(split_labels[split_name], scenario_dir / f'labels_{split_name}.pkl')
        save_pickle(split_files[split_name],  scenario_dir / f'files_{split_name}.pkl')

    import collections
    print(f'Dataset saved to: {scenario_dir}')
    for split_name in ('train', 'val', 'test'):
        n    = len(split_labels[split_name])
        dist = collections.Counter(split_labels[split_name])
        act_dist = {k: dist[v] for k, v in label_dict.items() if v in dist}
        print(f'  {split_name:5s}: {n:5d} windows  {act_dist}')


def main():
    parser = argparse.ArgumentParser(
        description='Build train/val/test splits from Doppler .npz files '
                    'for save_classroom and save_office scenarios.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('activities',
        help='Comma-separated activity letters, e.g. A,C,D,F,G. Use "all" for all.')
    parser.add_argument('window_length', type=int,
        help='Doppler frames per window (e.g. 100).')
    parser.add_argument('stride_length', type=int,
        help='Stride between windows in Doppler frames (e.g. 10).')
    parser.add_argument('--val_ratio', type=float, default=0.15,
        help='Fraction of all windows for validation.')
    parser.add_argument('--test_ratio', type=float, default=0.15,
        help='Fraction of all windows for test.')
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed for reproducible splits.')
    parser.add_argument('--scenarios', default='all',
        help='Comma-separated scenarios to process, e.g. save_classroom,save_office. '
             'Use "all" for both.')
    parser.add_argument('--temporal_stride', type=int, default=10,
        help='Subsample Doppler frames by this factor before windowing. '
             'Raw frames are 500 Hz; temporal_stride=10 -> 50 Hz, so a '
             '100-frame window covers 2 s. Use 1 for the original 0.2 s windows.')
    args = parser.parse_args()

    scenarios = SCENARIOS if args.scenarios.lower() == 'all' \
        else [s.strip() for s in args.scenarios.split(',')]

    activities = sorted(ACTIVITY_MAP.keys()) if args.activities.lower() == 'all' \
        else [a.strip().upper() for a in args.activities.split(',')]

    print(f'Input base : {INPUT_BASE}')
    print(f'Output base: {OUTPUT_BASE}')
    print(f'Scenarios  : {scenarios}')
    print(f'Activities : {activities}')

    for scenario in scenarios:
        process_scenario(
            scenario        = scenario,
            activities      = activities,
            window_length   = args.window_length,
            stride_length   = args.stride_length,
            val_ratio       = args.val_ratio,
            test_ratio      = args.test_ratio,
            seed            = args.seed,
            temporal_stride = args.temporal_stride,
        )

    print('\nAll scenarios done.')


if __name__ == '__main__':
    main()
