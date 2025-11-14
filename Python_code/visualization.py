#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os 
import joblib as jb
from collections import defaultdict
from itertools import groupby
from math import sqrt, atan2
import matplotlib.pyplot as plt
import pywt
from scipy.signal import savgol_filter
import pickle
import sys
import types


# In[ ]:


path = '/preprocessed/S2'
os.makedirs("./plots", exist_ok=True)
save_path = './plots'
path_files= os.listdir(path)
antenna =4


# In[ ]:


path = '/home/vijay/paper_jc/Sensys/data'
path_files= os.listdir(path)
data_file=[]
for name in path_files:
    if 'signal_S2aA' in name and 'signal_S2aF' in name:
        print(name)
        name_path = f'{path}/{name}'
        with open(name_path, 'rb') as file:
            df = pickle.load(file)
        df = np.transpose(df, (2, 1, 0))
        antennas = 4
        df_antenna = df[0]
        phase = np.angle(df_antenna)
        data_file.append(phase)


# In[ ]:


## remove some of the subcarrier 
import numpy as np
def wrap_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def phase_ratio(phase, unwrap=True):
    """Return phi_ratio shape (T, K//2) for input phase (T, K)."""
    if phase.ndim != 2:
        raise ValueError("Expect phase shape (time, subcarriers)")
    T, K = phase.shape
    if K % 2 != 0:
        phase = phase[:, :-1]
        K -= 1
    if unwrap:
        phase = np.unwrap(phase, axis=0)
    even = phase[:, 0:K:2]
    odd  = phase[:, 1:K:2]
    # robust wrapping via complex exponential
    phi_ratio = np.angle(np.exp(1j * (even - odd)))
    return phi_ratio
def circular_variance(phi):
    """
    phi: (T, M) phase values in radians
    returns: circ_var (M,) in [0,1], where 0 => low spread, 1 => uniform (high spread)
    """
    R = np.abs(np.mean(np.exp(1j * phi), axis=0))
    circ_var = 1.0 - R
    return circ_var

def mad_metric(phi):
    """Median absolute deviation (on unwrapped phi along time) per column."""
    # unwrap temporally before MAD
    phi_un = np.unwrap(phi, axis=0)
    med = np.median(phi_un, axis=0)
    mad = np.median(np.abs(phi_un - med[np.newaxis, :]), axis=0)
    return mad

def filter_noisy_pairs(phase, *,
                       unwrap=True,
                       metric='mad',
                       thresh=None,
                       pct_remove=None,
                       mad_scale=1.4826,
                       verbose=False):
    # 1) compute ratio
    phi = phase  # shape (T, M)
    T, M = phi.shape

    # 2) compute metric
    if metric == 'circvar':
        vals = circular_variance(phi)
    elif metric == 'mad':
        vals = mad_metric(phi) * mad_scale
    elif metric == 'std':
        vals = np.std(np.unwrap(phi, axis=0), axis=0)
    else:
        raise ValueError("metric must be 'circvar', 'mad', or 'std'")

    # 3) decide threshold
    if pct_remove is not None:
        if not (0 < pct_remove < 100):
            raise ValueError("pct_remove in (0,100)")
        # number to remove
        nrm = int(np.round(M * pct_remove / 100.0))
        # if nrm == 0, keep all
        if nrm == 0:
            kept_idx = np.arange(M)
            removed_idx = np.array([], dtype=int)
        else:
            # argsort descending: worst first
            worst_order = np.argsort(vals)[::-1]
            removed_idx = worst_order[:nrm]
            kept_idx = np.setdiff1d(np.arange(M), removed_idx, assume_unique=True)
    else:
        # use absolute thresh (if provided) else adaptive thresh = median + 3*IQR
        if thresh is None:
            med = np.median(vals)
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            iqr = q3 - q1
            thresh = med + 3.0 * iqr
            if verbose:
                print(f"Adaptive thresh (median+3*IQR) = {thresh:.4g}")
        kept_idx = np.where(vals <= thresh)[0]
        removed_idx = np.where(vals > thresh)[0]

    kept_idx = np.sort(kept_idx)
    removed_idx = np.sort(removed_idx)
    cleaned_phi = phi[:, kept_idx] if kept_idx.size else np.empty((T,0))
   
 
    return cleaned_phi


single_ratio=[]
for name_file in data_file:
    single_ratio.append(phase_ratio(name_file))

double_ratio=[]
for name_file in single_ratio:
    double_ratio.append(filter_noisy_pairs(phase_ratio(name_file)))
plot_double(double_ratio)


# In[ ]:


import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt

def _design_sos(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    if lowcut is None and highcut is None:
        raise ValueError("At least one of lowcut or highcut must be set")
    if lowcut is not None and lowcut <= 0 and highcut is None:

        pass

    if lowcut is None:

        if highcut >= nyq:
            raise ValueError("highcut must be < Nyquist (fs/2)")
        Wn = highcut / nyq
        btype = "lowpass"
    elif highcut is None:

        if lowcut <= 0:
            raise ValueError("lowcut must be > 0 for highpass")
        Wn = lowcut / nyq
        btype = "highpass"
    else:
        if not (0 < lowcut < highcut < nyq):
            raise ValueError("Require 0 < lowcut < highcut < Nyquist (fs/2)")
        Wn = [lowcut / nyq, highcut / nyq]
        btype = "bandpass"
    sos = butter(order, Wn, btype=btype, output="sos")
    return sos

def filter_phase(phi_ratio, fs=150.0, lowcut=3.0, highcut=40.0,
                 order=5, axis_time=0, fill_nan='interp'):
    phi = np.asarray(phi_ratio)

    if axis_time != 0:
        phi = np.moveaxis(phi, axis_time, 0)

    T = phi.shape[0]
    if T < 3:
        raise ValueError("Time length must be >= 3 to perform filtering")

    if np.isnan(phi).any():
        if fill_nan == 'interp':
  
            orig_shape = phi.shape
            flat = phi.reshape(T, -1)
            for col in range(flat.shape[1]):
                x = flat[:, col]
                if np.all(np.isnan(x)):
                    flat[:, col] = 0.0
                    continue
                nans = np.isnan(x)
                if nans.any():
                    notnan_idx = np.flatnonzero(~nans)
                    if notnan_idx.size == 0:
                        flat[:, col] = 0.0
                    else:
                        xp = notnan_idx
                        fp = x[~nans]
                        xi = np.arange(T)
                        flat[:, col] = np.interp(xi, xp, fp)
            phi = flat.reshape(orig_shape)
        elif fill_nan == 'zero':
            phi = np.nan_to_num(phi, nan=0.0)
        else:
            raise ValueError("phi contains NaN and fill_nan is None")

    # unwrap along time
    phi_unwrapped = np.unwrap(phi, axis=0).astype(np.float64, copy=False)

    # design sos
    sos = _design_sos(lowcut, highcut, fs, order=order)
    try:
        # vectorized call: filter all columns at once along axis=0
        filtered = sosfiltfilt(sos, phi_unwrapped, axis=0)
    except ValueError as e:
 
        forward = sosfilt(sos, phi_unwrapped, axis=0)
        reversed_ = forward[::-1].copy()
        backward = sosfilt(sos, reversed_, axis=0)
        filtered = backward[::-1]

    # wrap back into (-pi, pi]
    wrapped = (filtered + np.pi) % (2 * np.pi) - np.pi

    # move axis back if needed
    if axis_time != 0:
        wrapped = np.moveaxis(wrapped, 0, axis_time)
    return wrapped


# In[ ]:


double_ratio_filter=[]
for name_file in double_ratio:
    double_ratio_filter.append(filter_phase(name_file))


# In[ ]:


## smooth filter 
import numpy as np
from scipy.signal import savgol_filter

def _interp_nans_along_axis(x, axis=0):
    x = np.asarray(x)
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
    T = x.shape[0]
    flat = x.reshape(T, -1)
    xi = np.arange(T)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        if np.isnan(col).any():
            valid = ~np.isnan(col)
            if valid.sum() == 0:
                flat[:, c] = 0.0
            elif valid.sum() == 1:
                flat[:, c] = col[valid][0]
            else:
                flat[:, c] = np.interp(xi, xi[valid], col[valid])
    out = flat.reshape((T,) + x.shape[1:])
    if axis != 0:
        out = np.moveaxis(out, 0, axis)
    return out

def _ensure_odd_window(wl, T):
    wl = int(wl)
    if wl < 1:
        wl = 1
    if wl > T:
        wl = T if (T % 2 == 1) else T-1
    if wl % 2 == 0:
        wl -= 1
    if wl < 1:
        wl = 1
    return wl

def filter_double_ratio_savgol(
    phi,                       # array of phases in radians (any shape)
    window_length=11,          # in samples (will be adjusted to odd and <= T)
    polyorder=3,
    axis_time=0,
    mode_nans='interp',        # 'interp'|'zero'|None
    unwrap=True,               # unwrap before filtering
    return_unwrapped=False     # if True, return unwrapped filtered phase (radians)
):

    phi = np.asarray(phi)
    ndim = phi.ndim
    if axis_time < 0:
        axis_time += ndim
    if axis_time < 0 or axis_time >= ndim:
        raise ValueError("axis_time out of range for input array")

    T = phi.shape[axis_time]
    if T < 1:
        raise ValueError("time axis length must be >= 1")

    # handle NaNs
    if np.isnan(phi).any():
        if mode_nans == 'interp':
            phi = _interp_nans_along_axis(phi, axis=axis_time)
        elif mode_nans == 'zero':
            phi = np.nan_to_num(phi, nan=0.0)
        else:
            raise ValueError("Input contains NaNs; set mode_nans to 'interp' or 'zero'")

    # prepare window length and polyorder
    wl = _ensure_odd_window(window_length, T)
    if polyorder >= wl:
        polyorder = max(0, wl - 1)

    # unwrap if requested
    if unwrap:
        phi = np.moveaxis(phi, axis_time, 0)      # time -> axis 0 for unwrap
        phi_unw = np.unwrap(phi, axis=0)
        phi_unw = np.moveaxis(phi_unw, 0, axis_time)
    else:
        phi_unw = phi.astype(np.float64, copy=False)

    # if wl == 1, no filtering (just return copy or wrap)
    if wl == 1:
        filtered = phi_unw.copy()
    else:
        # savgol_filter supports axis argument
        filtered = savgol_filter(
            phi_unw.astype(np.float64, copy=False),
            window_length=wl,
            polyorder=polyorder,
            axis=axis_time,
            mode='interp'   # recommended boundary handling; user can change if desired
        )

    if unwrap:
        if return_unwrapped:
            out = filtered
        else:
            out = (filtered + np.pi) % (2 * np.pi) - np.pi
    else:
        # input wasn't unwrapped; we still honor return_unwrapped flag
        out = filtered if return_unwrapped else ((filtered + np.pi) % (2 * np.pi) - np.pi)

    return out


# In[ ]:


double_ratio_smooth=[]
for name_file in double_ratio_filter:
    double_ratio_smooth.append(filter_double_ratio_savgol(name_file))

plot_filter(double_ratio_smooth)


# In[ ]:


# PCA 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler

def apply_pca_to_filtered(double_ratio_filtered,
                          n_components=300,
                          use_incremental=False,
                          batch_size=150,
                          scale=True): 
    X = np.asarray(double_ratio_filtered)  # (T, P)
    if X.ndim != 2:
        raise ValueError("double_ratio_filtered must be 2D (time, features)")

    # optionally scale (zero-mean, unit variance)
    scaler = None
    if scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = scaler.fit_transform(X)
    else:
        Xs = X - np.mean(X, axis=0, keepdims=True)

    # choose PCA implementation
    if use_incremental:
        ipca = IncrementalPCA(n_components=n_components)
        # partial fit in batches
        n_samples = Xs.shape[0]
        for i in range(0, n_samples, batch_size):
            ipca.partial_fit(Xs[i:i+batch_size])
        scores = ipca.transform(Xs)
        reconstructed = ipca.inverse_transform(scores)
        pca = ipca
    else:
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(Xs)           # (T, n_components)
        reconstructed = pca.inverse_transform(scores)

    # If scaled, invert scaling to get reconstruction back into original units
    if scale:
        reconstructed = scaler.inverse_transform(reconstructed)

    return scores, pca, scaler, reconstructed


# In[ ]:


double_ratio_pca=[]
for name_file in double_ratio_smooth:
    double_ratio_pca.append(apply_pca_to_filtered(name_file)[0])


# In[ ]:


from matplotlib.ticker import AutoLocator

def plot_multi_manual(
    data_files_list,
    save_path='.',
    save_plot='raw_phase_time',
    ylabels=None,
    yticks_list=None,
    ylims_list=None,
    xlabels=None,
    lock_ylim=True,
    legend_on_top=True,
    legend_ncol=2
):
    plt.rcParams.update({"font.size": 14})
    fs = 150

    colors = {
        'Walk': 'blue', 'Run': 'magenta', 'Jump': 'green',
        'Sitting': 'red', 'Standing': 'black', 'Wave Hands': 'red'
    }
    linestyles = {k: '-' for k in colors.keys()}
    activities = ['Walk', 'Wave Hands']

    if not isinstance(data_files_list, (list, tuple)) or len(data_files_list) == 0:
        raise ValueError("data_files_list must be a non-empty list/tuple of datasets (each dataset => [phase_walk, phase_wave])")

    n_rows = len(data_files_list)
    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 2.2 * n_rows), sharex='col')
    if n_rows == 1:
        axs = np.expand_dims(axs, 0)

    collected_handles = []
    collected_labels = []
    for i, data_pair in enumerate(data_files_list):
        if not isinstance(data_pair, (list, tuple)):
            raise ValueError(f"data_files_list[{i}] must be a list/tuple of phase arrays (got {type(data_pair)})")

        for col_idx, activity in enumerate(activities):
            ax = axs[i, col_idx]

            try:
                phase = data_pair[col_idx]
            except IndexError:
                phase = None

            if phase is not None:
                if not hasattr(phase, 'shape'):
                    raise TypeError(f"phase for dataset {i}, activity '{activity}' must be numpy-like; got {type(phase)}")

                max_rows = phase.shape[0]
                slice_end = min(450, max_rows)

                if i == 4:  
                    chan_idx = 5
                else:
                    chan_idx = 0

                if phase.ndim >= 2 and chan_idx < phase.shape[1]:
                    y_data = phase[0:slice_end, chan_idx]
                else:
               
                    y_data = phase[0:slice_end] if phase.ndim == 1 else phase[0:slice_end, 0]

                time = np.arange(len(y_data)) / fs
                color = colors.get(activity, 'black')
                linestyle = linestyles.get(activity, '-')
                ax.plot(time, y_data, alpha=0.9, linewidth=1,
                        label=activity, color=color, linestyle=linestyle,
                        marker='o', markersize=3)

            # Y-label only on left column
            if col_idx == 0:
                if ylabels and i < len(ylabels):
                    ax.set_ylabel(ylabels[i])
                else:
                    ax.set_ylabel("Phase (rad)")
            else:
                ax.set_ylabel("")

            # --- Y ticks & grid alignment ---
            # Left column: apply user-provided yticks if available, else keep defaults.
            if col_idx == 0:
                if yticks_list and i < len(yticks_list) and yticks_list[i] is not None:
                    # ensure ticks are numeric floats
                    ax.set_yticks([float(x) for x in yticks_list[i]])
                else:
                    # keep matplotlib defaults
                    pass
                # ensure left axis shows grid lines
                ax.yaxis.grid(True, which='major', alpha=0.3)
            else:
             
                left_ticks = axs[i, 0].get_yticks()
                if left_ticks is None or len(left_ticks) == 0:
             
                    ax.yaxis.set_major_locator(AutoLocator())
                    left_ticks = axs[i, 0].get_yticks()
            
                ax.set_yticks(left_ticks)
             
                ax.set_yticklabels([''] * len(left_ticks))
           
                ax.yaxis.grid(True, which='major', alpha=0.3)

            # Y limits (apply to both columns)
            if ylims_list and i < len(ylims_list):
                validated = _validate_ylim_entry(ylims_list[i], i)
                if validated is not None:
                    ax.set_ylim(validated)
                    if lock_ylim:
                        ax.set_autoscale_on(False)

            # X labels on ALL subplots
            if xlabels and i < len(xlabels):
                ax.set_xlabel(xlabels[i])
            else:
                ax.set_xlabel("Time (seconds)")

            ax.grid(True, alpha=0.3)

        # Collect handles/labels from both columns (first row that has them)
        if not collected_handles:
            h0, l0 = axs[i, 0].get_legend_handles_labels()
            h1, l1 = axs[i, 1].get_legend_handles_labels()
            combined = []
            for h, l in zip(h0, l0):
                combined.append((l, h))
            for h, l in zip(h1, l1):
                combined.append((l, h))
            seen = set()
            for label, handle in combined:
                if label not in seen:
                    collected_labels.append(label)
                    collected_handles.append(handle)
                    seen.add(label)

    # Top legend
    if legend_on_top and collected_handles:
        fig.legend(collected_handles, collected_labels, loc='upper center',
                   ncol=legend_ncol, bbox_to_anchor=(0.55, 0.96),
                   fontsize=12, frameon=True)
        for r in range(n_rows):
            for c in range(2):
                ax = axs[r, c]
                if getattr(ax, "legend_", None):
                    ax.legend_.remove()
    else:
        for r in range(n_rows):
            for c in range(2):
                ax = axs[r, c]
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(loc='upper right', frameon=True, fontsize=9, ncol=1)

    plt.tight_layout()
    if legend_on_top:
        plt.subplots_adjust(top=0.92)

    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{save_plot}.pdf")
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()
    print(f"✅ Plot saved to: {save_file}")


# In[ ]:


data_files_list = [
    data_file, 
    single_ratio,
    double_ratio,
    double_ratio_smooth,
    double_ratio_pca,
]

ylabels = [
    "Phase (rad)", "Phase (rad)", "Phase (rad)",
    "Phase (rad)", "Phase (rad)"
]

yticks_list = [
    [-3, -2, -1, 0, 1, 2, 3],
    [-0.05, 0.05, 0.15, 0.25],
    [-0.05, -0.025,0, 0.025, 0.05],
    [-0.05, -0.025,0, 0.025, 0.05],
    [-5,-2.5, 0.0, 2.5,5],
    
   
]
ylims_list = [
    None,
    None,
    (-0.05,0.05),
    (-0.05, 0.07),
None,
]

xlabels = [
    "Time (seconds)\n (a) Raw CSI phase", "Time (seconds) \n (b) CSI phase single ratio","Time (seconds) \n (c) PDR",
    "Time (seconds)\n (d) Denoised PDR", "Time (seconds)\n(e) PCA of PDR"
]

plot_multi_manual(
    data_files_list,
    save_path='./plots_preprocessing',
    save_plot='multi_manual_axes',
    ylabels=ylabels,
    yticks_list=yticks_list,
    ylims_list=ylims_list,      # <-- make sure to pass this
    xlabels=xlabels,
    legend_on_top=True,         # single legend above the figure
    lock_ylim=True              # lock autoscale when ylim applied
)

