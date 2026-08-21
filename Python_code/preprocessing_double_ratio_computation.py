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
import time


# In[2]:


# CSI Phase ratio 
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

    # 4) produce cleaned matrix
    kept_idx = np.sort(kept_idx)
    removed_idx = np.sort(removed_idx)
    cleaned_phi = phi[:, kept_idx] if kept_idx.size else np.empty((T,0))
   
    # return cleaned_phi, kept_idx, removed_idx, {'values': vals, 'method': metric, 'thresh': thresh}
    return cleaned_phi


# In[3]:


# Filtering 
import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt

def _design_sos(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    if lowcut is None and highcut is None:
        raise ValueError("At least one of lowcut or highcut must be set")
    if lowcut is not None and lowcut <= 0 and highcut is None:
        # lowpass only (lowcut<=0 means pass DC)
        pass
    # choose filter type
    if lowcut is None:
        # lowpass
        if highcut >= nyq:
            raise ValueError("highcut must be < Nyquist (fs/2)")
        Wn = highcut / nyq
        btype = "lowpass"
    elif highcut is None:
        # highpass
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

def filter_phase(phi_ratio, fs=150.0, lowcut=1.0, highcut=30.0,
                 order=4, axis_time=0, fill_nan='interp'):
    phi = np.asarray(phi_ratio)
    # Move time to axis 0
    if axis_time != 0:
        phi = np.moveaxis(phi, axis_time, 0)

    T = phi.shape[0]
    if T < 3:
        raise ValueError("Time length must be >= 3 to perform filtering")

    # handle NaNs: simple linear interpolation along time axis if requested
    if np.isnan(phi).any():
        if fill_nan == 'interp':
            # perform 1D interpolation for each flattened column
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


# In[5]:


# smoothing 

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
    window_length=150,          # in samples (will be adjusted to odd and <= T)
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
            mode='interp'  
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


# In[6]:


# PCA 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler

def apply_pca_to_filtered(double_ratio_filtered,
                          n_components=100,
                          use_incremental=False,
                          batch_size=150,
                          scale=True): 
    X = np.asarray(double_ratio_filtered)  # (T, P)
    if X.ndim != 2:
        raise ValueError("double_ratio_filtered must be 2D (time, features)")

 
    scaler = None
    if scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = scaler.fit_transform(X)
    else:
        Xs = X - np.mean(X, axis=0, keepdims=True)

    # choose PCA implementation
    if use_incremental:
        ipca = IncrementalPCA(n_components=n_components)
    
        n_samples = Xs.shape[0]
        for i in range(0, n_samples, batch_size):
            ipca.partial_fit(Xs[i:i+batch_size])
        scores = ipca.transform(Xs)
        reconstructed = ipca.inverse_transform(scores)
        pca = ipca
    else:
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(Xs)         
        reconstructed = pca.inverse_transform(scores)

    if scale:
        reconstructed = scaler.inverse_transform(reconstructed)

    return scores, pca, scaler, reconstructed


# In[ ]:


# read the files
path = './input_data'
save_path = './preprocessed'
path_files= os.listdir(path)
antenna =4
data_file=['S1', 'S2','S2', 'S4', 'S5', 'S6', 'S7']

for name in path_files:
        print(name)
        for j in data_file:
            if j in name:
                name_path = f'{path}/{name}'
                with open(name_path, 'rb') as file:
                    df = pickle.load(file)
                df = np.transpose(df, (2, 1, 0))
                for i in range(0,antenna): 
                    df_antenna = df[i]
                    start = time.perf_counter()
                    phase = np.angle(df_antenna)
                    phase= phase_ratio(phase)
            
                    phase = phase_ratio(phase)
                    phase = filter_phase(phase)
                    phase = filter_double_ratio_savgol(phase)
                    phase=apply_pca_to_filtered(phase)[0]
                    end = time.perf_counter()
                    print(end-start, 'second')
                    save_file = f'{save_path}/{j}/{name[:-4]}_{i}.npz'

                    np.savez_compressed(save_file, data=phase.astype(np.float32))

