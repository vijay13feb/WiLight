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
import psutil
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()


# In[2]:


# and '0' ==file[13]
TEST_DIR = './preprocessed/S2'

label_map = {char: idx for idx, char in enumerate("ACDFGH")}

def load_data(directory):
    data = []
    labels = []
    for file in os.listdir(directory):
        
        if file.endswith('.npz') and file[11] in 'ACDFGH' and 'a' == file[9] and '0' ==file[13]:
            # print(file)
            filepath = os.path.join(directory, file)
            z = np.load(filepath, allow_pickle=True)
            csi=z['data']
            # print(csi)
            # print(csi.shape)
            label = label_map[file[11]]
            print(label)
            data.append(csi)
            # print(data[0].shape)
            labels.append(label)
    return np.array(data), np.array(labels)

x_train, y_train = load_data(TEST_DIR)
print(x_train.shape)
import numpy as np
from sklearn.model_selection import train_test_split

# Assuming `data` is your array of shape (6, 12000, 450)
num_classes = x_train.shape[0]
samples_per_class = x_train.shape[1]
feature_dim = x_train.shape[2]

x_train_list, x_val_list = [], []
y_train_list, y_val_list = [], []

for class_idx in range(num_classes):
    x_class = x_train[class_idx]  # shape: (12000, 450)
    y_class = np.full((samples_per_class,), class_idx)  # labels: all same class

    # Split per class
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_class, y_class, test_size=0.99, random_state=42, shuffle=True
    )

    x_train_list.append(x_tr)
    x_val_list.append(x_val)
    y_train_list.append(y_tr)
    y_val_list.append(y_val)

# Combine all classes
x_train = np.vstack(x_train_list)  # shape: (6*9600, 450)
x_val = np.vstack(x_val_list)      # shape: (6*2400, 450)
y_train = np.hstack(y_train_list)  # shape: (6*9600,)
y_val = np.hstack(y_val_list) 
x_train = x_train[..., np.newaxis]
x_val   = x_val[..., np.newaxis]     # shape: (6*2400,)


#load pruned model 
from tcn import TCN  
from tensorflow.keras.models import load_model
model = load_model("./TCN/final_pruned_model.h5", custom_objects={'TCN': TCN})


# freez the classifier 
from tensorflow.keras.models import load_model
from tcn import TCN
import tensorflow as tf
import os
target_names = ("conv1d", "dense")
# 1️⃣ Freeze TCN block
for layer in model.layers:
    if "tcn" in layer.name.lower():   # freeze all TCN-related layers
        layer.trainable = False
    else:
        layer.trainable = True
for layer in model.layers:
    # conv1d and dense: re-init kernel & bias using layer's initializers (or sensible defaults)
    if layer.name in ("conv1d", "dense"):
        weights = layer.get_weights()
        if not weights:
            continue
        new_weights = []
        # layer.weights are tf.Variable objects with names like ".../kernel:0" or ".../bias:0"
        # zip over layer.weights to match initializer selection
        for var, old in zip(layer.weights, weights):
            var_name = var.name.lower()
            # choose initializer
            if "kernel" in var_name:
                init_obj = tf.keras.initializers.get(getattr(layer, "kernel_initializer", "glorot_uniform"))
            elif "bias" in var_name:
                init_obj = tf.keras.initializers.get(getattr(layer, "bias_initializer", "zeros"))
            elif "gamma" in var_name:
                init_obj = tf.keras.initializers.get(getattr(layer, "gamma_initializer", "ones"))
            elif "beta" in var_name:
                init_obj = tf.keras.initializers.get(getattr(layer, "beta_initializer", "zeros"))
            else:
                # fallback
                init_obj = tf.keras.initializers.GlorotUniform()

            # create new random array with same shape & dtype
            new_arr = init_obj(shape=old.shape, dtype=old.dtype)
            new_weights.append(np.array(new_arr))

        # set reinitialized weights
        try:
            layer.set_weights(new_weights)
            print(f"Reinitialized weights for layer: {layer.name}")
        except Exception as e:
            print(f"Could not set weights for layer {layer.name}: {e}")



import time, os, json, numpy as np
import tensorflow as tf


try:
    import psutil
    _ps = psutil.Process()
    use_psutil = True
except Exception:
    use_psutil = False
    _ps = None

try:
    from sklearn.metrics import accuracy_score, f1_score
    use_sklearn = True
except Exception:
    use_sklearn = False

# --------- FLOPs helper (forward-pass) ----------
def compute_model_flops(model, sample_input_shape=None):
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        if sample_input_shape is None:
            inp = model.input_shape
            sample_input_shape = (1,) + tuple(inp[1:])
        full_model_fn = tf.function(lambda x: model(x))
        concrete_fn = full_model_fn.get_concrete_function(tf.TensorSpec(sample_input_shape, model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(concrete_fn)
        graph_def = frozen_func.graph.as_graph_def()
        tf.compat.v1.reset_default_graph()
        g = tf.Graph()
        with g.as_default():
            tf.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops_prof = tf.compat.v1.profiler.profile(graph=g, run_meta=run_meta, cmd='op', options=opts)
            if flops_prof is None:
                return None
            return int(flops_prof.total_float_ops)
    except Exception:
        return None

# --------- Callback: memory sampling + batch counting + per-epoch timing (validation excluded) ----------
class SimpleMemTimeCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        # overall timers / counters
        self.total_train_time = 0.0            # sum of training batch times across all epochs
        self.batches = 0                       # total training batches processed
        self.mem_samples = []
        self._in_validation = False            # whether validation (test) is running right now

        # per-epoch
        self.epoch_train_times = []            # list of epoch training durations (seconds)
        self._epoch_batch_time = 0.0
        self._epoch_batches = 0

        # small guard for per-batch timer
        self._batch_start = None

        if use_psutil:
            try:
                self.mem_samples.append(_ps.memory_info().rss)
            except Exception:
                pass

    def on_epoch_begin(self, epoch, logs=None):
        # reset epoch accumulators
        self._epoch_batch_time = 0.0
        self._epoch_batches = 0

    def on_train_batch_begin(self, batch, logs=None):
        # only measure when not in validation
        if not self._in_validation:
            self._batch_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        # only measure when not in validation
        if not self._in_validation:
            if self._batch_start is None:
                # defensive: nothing to add
                return
            dt = time.time() - self._batch_start
            self._epoch_batch_time += dt
            self.total_train_time += dt
            self.batches += 1
            self._epoch_batches += 1
            self._batch_start = None

            # sample memory occasionally during training (every 10 training batches)
            if use_psutil and (self.batches % 10 == 0):
                try:
                    self.mem_samples.append(_ps.memory_info().rss)
                except Exception:
                    pass

    # Keras calls these during validation (fit's validation_data). toggle sampling off.
    def on_test_begin(self, logs=None):
        # validation (test) is starting — do not sample memory during validation
        self._in_validation = True

    def on_test_end(self, logs=None):
        # validation finished — resume sampling for subsequent training batches
        self._in_validation = False

    def on_epoch_end(self, epoch, logs=None):
        # commit epoch training time (excludes validation time)
        self.epoch_train_times.append(self._epoch_batch_time)

    def on_train_end(self, logs=None):
        # finalize
        self.train_time_s = float(self.total_train_time)
        if use_psutil and self.mem_samples:
            self.mem_min = min(self.mem_samples)
            self.mem_max = max(self.mem_samples)
        else:
            self.mem_min = self.mem_max = None

# --------- prepare outputs dir ----------
os.makedirs("./TCN", exist_ok=True)

# --------- compute forward FLOPs (batch=1) ----------
flops_forward = compute_model_flops(model)
if flops_forward is not None:
    print(f"Estimated forward FLOPs (batch=1): {flops_forward:,}")
else:
    print("Could not compute forward FLOPs (profiler unavailable).")

# --------- fine-tune with callback (validation still runs, but memory sampling excludes it) ----------
memcb = SimpleMemTimeCallback()

# use single variable for batch_size (keep consistent)
batch_size = 128

# record memory before training (RSS) if available
mem_before = _ps.memory_info().rss if use_psutil else None

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=1,
    batch_size=batch_size,
    verbose=1,
    callbacks=[memcb]   # add other callbacks (ModelCheckpoint/EarlyStopping) if you want
)

# Save best/final outside if you used ModelCheckpoint earlier; otherwise save now
try:
    model.save("./TCN/final_pruned_finetuned_model.h5")
except Exception:
    pass


# --------- training measurements (training-only memory samples) ----------
train_time_s = getattr(memcb, "train_time_s", None)
num_batches = getattr(memcb, "batches", None)
mem_min = getattr(memcb, "mem_min", None)
mem_max = getattr(memcb, "mem_max", None)
mem_after = _ps.memory_info().rss if use_psutil else None

# Estimate training FLOPs using 2x forward (approx) per step
train_flops_total = None
train_flops_per_sec = None
if flops_forward is not None and num_batches and train_time_s and train_time_s > 0:
    fwd_per_batch = flops_forward * batch_size
    train_per_batch = 2 * fwd_per_batch
    train_flops_total = int(train_per_batch * num_batches)
    train_flops_per_sec = float(train_flops_total) / train_time_s

# --------- inference & metrics on validation (measured separately, after training) ----------
t0 = time.time()
y_pred_probs = model.predict(x_val, batch_size=batch_size, verbose=0)
t1 = time.time()
inf_time_total = t1 - t0
n_samples = int(x_val.shape[0]) if hasattr(x_val, "shape") else len(x_val)
inf_per_sample_ms = (inf_time_total / n_samples) * 1000.0 if n_samples > 0 else None

# map preds to labels
if y_pred_probs.ndim > 1 and y_pred_probs.shape[1] > 1:
    y_pred = np.argmax(y_pred_probs, axis=1)
else:
    y_pred = (y_pred_probs.ravel() >= 0.5).astype(int)
y_true = np.array(y_val).reshape(-1)

# accuracy & F1
if use_sklearn:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average='macro'))
else:
    acc = float(np.mean(y_true == y_pred))
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for lab in labels:
        tp = np.sum((y_pred==lab) & (y_true==lab))
        fp = np.sum((y_pred==lab) & (y_true!=lab))
        fn = np.sum((y_pred!=lab) & (y_true==lab))
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1s.append((2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0)
    f1 = float(np.mean(f1s)) if f1s else 0.0

# inference FLOPs/sec estimate
inf_flops_total = None
inf_flops_per_sec = None
if flops_forward is not None and inf_time_total > 0:
    inf_flops_total = int(flops_forward * n_samples)
    inf_flops_per_sec = float(inf_flops_total) / inf_time_total

# --------- print summary ----------
print("\n=== Summary ===")
print(f"Fine-tune time (s)        : {train_time_s:.2f}" if train_time_s is not None else "Fine-tune time: N/A")
print(f"Train batches processed   : {num_batches}")
if use_psutil:
    print(f"Memory before (RSS MB)    : {mem_before/(1024**2):.2f}" if mem_before is not None else "Memory before: N/A")
    print(f"Memory after  (RSS MB)    : {mem_after/(1024**2):.2f}" if mem_after is not None else "Memory after: N/A")
    if mem_min is not None:
        print(f"Observed mem min (RSS MB) : {mem_min/(1024**2):.2f}")
        print(f"Observed mem max (RSS MB) : {mem_max/(1024**2):.2f}")
if train_flops_total is not None:
    print(f"Estimated train FLOPs total: {train_flops_total:,}")
    print(f"Estimated train FLOPs/s    : {train_flops_per_sec:,.0f} FLOPs/s")
print(f"Validation inference time (s) : {inf_time_total:.3f}")
print(f"Per-sample inf time (ms)      : {inf_per_sample_ms:.3f}" if inf_per_sample_ms is not None else "Per-sample inf time: N/A")
if inf_flops_total is not None:
    print(f"Estimated inf FLOPs total     : {inf_flops_total:,}")
    print(f"Estimated inf FLOPs/s         : {inf_flops_per_sec:,.0f} FLOPs/s")
print(f"Val samples: {n_samples}")
print(f"Val accuracy: {acc:.4f}")
print(f"Val F1 (macro): {f1:.4f}")

# per-epoch timing details
epoch_times = getattr(memcb, "epoch_train_times", None)
if epoch_times is not None:
    for i, t in enumerate(epoch_times):
        print(f"Epoch {i:02d} training time (s): {t:.3f}")

# --------- save metrics ----------
metrics = {
    "train_time_s": float(train_time_s) if train_time_s is not None else None,
    "epoch_train_times_s": list(getattr(memcb, "epoch_train_times", [])),
    "num_batches": int(num_batches) if num_batches is not None else None,
    "mem_before_mb": float(mem_before/(1024**2)) if use_psutil and mem_before is not None else None,
    "mem_after_mb": float(mem_after/(1024**2)) if use_psutil and mem_after is not None else None,
    "mem_min_mb": float(mem_min/(1024**2)) if mem_min is not None else None,
    "mem_max_mb": float(mem_max/(1024**2)) if mem_max is not None else None,
    "inf_time_s": float(inf_time_total),
    "inf_per_sample_ms": float(inf_per_sample_ms) if inf_per_sample_ms is not None else None,
    "val_accuracy": float(acc),
    "val_f1_macro": float(f1),
    "val_samples": int(n_samples),
    "flops_forward_per_sample": int(flops_forward) if flops_forward is not None else None,
    "train_flops_total": int(train_flops_total) if train_flops_total is not None else None,
    "train_flops_per_sec": float(train_flops_per_sec) if train_flops_per_sec is not None else None,
    "inf_flops_total": int(inf_flops_total) if inf_flops_total is not None else None,
    "inf_flops_per_sec": float(inf_flops_per_sec) if inf_flops_per_sec is not None else None,
}
with open("./TCN/fine_tune_metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=2)

print("\nMetrics saved to ./TCN/fine_tune_metrics.json")

