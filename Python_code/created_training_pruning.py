#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf
from tcn import TCN
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
 
from tensorflow.keras.models import load_model


# #### Change the dataset paths

# In[ ]:


TRAIN_DIR = './preprocessed_/S1'
TEST_DIR = './preprocessed_/S2'

label_map = {char: idx for idx, char in enumerate("ACDFGH")}

def load_data(directory):
    data = []
    labels = []
    for file in os.listdir(directory):
        
        if file.endswith('.npz') and file[11] in 'ACDFGH' and 'a' == file[9] and '0' ==file[13]:
            print(file)
            # print(file)
            filepath = os.path.join(directory, file)
            z = np.load(filepath, allow_pickle=True)
            csi=z['data']
            # print(csi)
            # print(csi.shape)
            label = label_map[file[11]]
            # print(label)
            data.append(csi)
            # print(data[0].shape)
            labels.append(label)
    return np.array(data), np.array(labels)

x_train, y_train = load_data(TRAIN_DIR)

import numpy as np
from sklearn.model_selection import train_test_split

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
        x_class, y_class, test_size=0.2, random_state=42, shuffle=True
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


# In[ ]:


def csi_network_inc_res(input_sh, output_sh):
    
    nb_filters = 32
    L2 = 1e-4
    x_input = Input(shape=input_sh)

    tcn_block = TCN(
        nb_filters=nb_filters,
        nb_stacks=1,
        kernel_size=10,
        dilations=(1, 2, 4, 8, 16),
        use_layer_norm=True,
        return_sequences=True,  
        # use_skip_connections=True,
        kernel_initializer= 'glorot_uniform',
        name="tcn_block",
    )

    x = tcn_block(x_input)  
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(L2),name="conv1d")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_sh, activation="sigmoid", name="dense")(x)

    model = Model(inputs=x_input, outputs=x, name="csi_model_tcn_conv1d")
    return model
model = csi_network_inc_res((x_train.shape[1], x_train.shape[2]), num_classes)


# In[ ]:


## training the model
lr = 1e-4
opt = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=opt,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
os.makedirs("./TCN", exist_ok=True)
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint("./TCN/best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=25,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)


# In[16]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D

def _conv1d_sublayers(layer):
    # simple collector of Conv1D sublayers (works for keras-tcn typical structure)
    convs = []
    for sub in getattr(layer, "submodules", []):
        if isinstance(sub, Conv1D):
            convs.append(sub)
    if not convs:
        for sub in getattr(layer, "layers", []):
            if isinstance(sub, Conv1D):
                convs.append(sub)
            else:
                for s2 in getattr(sub, "layers", []):
                    if isinstance(s2, Conv1D):
                        convs.append(s2)

    uniq = []
    seen = set()
    for c in convs:
        if id(c) not in seen:
            uniq.append(c); seen.add(id(c))
    return uniq

def prune_tcn_block_structured_compact(model, tcn_layer_name="tcn_block", prune_fraction=0.3, verbose=False):
    """
    Compact structured pruning of internal Conv1D filters inside the TCN block.
    Returns a new Keras Model with a TCN that has fewer filters. Fine-tune after call.
    """
    # find tcn layer
    tcn_old = model.get_layer(tcn_layer_name)
    cfg = None
    try:
        cfg = tcn_old.get_config()
    except Exception:
        cfg = {}
    # infer old_nb_filters
    old_nb_filters = None
    for key in ("nb_filters", "filters"):
        if cfg and key in cfg:
            old_nb_filters = int(cfg[key]); break
    if old_nb_filters is None:
        old_nb_filters = int(getattr(tcn_old, "nb_filters", getattr(tcn_old, "filters", None) or 0))
    if old_nb_filters <= 0:
        raise RuntimeError("Cannot determine TCN nb_filters.")

    new_nb_filters = max(1, int(round(old_nb_filters * (1.0 - prune_fraction))))
    if verbose:
        print(f"TCN: {old_nb_filters} -> {new_nb_filters} filters (prune_fraction={prune_fraction})")

    # construct new TCN instance reusing same class
    TCNClass = tcn_old.__class__
   
    kwargs = {}
    if cfg:
        kwargs.update(cfg)
        kwargs.pop("name", None)
        if "nb_filters" in kwargs:
            kwargs["nb_filters"] = new_nb_filters
        elif "filters" in kwargs:
            kwargs["filters"] = new_nb_filters
    try:
        tcn_new = TCNClass(**kwargs)
    except Exception:
    
        try:
            tcn_new = TCNClass(nb_filters=new_nb_filters,
                               kernel_size=cfg.get("kernel_size", 3) if cfg else 3,
                               nb_stacks=cfg.get("nb_stacks", 1) if cfg else 1,
                               dilations=cfg.get("dilations", (1,2,4,8,16)) if cfg else (1,2,4,8,16),
                               return_sequences=cfg.get("return_sequences", True) if cfg else True,
                               use_layer_norm=cfg.get("use_layer_norm", False) if cfg else False)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate new TCN: {e}")

   
    in_shape = model.input_shape  # (None, time, channels)
    if in_shape is None or len(in_shape) < 3:
        raise RuntimeError("Unexpected model.input_shape.")
    time_steps, in_ch = in_shape[1], in_shape[2]
    dummy = tf.random.normal((1, time_steps, in_ch))
    try:
        _ = tcn_old(dummy)
    except Exception:
        pass
    try:
        _ = tcn_new(dummy)
    except Exception:
        pass

    old_convs = _conv1d_sublayers(tcn_old)
    new_convs = _conv1d_sublayers(tcn_new)
    if verbose:
        print(f"Found {len(old_convs)} old convs, {len(new_convs)} new convs")


    pair_count = min(len(old_convs), len(new_convs))
    old_convs = old_convs[:pair_count]; new_convs = new_convs[:pair_count]
    if pair_count == 0:
        raise RuntimeError("No Conv1D sublayers found in TCN block.")

   
    agg = np.zeros(old_nb_filters, dtype=float)
    valid = 0
    for c in old_convs:
        w = c.get_weights()
        if not w: continue
        k = w[0]
        if k.ndim != 3: continue
        if k.shape[-1] != old_nb_filters: continue
        agg += np.sum(np.abs(k), axis=(0,1))
        valid += 1
    if valid == 0:
   
        for c in old_convs:
            w = c.get_weights()
            if not w: continue
            k = w[0]
            if k.ndim == 3:
                out = k.shape[-1]
                agg[:out] += np.sum(np.abs(k), axis=(0,1))
                valid += 1
    if valid == 0:
        raise RuntimeError("Cannot compute importance; no matching conv kernel shapes.")

    keep_idx = np.sort(np.argsort(agg)[-new_nb_filters:])
    if verbose:
        print("keep_idx:", keep_idx.tolist())


    for cold, cnew in zip(old_convs, new_convs):
        old_w = cold.get_weights()
        if not old_w: 
            continue
        old_k = old_w[0]  # (k, in_ch_local, out_ch_local)
        old_b = old_w[1] if len(old_w) > 1 else None
        in_ch_local = old_k.shape[1]; out_ch_local = old_k.shape[2]
   
        if in_ch_local == old_nb_filters:
            in_sel = keep_idx[keep_idx < in_ch_local]
        else:
            in_sel = np.arange(in_ch_local)
        out_sel = keep_idx[keep_idx < out_ch_local]
        if out_sel.size == 0:
            # nothing to copy
            continue
        pruned_k = old_k[:, in_sel.astype(int), :][:, :, out_sel.astype(int)]
        pruned_b = old_b[out_sel.astype(int)] if old_b is not None else None

        try:
    
            try:
                _ = cnew(np.zeros((1, time_steps, in_ch)))
            except Exception:
                pass
            # direct set
            new_weights = [pruned_k]
            if pruned_b is not None:
                new_weights.append(pruned_b)
            cnew.set_weights(new_weights)
        except Exception:
            # skip silently (new conv stays init)
            if verbose:
                print(f"Could not set weights for {cold.name} -> {cnew.name}, skipping.")


    inp = Input(shape=(time_steps, in_ch), name="pruned_input")
    x = inp
    replaced = False
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if layer.name == tcn_layer_name and not replaced:
            x = tcn_new(x)
            replaced = True
            continue
        if not replaced:
            # reuse layer instance (pre-TCN)
            try:
                x = layer(x)
            except Exception:
                # fallback clone
                cfg = layer.get_config(); cls = layer.__class__
                nl = cls.from_config(cfg)
                x = nl(x)
                try:
                    w = layer.get_weights()
                    if w: nl.set_weights(w)
                except Exception:
                    pass
        else:
            # clone layers after TCN
            cfg = layer.get_config(); cls = layer.__class__
            nl = cls.from_config(cfg)
            x = nl(x)
            try:
                w = layer.get_weights()
                if w:
                    try: nl.set_weights(w)
                    except Exception: 
                        if verbose: print(f"post-TCN weights mismatch for {layer.name}, skipped weights copy.")
            except Exception:
                pass

    pruned_model = Model(inputs=inp, outputs=x, name=model.name + "_tcn_pruned")
    if verbose:
        pruned_model.summary()
    return pruned_model


# In[17]:


## load the train model
model = load_model("./TCN/best_model.h5", custom_objects={'TCN': TCN})


# 

# In[ ]:


# Pruned the trained model 

pruned = prune_tcn_block_structured_compact(model, tcn_layer_name="tcn_block", prune_fraction=0.5, verbose=True)
pruned.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


# 3. Fine-tune the pruned model (using the same data, fewer epochs)
history = pruned.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=28,          # small number, just to recover performance
    batch_size=64,
    verbose=1
)
pruned.save("./TCN/final_pruned_model.h5")


# In[ ]:


print("Original params:", model.count_params())
print("Pruned params:  ", pruned.count_params())
print("Reduction:      ", 100*(model.count_params()-pruned.count_params())/model.count_params(), "%")
original_path = "./TCN/best_model.h5"
pruned_path = "./TCN/final_pruned_model.h5"   
orig_size = os.path.getsize(original_path) / (1024 * 1024)  # MB
pruned_size = os.path.getsize(pruned_path) / (1024 * 1024)  # MB
size_reduction = 100 * (orig_size - pruned_size) / orig_size

print(f"\nOriginal model size   : {orig_size:.3f} MB")
print(f"Pruned model size     : {pruned_size:.3f} MB")
print(f"Size reduction        : {size_reduction:.2f}%")

