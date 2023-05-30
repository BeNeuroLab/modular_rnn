# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3.9 (uncertainty)
#     language: python
#     name: uncertainty
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# when running on CPU, I found that performance is pretty much the same as with many cores
torch.set_num_threads(1)

# %% [markdown]
# # Create task and RNN

# %%
from modular_rnn.connections import ConnectionConfig
from modular_rnn.models import MultiRegionRNN
from modular_rnn.loss_functions import MSEOnlyLoss

# %% [markdown]
# ## Set parameters

# %%
# time constant of each neuron's the dynamics
tau = 100

# timestep of the simulation
dt = 5

# need this
alpha = dt / tau

# noise in the dynamics
noise = 0.05

# activation function of the neurons
nonlin_fn = F.relu

# %%
# length of each trial
L = 1200

# number of trials in a batch
batch_size = 64

# special loss for the uncertainty task
loss_fn = MSEOnlyLoss(['hand'])

# %% [markdown]
# ## Create task

# %%
from modular_rnn.tasks import CenterOutTaskWithReachProfiles

task = CenterOutTaskWithReachProfiles(dt, tau, L, batch_size, n_targets = 8)

# %% [markdown]
# ## Create RNN

# %% tags=[]
# dictionary defining the modules in the RNN
# here we'll have a single region called motor_cortex
regions_config_dict = {
    'PMd' : {
        'n_neurons' : 50,
        'alpha' : alpha,
        'p_rec': 1.,
        'rec_rank' : 1,
        'dynamics_noise' : noise,
    },
    'M1' : {
        'n_neurons' : 50,
        'alpha' : alpha,
        'p_rec': 1.,
        'rec_rank' : 1,
        'dynamics_noise' : noise,
    }
}

# name and dimensionality of the outputs we want the RNN to produce
output_dims = task.output_dims

# name and dimensionality of the inputs we want the RNN to receive
input_dims = task.input_dims

# %%
rnn = MultiRegionRNN(
         input_dims,
         output_dims,
         alpha,
         nonlin_fn,
         regions_config_dict, 
         connection_configs = [
             ConnectionConfig('PMd', 'M1')
         ],
         input_configs = [
             ConnectionConfig('target', 'PMd'),
             ConnectionConfig('go_cue', 'PMd'),
         ],
         output_configs = [
             ConnectionConfig('M1', 'hand'),
         ],
         feedback_configs = []
)

# %% [markdown]
# # Train

# %% tags=[]
from modular_rnn.training import train

losses = train(rnn, task, 300, loss_fn)
plt.plot(losses[10:]);

# %% [markdown]
# # Test the model's behavior on some test trials

# %% [markdown]
# Run a few batches of test trials

# %%
from modular_rnn.testing import run_test_batches

test_df = run_test_batches(10, rnn, task)

# %% [markdown]
# Produced "hand" trajectories

# %%
from pysubspaces.plotting import get_color_cycle

# %%
fig, ax = plt.subplots()

for (tid, target_df) in test_df.groupby('target_id'):
    for arr in target_df.hand_model_output.values[:10]:
        ax.scatter(*arr.T, alpha = 0.1, color = get_color_cycle()[tid])
        
ax.set_title('model output')
ax.set_xlabel('x')
ax.set_ylabel('y')

# %% [markdown]
# Hand velocities

# %%
from pyaldata import *

test_df = add_gradient(test_df, 'hand_model_output', 'model_vel')
test_df = add_norm(test_df, 'model_vel')

test_df = add_gradient(test_df, 'hand_target_output', 'target_vel')
test_df = add_norm(test_df, 'target_vel')

# %%
fig, ax = plt.subplots(ncols = 2, sharey = True)

for arr in restrict_to_interval(test_df, 'idx_go_cue', rel_start = -10, rel_end = 40).model_vel_norm.values:
    ax[0].plot(arr)
for arr in restrict_to_interval(test_df, 'idx_go_cue', rel_start = -10, rel_end = 40).target_vel_norm.values:
    ax[1].plot(arr)

# %%
fig, ax = plt.subplots(ncols = 2, sharey = True)

for arr in restrict_to_interval(test_df, 'idx_target_on', rel_start = -10, rel_end = 40).model_vel_norm.values:
    ax[0].plot(arr)
for arr in restrict_to_interval(test_df, 'idx_target_on', rel_start = -10, rel_end = 40).target_vel_norm.values:
    ax[1].plot(arr)

# %% [markdown]
# Latent trajectories

# %%
import plotly.graph_objects as go
import plotly.express as px

# %%
from sklearn.decomposition import PCA

test_df = dim_reduce(test_df, PCA(30), 'PMd_rates', 'PMd_proj')
test_df = dim_reduce(test_df, PCA(30), 'M1_rates', 'M1_proj')

# %%
fig = go.Figure()

for (target_id, target_df) in test_df.groupby('target_id'):
    x = concat_trials(target_df.iloc[:10, :], 'PMd_proj')
    fig.add_scatter3d(
        x = x[:, 0],
        y = x[:, 1],
        z = x[:, 2],
    )

fig

# %% [markdown]
# Decoding target ID

# %%
from pysubspaces import get_classif_cv_scores_through_time
from sklearn.linear_model import RidgeClassifier

# %%
prep_td = restrict_to_interval(test_df, 'idx_target_on', rel_start = 0, rel_end = 70)

cv_scores = get_classif_cv_scores_through_time(prep_td, RidgeClassifier, 'PMd_proj', 'target_id')
plt.plot(cv_scores.mean(axis = 1))
plt.xlabel('timestep')
plt.ylabel('accuracy')

# %%
