
# %% Script Parameters

url = 'https://uni-bonn.sciebo.de/s/oTfGigwXQ4g0raW'
filename = 'data.nc'

# %% Download Data
# Exercise (Example): Make a download_data(url, filename) function:

def download_data(url, filename):
    from pathlib import Path
    import owncloud

    client = owncloud.Client.from_public_link(url)
    client.get_file('/', filename)

    if Path(filename).exists():
        print('Download Succeeded.')

    return None


download_data(url=url, filename=filename)


# %% Extract Experiment-Level Data
# Exercise: Make an `extract_trials(filename)` function, returning the `trials` variable.

import xarray as xr

def extract_trials(filename):
    dset = xr.load_dataset(filename)
    trials = dset[['contrast_left', 'contrast_right', 'stim_onset']].to_dataframe()
    return trials


trials = extract_trials(filename=filename)
trials

# %% Extract Spike-Time Data
# Exercise: Make an `extract_spikes(filename)` function, returning the `spikes` variable.

import xarray as xr

def extract_spikes(filename):
    dset = xr.load_dataset(filename)
    spikes = dset[['spike_trial', 'spike_cell', 'spike_time']].to_dataframe()
    return spikes

spikes = extract_spikes(filename=filename)
spikes

# %% Extract Cell-Level Data
# Exercise: Make an `extract_cells(filename)` function, returning the `cells` variable.

import xarray as xr

def extract_cells(filename):
    dset = xr.load_dataset(filename)
    cells = dset['brain_groups'].to_dataframe()
    cells
    return cells

cells = extract_cells(filename)
cells

# %% Merge and Compress Extracted Data
# Exercise: Make a `merge_data(trials, cells, spikes)` function, returning the `merged` variable.

import pandas as pd
def merge_data(trials, spikes, cells):
    merged = pd.merge(left=cells, left_index=True, right=spikes, right_on='spike_cell')
    merged = pd.merge(left=trials, right=merged, left_index=True, right_on='spike_trial').reset_index(drop=True)
    merged.columns
    merged = (merged
    .rename(columns=dict(
        brain_groups="brain_area",
        spike_trial="trial_id",
        spike_cell="cell_id",
        spike_time="time"
    ))
    [[
        'trial_id',
        'contrast_left',
        'contrast_right',
        'stim_onset',
        'cell_id',
        'brain_area',
        'time'
    ]]
    .astype(dict(   
        brain_area = 'category',
    ))
    # 
)
    merged.info()
    return merged

merged = merge_data(trials, spikes, cells)
merged


# %% Calculate Time Bins for PSTH
# Exercise: Make a `compute_time_bins(time, bin_interval)` function, returning the `time_bins` variable.

import numpy as np
bin_interval = 0.05
time = merged['time']

def compute_time_bins(time, bin_interval):
    time = np.round(time, decimals=6)  # Round time to the nearest microsecond, to reduce floating point errors.
    time_bins = np.floor(time /bin_interval) * bin_interval  # Round down to the nearest time bin start
    return time_bins

time_bins = compute_time_bins(time, bin_interval)
time_bins


# %% filter out stimuli with contrast on the right.
# No function needed here for this exercise.

filtered = merged[merged['contrast_right'] == 0]
print(f"Filtered out {len(merged) - len(filtered)} ({len(filtered) / len(merged):.2%}) of spikes in dataset.")
filtered

# %% Make PSTHs
# Exercise: Make a `compute_psths(data, time_bins)` function here, returning the `psth` variable.

def compute_psths(bin_interval, time_bins, filtered):
    psth = (
    filtered
    .groupby([time_bins, 'trial_id', 'contrast_left', 'cell_id', 'brain_area'], observed=True, )
    .size()
    .rename('spike_count')
    .reset_index()
)
    psth
    psth = (
    psth
    .groupby(['time', 'contrast_left', 'brain_area'], observed=True)
    .spike_count
    .mean()
    .rename('avg_spike_count')
    .reset_index()
)
    psth
    psth['avg_spike_rate'] = psth['avg_spike_count'] * bin_interval
    psth
    return psth

psth = compute_psths(bin_interval, time_bins, filtered)
psth


# %% Plot PSTHs
# Make a `plot_psths(psth)` function here, returning the `g` variable.
import seaborn as sns

def plot_psths(psth):
    g = sns.FacetGrid(data=psth, col='brain_area', col_wrap=2)
    g.map_dataframe(sns.lineplot, x='time', y='avg_spike_count', hue='contrast_left')
    g.add_legend()

g = plot_psths(psth=psth)
g.savefig('PSTHs.png')


# %%
