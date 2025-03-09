
# %% Script Parameters
import utils
import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
url = 'https://uni-bonn.sciebo.de/s/oTfGigwXQ4g0raW'
filename = 'data.nc'
from utils import compute_psths





#from netCDF4 import Dataset
# %% Download Data
# Exercise (Example): Make a download_data(url, filename) function:




utils.download_data(url=url, filename=filename)

# %% Load Data
# Exercise: Make a `load_data(filename)` function, returning the `dset` variable.




dset=utils.load_data(filename)
print(type(dset))





# %% Extract Experiment-Level Data
# Exercise: Make an `extract_trials(filename)` function, returning the `trials` variable.





trials = utils.extract_trials(dset)

# %% Extract Spike-Time Data
# Exercise: Make an `extract_spikes(filename)` function, returning the `spikes` variable.




spikes = utils.extract_spikes(dset)


# %% Extract Cell-Level Data
# Exercise: Make an `extract_cells(filename)` function, returning the `cells` variable.



cells = utils.extract_cells(dset)

# %% Merge and Compress Extracted Data
# Exercise: Make a `merge_data(trials, cells, spikes)` function, returning the `merged` variable.







merged= utils.merge_data(trials, cells, spikes)


# %% Calculate Time Bins for PSTH
# Exercise: Make a `compute_time_bins(time, bin_interval)` function, returning the `time_bins` variable.



time_bins = utils.compute_time_bins(merged['time'], bin_interval=0.05)

# %% filter out stimuli with contrast on the right.
# No function needed here for this exercise.

filtered = merged[merged['contrast_right'] == 0]
print(f"Filtered out {len(merged) - len(filtered)} ({len(filtered) / len(merged):.2%}) of spikes in dataset.")
filtered

# %% Make PSTHs
# Exercise: Make a `compute_psths(data, time_bins)` function here, returning the `psth` variable.
#data=filtered
import inspect
print(inspect.signature(compute_psths))
psth = utils.compute_psths(filtered, time_bins, 0.05)
# %% Plot PSTHs
# Make a `plot_psths(psth)` function here, returning the `g` variable.






utils.plot_psths(psth)

# %%
