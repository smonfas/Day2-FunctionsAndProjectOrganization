import utils

url = 'https://uni-bonn.sciebo.de/s/oTfGigwXQ4g0raW'
filename = 'data.nc'


utils.download_data(url=url, filename=filename)

trials = utils.extract_trials(filename=filename)
trials

spikes = utils.extract_spikes(filename=filename)

cells = utils.extract_cells(filename)

merged = utils.merge_data(trials, spikes, cells)

bin_interval = 0.05
time = merged['time']


time_bins = utils.compute_time_bins(time, bin_interval)

filtered = merged[merged['contrast_right'] == 0]
print(f"Filtered out {len(merged) - len(filtered)} ({len(filtered) / len(merged):.2%}) of spikes in dataset.")
filtered

psth = utils.compute_psths(bin_interval, time_bins, filtered)
psth

g = utils.plot_psths(psth=psth)
g.savefig('PSTHs.png')

