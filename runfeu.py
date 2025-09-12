from feu.inf import run_inference
from feu.prep import prep
import os
from feu.postinf import find_best_clust_and_params, make_raster_fig
import torch
import pandas as pd
import numpy as np

job_summary = "Test FEU RUN"

# Define the parameters
title = '09-12-2025-test'
device = 'cpu'
num_iterations = 5
conc = 1
max_clusters = 20
stimulus_timepoint = 1
num_trials = 10
additional_info = 'Test run on 09-12-2025'
# num_trials = np.arange(100, 130)

# simulate a numpy array with shape 10 x 30 x 200
spike_data = np.random.randint(0, 2, size=(10, 30, 200))
np.save('test_data/array_30_200.npy', spike_data.astype(np.float32))
print(f"Simulated data shape: {spike_data.shape}")

original_data = 'test_data/array_30_200.npy'

# sum over trials to get spike counts per neuron
spike_data_summed = np.sum(spike_data, axis=0)
print(f"Summed data shape: {spike_data_summed.shape}")
print(spike_data_summed)
# save the summed data as a numpy file
np.save('test_data/array_30_200_2d.npy', spike_data_summed.astype(np.float32))

original_data_2d = 'test_data/array_30_200_2d.npy'

# Preprocess the data
preprocessed_data = f'{title}/processed_data.p'

if not os.path.exists(title):
    os.makedirs(title)

prep(original_data_2d, preprocessed_data)


best_assigns, best_params = run_inference(
                                        preprocessed_data,
                                        title=title, 
                                        device=device,
                                        iterations=num_iterations,
                                        concentration=conc, 
                                        max_clusters=max_clusters,
                                        num_trials=num_trials,
                                        t_stimulus=stimulus_timepoint,
                                        seed=42,
                                        figures=True,
                                        additional_info=job_summary
                                        )

# Assuming you have already run the inference process, you can find best assignment and paramsn and visualize the results using the following code:
# best_assigns, best_params = viz_heatmap('human_ephys_test4', 2641, 'outputs/simhuman_ephys_test4_assigns.csv', 'outputs/simhuman_ephys_test4_params.tsv', max_clusters=20)

# Making rasters
data = np.load(original_data, allow_pickle=True)
print(data.shape)

# Reshape the rasters to neurons x trials x timepoints
data_reshaped = np.transpose(data, (1, 0, 2))

data_df = pd.DataFrame({'data': list(data_reshaped)})  # Convert each row of the array to a separate row in DataFrame
print(data_df.values[0][0].shape)
data_df.to_csv(f'{title}/data_for_raster.csv', index=False)
make_raster_fig(data_df, t_stimulus=stimulus_timepoint, best_assigns=best_assigns, title=title)