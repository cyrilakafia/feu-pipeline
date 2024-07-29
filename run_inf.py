from feu.inf import run_inference
from feu.prep import prep
import os
from feu.visualize_heatmap import viz_heatmap, make_raster_fig
import torch
import pandas as pd
import numpy as np

# Define the parameters
title = 'demo'
device = 'cpu'
num_iterations = 5
conc = 1
max_clusters = 20
stimulus_timepoint = 0
num_trials = 100
# num_trials = np.arange(100, 130)

original_data = 'test_data/array_30_200.pkl'

# Preprocess the data
preprocessed_data = f'{title}/processed_data.p'

if not os.path.exists(title):
    os.makedirs(title)

prep(original_data, preprocessed_data)

print(type(num_trials))

best_assigns, best_params = run_inference(
                                        preprocessed_data,
                                        title=title, 
                                        device=device,
                                        iterations=num_iterations,
                                        concentration=conc, 
                                        max_clusters=max_clusters,
                                        num_trials=num_trials,
                                        t_stimulus=stimulus_timepoint,
                                        seed=None,
                                        figures=True)

# print(type(best_assigns))
# print(type(best_params))

# <class 'pandas.core.frame.DataFrame'>
# <class 'pandas.core.frame.DataFrame'>

# Assuming you have already run the inference process, you can find best assignment and paramsn and visualize the results using the following code:

# best_assigns, best_params = viz_heatmap('human_ephys_test4', 2641, 'outputs/simhuman_ephys_test4_assigns.csv', 'outputs/simhuman_ephys_test4_params.tsv', max_clusters=20)


# Making rasters
# data = torch.load('outputs/human_ephys_trials.p', map_location=device)
# make_raster_fig(data, t_stimulus=500, best_assigns=best_assigns, title='human_ephys_test4')