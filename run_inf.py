from feu.inf import run_inference
from feu.prep import prep
import os
from feu.visualize_heatmap import viz_heatmap, make_raster_fig
import torch
import pandas as pd
import numpy as np

# original_data = 'test_data/array_30_200.pkl'

# # Preprocess the data
# preprocessed_data = 'outputs/processed_data.p'

# if not os.path.exists('outputs'):
#     os.makedirs('outputs')

# prep(original_data, preprocessed_data)

# Run the inference process
title = 'demo_run'
device = 'cpu'
num_iterations = 5
conc = 1
max_clusters = 20
stimulus_timepoint = 1
num_trials = 100


# best_assigns, best_params = run_inference(
#                                         preprocessed_data,
#                                         title=title, 
#                                         device=device,
#                                         iterations=num_iterations,
#                                         concentration=conc, 
#                                         max_clusters=max_clusters,
#                                         num_trials=num_trials,
#                                         t_stimulus=stimulus_timepoint,
#                                         seed=None)

# print(type(best_assigns))
# print(type(best_params))

# <class 'pandas.core.frame.DataFrame'>
# <class 'pandas.core.frame.DataFrame'>

# Assuming you have already run the inference process, you can find best assignment and paramsn and visualize the results using the following code:

best_assigns, best_params = viz_heatmap('human_ephys_test4', 2641, 'outputs/simhuman_ephys_test4_assigns.csv', 'outputs/simhuman_ephys_test4_params.tsv', max_clusters=20)


# Making rasters
data = torch.load('outputs/human_ephys_trials.p', map_location=device)
make_raster_fig(data, t_stimulus=500, best_assigns=best_assigns, title='human_ephys_test4')