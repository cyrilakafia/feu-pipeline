import unittest
import torch
import os
from feu.prep import prep
from feu.inf import run_inference

data = torch.rand(20, 1000)


# Run the inference process
title = 'demo_run'
device = 'cpu'
num_iterations = 5
conc = 1
max_clusters = 20
stimulus_timepoint = 1
num_trials = 100


best_assigns, best_params = run_inference(
                                        data,
                                        title=title, 
                                        device=device,
                                        iterations=num_iterations,
                                        concentration=conc, 
                                        max_clusters=max_clusters,
                                        num_trials=num_trials,
                                        t_stimulus=stimulus_timepoint,
                                        seed=None)