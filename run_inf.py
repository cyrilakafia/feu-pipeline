from feu.inf import run_inference
from feu.prep import prep
import os
from feu.visualize_heatmap import viz_heatmap

original_data = 'test_data/array_30_200.pkl'

# Preprocess the data
preprocessed_data = 'outputs/processed_data.p'

if not os.path.exists('outputs'):
    os.makedirs('outputs')

prep(original_data, preprocessed_data)

# Run the inference process
title = 'demo_run'
device = 'cpu'
num_iterations = 5
conc = 1
max_clusters = 20
stimulus_timepoint = 1
num_trials = 100


best_assigns, best_params = run_inference(
                                        preprocessed_data,
                                        title=title, 
                                        device=device,
                                        iterations=num_iterations,
                                        concentration=conc, 
                                        max_clusters=max_clusters,
                                        num_trials=num_trials,
                                        t_stimulus=stimulus_timepoint,
                                        seed=None)

print(type(best_assigns))
print(type(best_params))

# <class 'pandas.core.frame.DataFrame'>
# <class 'pandas.core.frame.DataFrame'>

# Assuming you have already run the inference process, you can find best assignment and paramsn and visualize the results using the following code:

#best_assigns, best_params = viz_heatmap(title, num_iterations, 'outputs/simdemo_run_assigns.csv', 'outputs/simdemo_run_params.tsv', max_clusters=max_clusters)

