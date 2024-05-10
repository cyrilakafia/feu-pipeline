from inf import run_inference
from dpnssm.prep import prep
import os


original_data = 'test_data/array_30_200.pkl'

# Preprocess the data
preprocessed_data = 'outputs/processed_data.p'

if not os.path.exists('outputs'):
    os.makedirs('outputs')

prep(original_data, preprocessed_data)

# Run the inference process
best_assigns, best_params = run_inference(
                                        preprocessed_data,
                                        title='demo', 
                                        device='cpu', 
                                        iterations=5, 
                                        concentration=1, 
                                        max_clusters=20, 
                                        timepoint=0, 
                                        seed=None
                                        )

# print(type(best_assigns))
# print(type(best_params))

# <class 'pandas.core.frame.DataFrame'>
# <class 'pandas.core.frame.DataFrame'>ex

