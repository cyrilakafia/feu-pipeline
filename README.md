# Functional Encoding Units (FEU)

## Overview

This repository contains the implementation for extracting Functional Encoding Units (FEUs) using PyTorch. The code allows users to perform inference to identify optimal cluster assignments and parameters based on provided data.

Disclaimer: This algorithm and library is still under construction

## Installation

Start by cloning this repository.

```bash
git clone https://github.com/cyrilakafia/feu-pipeline.git
```

Ensure you are working in an environ with PyTorch, numpy, matplotlib and pandas install. We reccommend you create a new environment with conda and install these packages

```bash
conda create -n 'feu' python=3.10
conda activate feu
cd feu-pipeline
pip install -r requirements.txt
```
## Data 

The data processed by the pipeline is `two-dimensional time series data`, where the first dimension represents the variables to be clustered, and the second dimension represents the corresponding time series data/timepoints.

For example, consider electrophysiological data recorded from 25 neurons over a duration of 1000 seconds. Assuming that one measurement is recorded every second, the data will have a shape of (25, 1000). Each row corresponds to a different neuron, and each column represents sequential time points.

This structured data should be saved in a `PyTorch file with the .p` extension. Conversion from other data formats to the PyTorch format is supported within our pipeline, ensuring compatibility and ease of integration for various data sources. This structure facilitates the application of time series analysis or clustering algorithms on the dataset.

To test the code, run `python sim.py --seed 1231` in a terminal environment.  You can modify the computing device with the `--device` flag (default is `cpu`).  


## Usage

To use this codebase, follow the steps outlined below:

1. Import the `run_inference` function from the `inf` module.

```python
from inf import run_inference
from feu.prep import prep
import os
```

We handle conversion to PyTorch file from other formats listed below:

- .pkl
- .pickle
- .npy
- .txt
- .csv
- .xlsx
- .nwb [TODO]

2. Preprocess the data
```python
original_data = 'test_data/array_30_200.pkl'    # path to the data

preprocessed_data = 'outputs/processed_data.p'  # path to save the preprocessed data

if not os.path.exists('outputs'):
    os.makedirs('outputs')

prep(original_data, preprocessed_data)
```

3. Execute the `run_inference` function with the necessary parameters.

```python
best_assigns, best_params = run_inference(
                                        preprocessed_data,
                                        title='demo',   # title of the run
                                        device='cpu',   # device to run the model on (cpu or cuda)
                                        iterations=1500,# number of iterations to run the model
                                        concentration=1,# probability of increasing the number of clusters. 1 is the default and 
                                        max_clusters=20,# maximum number of clusters to consider   
                                        timepoint=0,    # timepoint of stimulus if there is one
                                        seed=None       # seed for reproducibility
                                        )
```

3. Print the data types of the outputs to verify.

```python
print(type(best_assigns))
print(type(best_params))
```

## Outputs

Running the `run_inference` function generates the following outputs:

- **Best Cluster Assignments CSV:** Saves the best cluster assignments to `outputs/sim{run}_best_assigns.csv`.
- **Best Cluster Parameters CSV:** Saves the best cluster parameters to `outputs/sim{run}_best_params.csv`.
- **Scatter Plot PNG:** Saves a scatter plot of the cluster parameters to `outputs/sim{run}_params.png`.
- **Heatmap PNG:** Saves a heatmap of the cluster assignments to `outputs/sim{run}_assigns.png`.
- **Cluster Assignments CSV:** Saves the cluster assignments to `outputs/sim{run}_assigns.csv`.
- **Cluster Parameters CSV:** Saves the cluster parameters to `outputs/sim{run}_params.csv`.

These outputs are stored in the `outputs` directory, allowing easy access and analysis of the results.

Assuming you have already run the inference process but visualizations and best assign/paramaters didn't run, you can find best assignment and paramaters and visualize the results using the following code:

```python
best_assigns, best_params = viz_heatmap(title, num_iterations, 'cluster_assigns.csv', 'cluster_params.tsv', max_clusters=max_clusters)
```
See [run_inf.py](https://github.com/cyrilakafia/feu-pipeline/blob/main/run_inf.py) for more details on how to use the pipeline