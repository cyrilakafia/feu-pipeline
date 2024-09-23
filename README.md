# Functional Encoding Units (FEU)

## Overview

This repository contains the implementation for extracting Functional Encoding Units (FEUs) using PyTorch. The code allows users to perform inference to identify optimal cluster assignments and parameters based on provided data.

Disclaimer: This algorithm and library is still under construction

## Installation

We recommend you create a new environment with conda before installing the package. If you don't have conda installed, you can install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Create a new environment using the commands below.

```bash
conda create -n 'feu' python=3.10   # create a new environment
conda activate feu      # activate the environment
```
Now you can install the package using the command below

```bash
pip install -U git+https://github.com/cyrilakafia/feu-pipeline.git
```

Alternatively, you can clone the repository.


```bash
git clone https://github.com/cyrilakafia/feu-pipeline.git
```

## Data 

The data processed by the pipeline is `two-dimensional time series data`, where the first dimension represents the variables to be clustered, and the second dimension represents the corresponding time series data/timepoints.

For example, consider electrophysiological data recorded from 25 neurons over a duration of 1000 seconds. Assuming that the data is binned at every second , the data will have a shape of (25, 1000). Each row corresponds to a different neuron, and each column represents sequential time points.

The algorithm currently supports Binary data (Data with only 0s and 1s `eg. spike train data where 1=spike; 0=no spike`) and Count data (Data with only positive integers `eg. spike train data for multiple trials where positive int represents number of spikes across all trials`). 

You can use the algorithm with other data types but this is not recommended. 

This structured data should be saved in a `PyTorch file with the .p` extension. Conversion from other data formats to the PyTorch format is supported within our pipeline, ensuring compatibility and ease of integration for various data sources. 

We handle conversion to PyTorch file from other formats listed below:

- .pkl
- .pickle
- .npy
- .txt
- .csv
- .xlsx
- .nwb [TODO]

To test the module, run the file `sim_test.py`. This will simulate and apply FEU to a new dataset.  


## Usage

To apply FEU to your data, follow the steps outlined below:

1. Import the `run_inference` and `prep` functions.

```python
from feu.inf import run_inference
from feu.prep import prep
import os
```
2. Define the parameters
```python
# Define the parameters
title = 'demo'
device = 'cpu'  #or 'cuda'
num_iters = 1500
conc = 1
max_clusters = 20
stimulus_timepoint = 200 #if no stimulus, set to 0
num_trials = 100
```

3. Preprocess the data
```python
original_data = 'test_data/array_30_200.pkl'

preprocessed_data = f'{title}/processed_data.p'

if not os.path.exists(title):
    os.makedirs(title)

prep(original_data, preprocessed_data)
```

4. Execute the `run_inference` function with the necessary parameters.

```python
best_assigns, best_params = run_inference(
                                        preprocessed_data,
                                        title=title,   # title of the run
                                        device=device,   # device to run the model on (cpu or cuda)
                                        iterations=num_iters,# number of iterations to run the model
                                        concentration=conc,# probability of increasing the number of clusters. 1 is the default and 
                                        max_clusters=max_clusters,# maximum number of clusters to consider 
                                        num_trials=num_trials,   # number of trials of the data
                                        t_stimulus=stimulus_timepoint,    # timepoint of stimulus. if no stimulus, set to 0
                                        seed=None,       # seed for reproducibility
                                        figures=True,    # whether to generate figures
                                        )
```

5. Print the data types of the outputs to verify.

```python
print(type(best_assigns))
print(type(best_params))
```

## Outputs

Running the `run_inference` function generates the following outputs:

- **Best Cluster Assignments CSV:** Saves the best cluster assignments to `title/{run}_best_assigns.csv`.
- **Best Cluster Parameters CSV:** Saves the best cluster parameters to `title/{run}_best_params.csv`.
- **Scatter Plot PNG:** Saves a scatter plot of the cluster parameters to `title/{run}_params.png`.
- **Heatmap PNG:** Saves a heatmap of the cluster assignments to `title/{run}_assigns.png`.
- **Cluster Assignments CSV:** Saves the cluster assignments to `title/{run}_assigns.csv`.
- **Cluster Parameters CSV:** Saves the cluster parameters to `title/{run}_params.csv`.

Assuming you have already run the inference process but visualizations and best assign/paramaters didn't run, you can find best assignment and paramaters and visualize the results using the following code:

```python
best_assigns, best_params = viz_heatmap(title, num_iterations, 'cluster_assigns.csv', 'cluster_params.tsv', max_clusters=max_clusters)
```
See [run_inf.py](https://github.com/cyrilakafia/feu-pipeline/blob/main/run_inf.py) for more details on how to use the pipeline

Click the button below to quickly test the pipeline in google colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cyrilakafia/feu-pipeline/blob/main/feu_colab_demo.ipynb)
