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

Ensure that you have PyTorch installed in your environment. If PyTorch is not installed, you can install it via pip:

```bash
pip install torch
```

## Usage

To use this codebase, follow the steps outlined below:

1. Import the `run_inference` function from the `inf` module.

```python
from inf import run_inference
```

2. Execute the `run_inference` function with the necessary parameters.

```python
best_assigns, best_params = run_inference('yourdata.p', 
                                          title='demo', 
                                          device='cpu', 
                                          iterations=5, 
                                          concentration=1, 
                                          max_clusters=20, 
                                          timepoint=100, 
                                          seed=None
                                          )
```

3. Print the types of the outputs to verify.

```python
print(type(best_assigns))
print(type(best_params))
```

## Outputs

Running the `run_inference` function generates the following outputs:

- **Cluster Assignments CSV:** Saves the best cluster assignments to `outputs/sim{run}_best_assigns.csv`.
- **Cluster Parameters CSV:** Saves the best cluster parameters to `outputs/sim{run}_best_params.csv`.
- **Scatter Plot PNG:** Saves a scatter plot of the cluster parameters to `outputs/sim{run}_params.png`.
- **Heatmap PNG:** Saves a heatmap of the cluster assignments to `outputs/sim{run}_assigns.png`.

These outputs are stored in the `outputs` directory, allowing easy access and analysis of the results.