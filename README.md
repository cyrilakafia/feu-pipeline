# Functional Encoding Units (FEU)

Web Application is available at [feucluster.com](https://www.feucluster.com/)

FEU is a PyTorch-based pipeline for discovering Functional Encoding Units (clusters of similar time series) using a Dirichlet Process nonparametric state-space model (DPnSSM). Given 2D time-series data (entities × timepoints), FEU infers cluster assignments and per-cluster parameters and produces ready-to-use visualizations and CSVs.

Note: This library is under active development. APIs may change and some features are experimental.

## Installation

We recommend using a fresh environment.

1) Create and activate an environment (example with conda):

```bash
conda create -n feu python=3.10
conda activate feu
```

2) Install from GitHub (recommended):

```bash
pip install -U git+https://github.com/cyrilakafia/feu-inference.git
```

Alternatively, clone locally for development:

```bash
git clone https://github.com/cyrilakafia/feu-inference.git
cd feu-pipeline
pip install -e .
```

Dependencies (installed automatically): torch, numpy, pandas, matplotlib, openpyxl, h5py.

## Data formats

FEU expects 2D time series shaped as (N, T):

- N = number of entities/variables (e.g., neurons)
- T = number of timepoints

Supported input file types (converted internally via `feu.prep.prep`):

- .pkl, .pickle (pickled numpy arrays or lists)
- .npy (NumPy)
- .csv (no header)
- .xls, .xlsx (Excel)
- .txt (plain text matrix)
- .p, .pt (PyTorch tensor or tuple)

NWB is currently a placeholder (not yet implemented).

Binary and count data are both supported (e.g., spike trains or spike counts). Other data types may work but are not currently recommended.

## Quickstart

This example shows how to preprocess data and run inference.

```python
import os
from feu.prep import prep
from feu.inf import run_inference

title = "demo_run"           # will be used as the output directory name
device = "cpu"               # or "cuda" if available
iterations = 500
concentration = 1.0
max_clusters = 20
t_stimulus = 0               # set > 0 if you have a stimulus onset timepoint
num_trials = 10              # int or per-entity list/array

# 1) Convert your data to a FEU-ready PyTorch file (.p)
src = "test_data/array_30_200.npy"    # any supported format (see above)
dst = f"{title}/processed_data.p"
os.makedirs(title, exist_ok=True)
prep(src, dst)

# 2) Run inference
best_assigns, best_params, output_prefix = run_inference(
    data=dst,
    device=device,
    iterations=iterations,
    title=title,
    concentration=concentration,
    max_clusters=max_clusters,
    num_trials=num_trials,
    t_stimulus=t_stimulus,
    seed=42,
    figures=True,
    # Optional early-stopping controls:
    early_stopping=False,
    conv_tol=1e-1,
    conv_patience=10,
    cluster_count_tol=1,
    # Optional free-form text added to the log file
    additional_info="My first FEU run"
)

print(type(best_assigns))   # pandas.DataFrame, 1×N (row of cluster ids)
print(type(best_params))    # pandas.DataFrame, N×2 (columns: jump, phasicity)
print(output_prefix)        # e.g., demo_run/run1/demo_run
```

### What gets written where?

Output files are created in a run subfolder, so multiple runs don’t overwrite each other:

```
<title>/
  run1/
    <title>_assigns.csv         # per-iteration cluster assignments
    <title>_params.tsv          # per-iteration cluster parameters
    <title>_log.txt             # run log with context and settings
    <title>_assigns.png         # heatmap of average co-assignment (saved by post-processing)
    <title>_params.png          # scatter of (phasicity, jump) per entity (post-processing)
    <title>_ensembles.png       # ensemble parameter summary (post-processing)
    <title>_best_assigns.csv    # single-row best assignment (post-processing)
    <title>_best_params.csv     # N×2 table of best per-entity (jump, phasicity)
    <title>_ensemble_params.csv # mean ensemble parameters
    <title>_raster_clusters.png # optional if you render rasters
```

The `run_inference` call returns `(best_assigns, best_params, output_prefix)`, where
`output_prefix` includes the run folder, e.g., `demo_run/run1/demo_run`.

## API overview

### Preprocessing

```python
from feu.prep import prep
prep(src_path: str, dst_path: str) -> None
```

Converts supported formats into a PyTorch file at `dst_path`. The saved file contains a tuple `(tensor,)` where `tensor` has shape `(N, T)`. Data must be 2D.

### Inference

```python
from feu.inf import run_inference

best_assigns, best_params, output_prefix = run_inference(
    data,                    # str path to .p/.pt (or a 2D torch.Tensor)
    device,                  # 'cpu' or 'cuda'
    iterations,              # number of Gibbs iterations
    title='outputs',
    concentration=1.0,       # DP concentration
    max_clusters=20,
    num_trials=1,            # int or list/np.ndarray (length N)
    t_stimulus=0,            # stimulus onset timepoint; 0 means no stimulus
    avg_rate=0.1,            # baseline rate used when t_stimulus==0
    seed=42,
    figures=True,            # generate figures in post-processing
    early_stopping=False,
    conv_tol=1e-1,
    conv_patience=10,
    cluster_count_tol=1,
    additional_info=None     # string stored in the run log
)
```

Notes:

- If `data` is a path, FEU loads it with `torch.load(path)[0]` (expecting a tuple as produced by `prep`).
- `num_trials` can be an integer (same for all entities) or a list/array of length N to specify per-entity trial counts.
- When `t_stimulus > 0`, FEU estimates pre-stimulus rates and uses a "jump" parameter at stimulus onset. When `t_stimulus == 0`, a baseline `avg_rate` is used.
- Enable `early_stopping=True` to stop when the number of clusters and parameters stabilize (controlled by `conv_tol`, `conv_patience`, and `cluster_count_tol`).

### Simulation (optional)

```python
from feu.sim import run_sim

run_sim(
    title="sim", seed=1231, device="cpu",
    rate_change_noise=0.02, fix_num_neurons=None,
    num_neurons=10, amplitude=1.0, iterations=1500
)
```

Generates synthetic spike-count-like data, runs inference (with early stopping enabled inside), and writes outputs to `<title>_<seed>/run*/...`. Use this to sanity-check your environment and visualize typical outputs.

### Visualization helpers

```python
from feu.postinf import find_best_clust_and_params, make_raster_fig

# Recompute best assignments/params and figures later, if needed
best_assigns, best_params = find_best_clust_and_params(
    title="demo_run",
    output_folder=output_prefix,   # e.g., demo_run/run1/demo_run
    max_clusters=20,
    figures=True
)

# Optional raster plots by cluster (if you have trials × time rasters per entity)
# 'data_df' must be a DataFrame with a 'data' column of 2D arrays (trials × timepoints)
make_raster_fig(data_df, t_stimulus=0, best_assigns=best_assigns,
                title="demo_run", output_folder=output_prefix)
```

Old references to `viz_heatmap` have been replaced by `find_best_clust_and_params`.

## Tips and troubleshooting

- Data must be 2D (entities × timepoints). If your source is 3D (e.g., trials × entities × time), reduce along trials to get a 2D array (e.g., sum or mean across trials) before calling `prep`.
- If you pass `num_trials` as a list/array, its length must equal N (the number of rows in your data).
- GPU: set `device="cuda"` if a CUDA-enabled PyTorch build is available.
- If `prep` reports "Unsupported file type", convert to one of the supported formats above.
- Logs are stored next to your run outputs as `<title>_log.txt` and include the parameters used.

## Colab demo

Click the badge to run FEU in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cyrilakafia/feu-pipeline/blob/main/feu_colab_demo.ipynb)

## License

This project is released under the terms in `LICENSE.txt`.

