import torch
from torch.distributions import Uniform, Normal
from feu.nssm import nssm_log_likelihood
from feu.sb import infer_dp
from feu.visualize_heatmap import viz_heatmap, plot_raster, make_raster_fig
import numpy as np
import matplotlib.pyplot as plt

def run_inference(data, title, device, iterations, concentration=1.0, max_clusters=20, num_trials = 1, t_stimulus=0, seed=None, figures=True):
    """
    Run the inference process on the given data.

    Parameters:
    data (str): Path to the file containing the data to be processed.
    title (str): Name for the run. Eg. 'rodent_5'
    device (str): Device to be used for PyTorch computation. Options: 'cpu', 'cuda'
    iterations (int): Number of iterations for the inference process.
    concentration (float, optional): Probability of increased number of clusters. Defaults to 1.0.
    max_clusters (int, optional): Maximum number of clusters for the Dirichlet Process. Defaults to 20.
    num_trials (int, str, list, np.ndarray, optional): Number of trials for the data. This can be a number or a list of numbers separated by commas with the same length as the number of timeseries (neurons). Defaults to 1.
    t_stimulus (int, optional): Timepoint for stimulus. If there is no stimulus in the data, set this to 0.
    seed (int, optional): Seed for the random number generator. Defaults to None.
    figures (bool, optional): Whether to generate figures. Defaults to True.

    Returns:
    tuple: The best assignments and parameters resulting from the inference process.

    Outpus:
    Best cluster assignments CSV: Saves the best cluster assignments to `outputs/sim{run}_best_assigns.csv`.
    Best cluster parameters CSV: Saves the best cluster parameters to `outputs/sim{run}_best_params.csv`.
    Scatter Plot PNG: Saves a scatter plot of the cluster parameters to `outputs/sim{run}_params.png`.
    Heatmap PNG: Saves a heatmap of the cluster assignments to `outputs/sim{run}_assigns.png`.
    Cluster assignments CSV: Saves the cluster assignments to `outputs/sim{run}_assigns.csv`.
    Cluster parameters CSV: Saves the cluster parameters to `outputs/sim{run}_params.csv`.
    """

   # Check if number of trials is one number or many numbers seperated by a comma and print inference parameters

    torch.set_default_dtype(torch.double)
    torch.set_default_device(device)

    # check to see if data is file or tensor
    if isinstance(data, str):
        obs_all = torch.load(data, map_location=device)[0]
    else:   
        obs_all = data
    print("Device", obs_all.device)

    if ',' in str(num_trials):
        num_trials = [int(i) for i in str(num_trials).split(',')]
        num_trials = torch.tensor(num_trials)
        
        # Make sure length of num_trials list is the same as the number of timeseries
        if len(num_trials) != obs_all.shape[0]:
            print('Number of trials must be the same as the number of timeseries, using the max of the list provided')
            num_trials = int(np.max(num_trials))
            print(f'{title}, Iterations {iterations}, Concentration {concentration}, Max Clusters {max_clusters}, Number of trials {str(num_trials)} Stimulus Timepoint {t_stimulus}, Seed {seed}')
        else:
            print(f'{title}, Iterations {iterations}, Concentration {concentration}, Max Clusters {max_clusters}, Varying number of trials, Stimulus Timepoint {t_stimulus}, Seed {seed}')

    # Else if number of trials is a regular python list or a numpy array
    elif isinstance(num_trials, list) or isinstance(num_trials, np.ndarray):
        num_trials = torch.tensor(num_trials)
        print(f'{title}, Iterations {iterations}, Concentration {concentration}, Max Clusters {max_clusters}, Varying number of trials i.e list, Stimulus Timepoint {t_stimulus}, Seed {seed}')
    
    else: 
        num_trials = int(num_trials)
        num_trials = torch.ones(obs_all.shape[0]) * num_trials
        print(f'{title}, Iterations {iterations}, Concentration {concentration}, Max Clusters {max_clusters}, Number of trials {str(num_trials[0])}, Stimulus Timepoint {t_stimulus}, Seed {seed}')
    

    # base distribution (G)
    def samp_base(num):
        jumps = Normal(0, 2).sample((num,))
        log_variances = Uniform(-20, 0).sample((num,))
        all_samps = torch.stack([jumps, log_variances], dim=-1)
        return all_samps

    def base_logpdf(params):
        jump_logpdf = Normal(0, 2).log_prob(params[:, 0])
        oob_mask = (params[:, 1] < -20) | (params[:, 1] > 0)
        variances_logpdf = torch.zeros_like(jump_logpdf)
        variances_logpdf[oob_mask] = -float("inf")
        variances_logpdf[~oob_mask] = Uniform(-20, 0).log_prob(params[:, 1][~oob_mask])

        total_logpdf = jump_logpdf + variances_logpdf
        return total_logpdf

    def samp_prop(params):
        jump_prop = Normal(params[:, 0], 0.25).sample()
        log_var_prop = Normal(params[:, 1], 0.5).sample()
        return torch.stack([jump_prop, log_var_prop], dim=-1)

    def calc_bssm_log_like_sim(obs, params, ns, num_trials):
        jumps = params[:, 0]
        log_vars = params[:, 1]
        p_inits = torch.sum(obs[:, :t_stimulus], dim=-1) / (num_trials * t_stimulus)
        mean_inits = torch.log(p_inits) 
        variances = torch.exp(log_vars)
        return nssm_log_likelihood(
            obs[:, t_stimulus:],
            var=variances,
            num_particles=64,
            num_bin_trials=num_trials,
            max_iters=3,
            mean_init=mean_inits + jumps, 
            var_init=1e-10,
        )
    
    # If there is no stimulus in the data
    def calc_bssm_log_like_sim_no_stim(obs, params, ns):
        jumps = params[:, 0]
        log_vars = params[:, 1]
        variances = torch.exp(log_vars)
        return nssm_log_likelihood(
            obs,
            var=variances,
            num_particles=64,
            num_bin_trials=num_trials,
            max_iters=3,
            mean_init=jumps,   
            var_init=1e-10,
        )    

    if t_stimulus == 0:
        output = infer_dp(
            obs_all.to(device),
            num_trials,
            calc_bssm_log_like_sim_no_stim,
            concentration,
            samp_base,
            base_logpdf,
            num_gibbs_iters=iterations,
            samp_prop=samp_prop,
            out_prefix=f"outputs/sim{title}",
            max_clusters=max_clusters,
            seed=seed,
        )

    else:
        output = infer_dp(
            obs_all.to(device),
            num_trials,
            calc_bssm_log_like_sim,
            concentration,
            samp_base,
            base_logpdf,
            num_gibbs_iters=iterations,
            samp_prop=samp_prop,
            out_prefix=f"outputs/sim{title}",
            max_clusters=max_clusters,
            seed=seed,
        )



    print(f'{title} run for {iterations} iterations - Inference done')
    best_assings, best_params = viz_heatmap(title, iterations, max_clusters=max_clusters, figures=figures)

    #Viz rasters

    # if t_stimulus > 0:

    #     make_raster_fig(obs_all, t_stimulus=t_stimulus, best_assigns=best_assings, title=title)

    print('Pipeline done')

    return best_assings, best_params


# # Example of how to call the function
# output = run_inference('outputs/sim1231_true.p', '1', 'cpu', 1500, 1.0)
