import torch
from torch.distributions import Uniform, Normal
from feu.nssm import nssm_log_likelihood
from feu.sb import infer_dp
from feu.visualize_heatmap import viz_heatmap

def run_inference(file, title, device, iterations, concentration=1.0, max_clusters=20, num_trials = 1, t_stimulus=100, seed=None):
    """
    Run the inference process on the given data.

    Parameters:
    file (str): Path to the file containing the data to be processed.
    title (str): Name for the run. Eg. 'rodent_5'
    device (str): Device to be used for PyTorch computation. Options: 'cpu', 'cuda'
    iterations (int): Number of iterations for the inference process.
    concentration (float, optional): Probability of increased number of clusters. Defaults to 1.0.
    max_clusters (int, optional): Maximum number of clusters for the Dirichlet Process. Defaults to 20.
    t_stimulus (int, optional): Timepoint for stimulus. Defaults to 100.
    seed (int, optional): Seed for the random number generator. Defaults to None.

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

    if t_stimulus <= 0:
        raise ValueError('t_stimulus should be greater than 0')
    
    # Print parameters
    print(f'{title}, Iterations {iterations}, Concentration {concentration}, Max Clusters {max_clusters}, Number of trials {num_trials}, Stimulus Timepoint {t_stimulus}, Seed {seed}')

    torch.set_default_dtype(torch.double)
    torch.set_default_device(device)

    obs_all = torch.load(file, map_location=device)[0]
    print("Device", obs_all.device)

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

    def calc_bssm_log_like_sim(obs, params, ns):
        jumps = params[:, 0]
        log_vars = params[:, 1]
        p_inits = torch.sum(obs[:, :t_stimulus], dim=-1) / (num_trials * t_stimulus)   # TO DO
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

    output = infer_dp(
        obs_all.to(device),
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

    # print(f'Title of {title}, Iterations {iterations}')
    print(f'{title} run for {iterations} iterations - Inference done')
    best_assings, best_params = viz_heatmap(title, iterations, max_clusters=max_clusters)
    print('Pipeline done')
    return best_assings, best_params

# # Example of how to call the function
# output = run_inference('outputs/sim1231_true.p', '1', 'cpu', 1500, 1.0)
