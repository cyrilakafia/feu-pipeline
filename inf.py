import torch
from torch.distributions import Uniform, Normal
from dpnssm.nssm import nssm_log_likelihood
from dpnssm.sb import infer_dp
from dpnssm.visualize_heatmap import viz_heatmap

def run_inference(file, title, device, iterations, concentration=1.0, max_clusters=20, timepoint=100, seed=None):
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
        p_inits = torch.sum(obs[:, :100], dim=-1) / (45 * 5 * 100)   # TO DO
        mean_inits = torch.log(p_inits)
        variances = torch.exp(log_vars)
        return nssm_log_likelihood(
            obs[:, timepoint:],
            var=variances,
            num_particles=64,
            num_bin_trials=45 * 5,
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

    print(f'Title {title}, Iterations {iterations}')
    best_assings, best_params = viz_heatmap(title, iterations)
    return best_assings, best_params

# # Example of how to call the function
# output = run_inference('outputs/sim1231_true.p', '1', 'cpu', 1500, 1.0)
