import torch
from torch.distributions import Uniform, Normal, Binomial, Poisson

from dpnssm.nssm import nssm_log_likelihood
from dpnssm.sb import infer_dp

from dpnssm.visualize_heatmap import viz_heatmap

import argparse
import os.path

import subprocess

parser = argparse.ArgumentParser(
    description="Simulation of clustering neural data with DPnSSM"
)
parser.add_argument("--seed", default=1231, type=int)
parser.add_argument("--device", default="cpu", type=str)
args = parser.parse_args()

torch.set_default_dtype(torch.double)
torch.set_default_device(args.device)

SEED = args.seed
torch.manual_seed(SEED)

title = f"{SEED}"
iterations = 5

delta = 1 / 1000
obs_all = []
rate_change_noise = 0.02
num_neurons = 10
amplitude = 1.0

class_counts = Poisson(num_neurons).sample((5,)).to(torch.int)

for _ in range(class_counts[0]):
    l1 = Uniform(10, 25).sample()
    l2 = l1 * torch.exp(amplitude + Normal(0, rate_change_noise).sample())
    p1 = l1 * delta
    p2 = l2 * delta
    obs = torch.cat(
        [Binomial(225, p1).sample((100,)), Binomial(225, p2).sample((200,))]
    )
    obs_all.append(obs)
for _ in range(class_counts[1]):
    l1 = Uniform(10, 25).sample()
    l2 = l1 * torch.exp(-amplitude + Normal(0, rate_change_noise).sample())
    p1 = l1 * delta
    p2 = l2 * delta
    obs = torch.cat(
        [Binomial(225, p1).sample((100,)), Binomial(225, p2).sample((200,))]
    )
    obs_all.append(obs)
for _ in range(class_counts[2]):
    l1 = Uniform(10, 25).sample()
    p1 = l1 * delta
    obs = Binomial(225, p1).sample((300,))
    obs_all.append(obs)
for _ in range(class_counts[3]):
    l1 = Uniform(10, 25).sample()
    l2 = l1 * torch.exp(amplitude + Normal(0, rate_change_noise).sample())
    p1 = l1 * delta
    p2 = l2 * delta
    obs = torch.cat(
        [
            Binomial(225, p1).sample((100,)),
            Binomial(225, p2).sample((100,)),
            Binomial(225, p1).sample((100,)),
        ]
    )
    obs_all.append(obs)
for _ in range(class_counts[4]):
    l1 = Uniform(10, 25).sample()
    l2 = l1 * torch.exp(-amplitude + Normal(0, rate_change_noise).sample())
    p1 = l1 * delta
    p2 = l2 * delta
    obs = torch.cat(
        [
            Binomial(225, p1).sample((100,)),
            Binomial(225, p2).sample((100,)),
            Binomial(225, p1).sample((100,)),
        ]
    )
    obs_all.append(obs)

obs_all = torch.stack(obs_all)

print("Class counts: ", class_counts)

# Create output folder
if not os.path.exists('outputs'):
    os.makedirs('outputs')
torch.save((obs_all.cpu(), class_counts.cpu()), f"outputs/sim{SEED}_true.p")

print("Simulated data")
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


# Log likelihood function
def calc_bssm_log_like_sim(obs, params, ns):
    jumps = params[:, 0]
    log_vars = params[:, 1]
    p_inits = torch.sum(obs[:, :100], dim=-1) / (45 * 5 * 100)
    mean_inits = torch.log(p_inits)
    variances = torch.exp(log_vars)
    return nssm_log_likelihood(
        obs[:, 100:],
        var=variances,
        num_particles=64,
        num_bin_trials=45 * 5,
        max_iters=3,
        mean_init=mean_inits + jumps,
        var_init=1e-10,
    )


output = infer_dp(
    obs_all,
    calc_bssm_log_like_sim,
    1.0,
    samp_base,
    base_logpdf,
    num_gibbs_iters=iterations,
    samp_prop=samp_prop,
    out_prefix=f"outputs/sim{SEED}",
    seed=SEED,
)

print(f'Simulated {title} run - Inference done')
best_assings, best_params = viz_heatmap(title, iterations)
print('Pipeline done')