from typing import Optional, Callable

import time
import torch
from torch import Tensor
from torch.distributions import Categorical, Normal


# Run DPnSSM inference algorithm
def infer_dp(
    obs: Tensor,
    calc_log_like: Callable[[Tensor, Tensor, Tensor], Tensor],
    concentration: float,
    samp_base: Callable[[int], Tensor],
    base_logpdf: Callable[[Tensor], Tensor],
    num_gibbs_iters: int = 500,
    num_clusters_init: int = 1,
    samp_prop: Optional[Callable[[Tensor], Tensor]] = None,
    num_aux: int = 5,
    dump_file: Optional[str] = None,
    seed: Optional[int] = None,
):
    settings = {"dtype": obs.dtype, "device": obs.device}

    # Set seed for reproducible results
    if seed is not None:
        torch.random.manual_seed(seed)

    # Initialize variables
    num_obs = len(obs)
    num_clusters = num_clusters_init
    cluster_ids = Categorical(
        probs=1 / num_clusters * torch.ones(num_clusters, **settings)
    ).sample((num_obs,))
    params = samp_base(num_clusters)
    num_params = params.size(dim=-1)

    if samp_prop is None:
        samp_prop = lambda x: Normal(
            x, 1 / num_params * torch.ones_like(params)
        ).sample()

    cluster_ids_samps = []
    params_samps = []
    num_clusters_samps = []
    num_accept = 0
    num_total = 0

    for i in range(num_gibbs_iters):
        t = time.time()

        if i > 0:
            cluster_ids = cluster_ids_samps[-1].clone()
            params = params_samps[-1].clone()

        # Run one iteration
        (
            cluster_ids,
            params,
            num_clusters,
            accept_stats,
        ) = iterate_metropolis_within_gibbs_for_dp(
            obs,
            cluster_ids,
            params,
            calc_log_like,
            concentration,
            samp_base,
            base_logpdf,
            samp_prop,
            num_aux,
        )
        num_accept += accept_stats["accepted"].sum()
        num_total += accept_stats["accepted"].sum() + accept_stats["rejected"].sum()

        # Save samples
        cluster_ids_samps.append(cluster_ids)
        params_samps.append(params)
        num_clusters_samps.append(num_clusters)

        # Print and log output
        iter_time = time.time() - t
        print(
            f"iter: #{i:3d} | time: {iter_time:4.2f} | num clusters: {num_clusters} "
            f"| counts: {str(torch.bincount(cluster_ids))}"
        )
        print(f"\t\t params: {str(params.T)}")
        print(f"\t\t accept prob: {num_accept / num_total:.3f}")

        # Write pickle
        # if dump_file:
        #     write_pickle((cluster_ids_samps, params_samps, num_clusters_samps), dump_file)

    return cluster_ids_samps, params_samps, num_clusters_samps


def iterate_metropolis_within_gibbs_for_dp(
    obs: Tensor,
    cluster_ids: Tensor,
    params: Tensor,
    calc_log_like: Callable[[Tensor, Tensor, Tensor], Tensor],
    concentration: float,
    samp_base: Callable[[Tensor], Tensor],
    base_logpdf: Callable[[Tensor], Tensor],
    samp_prop: Callable[[Tensor], Tensor],
    num_aux: int,
):
    settings = {"dtype": obs.dtype, "device": obs.device}
    num_obs = len(obs)
    params_curr_loglikes = torch.zeros(num_obs)

    # Sample cluster assignment
    for n in range(num_obs):
        num_states = len(obs[n])

        # Drop current index and renumber all others
        cluster_ids_temp, num_clusters_temp, params_temp, counts_temp = renumber(
            cluster_ids, n, params
        )

        # Add auxiliary parameters
        params_temp = torch.cat([params_temp, samp_base(num_aux)], dim=0)
        num_clusters_temp += num_aux

        # Compute prior probability of each cluster using CRP
        nums = torch.cat(
            [counts_temp, (concentration / num_aux) * torch.ones(num_aux, **settings)]
        )
        denom = num_obs - 1 + concentration
        class_probs = nums / denom

        # Use log space for numerical stability
        log_class_probs = torch.log(class_probs)

        # Compute posterior probability of each cluster using likelihood
        log_class_cond_likes = calc_log_like(
            obs[n].expand((num_clusters_temp, -1)),
            params_temp,
            n * torch.ones_like(params_temp, dtype=torch.int),
        )

        class_post_probs = torch.softmax(log_class_probs + log_class_cond_likes, dim=0)

        # Sample new cluster identity and record parameter likelihood
        cluster_id = Categorical(class_post_probs).sample()
        cluster_ids[n] = cluster_id
        params_curr_loglikes[n] = log_class_cond_likes[cluster_id]

        # Update number of clusters and renumber them
        clusters = torch.unique(cluster_ids)
        num_clusters = len(clusters)
        for k in range(num_clusters):
            cluster_ids[cluster_ids == clusters[k]] = k
        params = params_temp[clusters]

    # Sample parameters
    params_prop = samp_prop(params)

    # Calculate param priors and likelihoods
    params_curr_logprobs = base_logpdf(params)
    params_prop_logprobs = base_logpdf(params_prop)
    params_prop_loglikes = calc_log_like(
        obs, params_prop[cluster_ids], torch.arange(num_obs, device=obs.device)
    )
    params_curr_loglikes = torch.bincount(cluster_ids, params_curr_loglikes)
    params_prop_loglikes = torch.bincount(cluster_ids, params_prop_loglikes)

    # Calculate param posteriors and Metropolis acceptances probs
    params_curr_logposts = params_curr_logprobs + params_curr_loglikes
    params_prop_logposts = params_prop_logprobs + params_prop_loglikes
    accept_probs = torch.min(
        torch.exp(params_prop_logposts - params_curr_logposts),
        torch.ones(num_clusters, **settings),
    )
    accepts = torch.rand(num_clusters, **settings) < accept_probs
    accept_stats = {"rejected": torch.sum(~accepts), "accepted": torch.sum(accepts)}

    # Determine new parameters
    params = torch.where(accepts.unsqueeze(dim=-1), params_prop, params)

    return cluster_ids, params, num_clusters, accept_stats


def renumber(cluster_ids, idx, params):
    # Remove ith cluster assignments
    cluster_ids_temp = torch.cat([cluster_ids[:idx], cluster_ids[idx + 1 :]], dim=0)
    counts_temp = torch.bincount(cluster_ids_temp)

    if (cluster_ids_temp == cluster_ids[idx]).any():
        # Number of clusters remains the same
        num_clusters_temp = params.shape[0]
        return cluster_ids_temp, num_clusters_temp, params, counts_temp
    else:
        # Number of clusters decreases by one
        cluster_ids_temp[cluster_ids_temp > idx] -= 1
        # params_temp = params[np.arange(params.shape[0]) != cluster_ids[idx]]
        exclude_id = cluster_ids[idx]
        params_temp = torch.cat([params[:exclude_id], params[exclude_id + 1 :]], dim=0)
        num_clusters_temp = len(params_temp)
        counts_temp = counts_temp[counts_temp > 0]
        return cluster_ids_temp, num_clusters_temp, params_temp, counts_temp
