from typing import Optional, Callable

import time
import torch
from torch import Tensor
from torch.distributions import Categorical, Normal, Beta
from torch.nn.functional import pad
import numpy as np
import os.path
import logging

# Run DPnSSM inference algorithm
def infer_dp(
    obs: Tensor,
    num_trials: Tensor,
    calc_log_like: Callable[[Tensor, Tensor, Tensor], Tensor],
    concentration: float,
    samp_base: Callable[[int], Tensor],
    base_logpdf: Callable[[Tensor], Tensor],
    num_gibbs_iters: int = 500,
    max_clusters: int = 20,
    samp_prop: Optional[Callable[[Tensor], Tensor]] = None,
    out_prefix: Optional[str] = None,
    seed: Optional[int] = None,
    t_stimulus: int = 0,
    additional_info: Optional[str] = None,

    # Early stopping controls
    early_stopping: bool = False,
    conv_tol: float = 1e-1, 
    conv_patience: int = 10,
    cluster_count_tol: int = 1,
):
    
    settings = {"dtype": obs.dtype, "device": obs.device}

    # Set seed for reproducible results
    if seed is not None:
        torch.random.manual_seed(seed)

    # Initialize variables
    num_obs = len(obs)
    num_clusters = max_clusters

    # Initialize stick-breaking fractions
    # stick_fracs = Beta(concentration, 1).sample((max_clusters - 1,))
    stick_fracs_np = np.random.beta(concentration, 1, size=(max_clusters - 1,)).astype(np.float32)
    stick_fracs = torch.tensor(stick_fracs_np, device=obs.device)

    # Initialize parameters
    params = samp_base(num_clusters)
    num_params = params.size(dim=-1)

    if samp_prop is None:
        samp_prop = lambda x: x + (1 / torch.sqrt(x.shape[-1])) * torch.randn_like(x)

    num_accept = 0
    num_total = 0

    # Early-stopping trackers
    prev_params = params.clone()
    stable_iters = 0
    params_active_list = []
        
    # create output directory and run subfolder
    if out_prefix is not None:
        base_dir = out_prefix.split("/")[0] if "/" in out_prefix else "."
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # Find the next available run folder
        run_counter = 1
        while True:
            run_folder = os.path.join(base_dir, f"run{run_counter}")
            if not os.path.exists(run_folder):
                os.makedirs(run_folder)
                break
            run_counter += 1
        
        # Update out_prefix to include the run folder
        filename = out_prefix.split("/")[-1] if "/" in out_prefix else out_prefix
        out_prefix = os.path.join(run_folder, filename)

        with open(out_prefix + "_assigns.csv", "w") as fassign:
            fassign.write("")

        with open(out_prefix + "_params.tsv", "w") as fcluster:
            fcluster.write("")
        # if os.path.isfile(out_prefix + "_assigns.txt") or os.path.isfile(
        #     out_prefix + "_params.txt"
        # ):
        #     raise ValueError(f"There already exists a file with prefix {out_prefix}.")

        # Configure logging
        log_file = out_prefix + "_log.txt"
        counter = 1
        while os.path.exists(log_file):
            log_file = f"{out_prefix}_log_{counter}.txt"
            counter += 1

        logging.basicConfig(level=logging.INFO, 
                            format='%(message)s',
                            handlers=[
                                logging.FileHandler(log_file),
                                logging.StreamHandler()
                            ])
        logging.info(f"Context on FEU run: {additional_info}\n")
        logging.info("===================================")
        logging.info("===================================\n")
        logging.info("Starting DPnSSM inference algorithm")
        # log the current date and time
        logging.info(time.strftime("%c"))
        logging.info("===================================")
        logging.info("Parameters:")
        if isinstance(num_trials, list) or isinstance(num_trials, np.ndarray):
            logging.info(f'Title: {out_prefix}')
            logging.info(f'Concentration: {concentration}')
            logging.info(f'Stimulus Timepoint: {t_stimulus}')
            logging.info(f'Iterations: {num_gibbs_iters}')
            logging.info(f'Max clusters: {max_clusters}')
            logging.info(f'Different number of trials for each timeseries so a list is provided i.e list')
            logging.info(f'Device: {str(obs.device)}')
            logging.info(f'Seed: {seed}')
        else:
            logging.info(f'Title: {out_prefix}')
            logging.info(f'Concentration: {concentration}')
            logging.info(f'Stimulus Timepoint: {t_stimulus}')
            logging.info(f'Iterations: {num_gibbs_iters}')
            logging.info(f'Max clusters: {max_clusters}')
            logging.info(f'Number of trials: {str(num_trials[0])}')
            logging.info(f'Device: {str(obs.device)}')
            logging.info(f'Seed: {seed}')
        logging.info("===================================")

    for i in range(num_gibbs_iters): 
        t = time.time()

        #########################################
        ### STEP 1: UPDATE CLUSTER IDENTITIES ###
        #########################################

        # Compute cluster priors
        stick_lengths = pad(stick_fracs, (0, 1), value=1) * pad(
            torch.cumprod(1 - stick_fracs, dim=-1), (1, 0), value=1
        )
        assert torch.abs(torch.sum(stick_lengths) - 1.0) < 1e-6
        log_priors = torch.log(stick_lengths)

        # Flatten tensors for batch computation
        obs_ = obs.unsqueeze(dim=1).expand((-1, num_clusters, -1)).flatten(0, 1)
        params_ = params.unsqueeze(dim=0).expand((num_obs, -1, -1)).flatten(0, 1)
        data_ids_ = (
            torch.arange(num_obs, **settings)
            .unsqueeze(dim=-1)
            .expand((-1, num_clusters))
            .flatten(0, 1)
        )

        num_trials_ = (num_trials).unsqueeze(dim=1).expand((-1, max_clusters)).flatten(0, 1)

        # Compute cluster likelihoods
        log_likelihoods = calc_log_like(obs_, params_, data_ids_, num_trials_).unflatten(
            dim=0, sizes=(num_obs, num_clusters)
        )

        # Sample cluster id from posterior
        posteriors = torch.softmax(log_priors + log_likelihoods, dim=-1)
        cluster_ids = Categorical(posteriors).sample()

        # Compute cluster counts
        cluster_counts = torch.bincount(cluster_ids, minlength=max_clusters)

        ###############################################
        ### STEP 2: UPDATE STICK-BREAKING FRACTIONS ###
        ###############################################

        # Sample stick fracs
        cluster_cumcounts = torch.cumsum(cluster_counts.flip([-1]), dim=-1).flip([-1])
        # stick_fracs = Beta(
        #     concentration0=1 + cluster_counts[:-1],
        #     concentration1=concentration + cluster_cumcounts[1:],
        # ).sample()

        stick_fracs_np = np.random.beta(1 + cluster_counts[:-1].cpu().numpy(), concentration + cluster_cumcounts[1:].cpu().numpy())
        stick_fracs = torch.tensor(stick_fracs_np, device=obs.device)



        #########################################
        ### STEP 3: UPDATE CLUSTER PARAMETERS ###
        #########################################

        # Sample proposals for active clusters
        active_clusters = cluster_counts > 0
        num_active = torch.sum(active_clusters)
        params_prop = samp_prop(params[active_clusters])

        # Renumber ids for active clusters
        id_to_active = torch.zeros(num_clusters, dtype=torch.long, device=obs.device)
        id_to_active[active_clusters] = torch.arange(num_active, device=obs.device)
        id_to_active[~active_clusters] = 1e6
        active_cluster_ids = id_to_active[cluster_ids]

        # Calculate param priors and likelihoods
        params_curr_logprobs = base_logpdf(params[active_clusters])
        params_prop_logprobs = base_logpdf(params_prop)

        params_curr_loglikes = log_likelihoods[
            torch.arange(num_obs, device=obs.device), cluster_ids
        ]
        params_prop_loglikes = calc_log_like(
            obs,
            params_prop[active_cluster_ids],
            torch.arange(num_obs, device=obs.device),
            num_trials,
        )
        params_curr_loglikes = torch.bincount(
            active_cluster_ids, weights=params_curr_loglikes
        )
        params_prop_loglikes = torch.bincount(
            active_cluster_ids, weights=params_prop_loglikes
        )

        # Calculate param posteriors and Metropolis acceptances probs
        params_curr_logposts = params_curr_logprobs + params_curr_loglikes
        params_prop_logposts = params_prop_logprobs + params_prop_loglikes
        accept_probs = torch.min(
            torch.exp(params_prop_logposts - params_curr_logposts),
            torch.ones(num_active, **settings),
        )
        accepts = torch.rand(num_active, **settings) < accept_probs
        num_accept += torch.sum(accepts).item()
        num_total += len(accepts)

        # Determine new active parameters
        params_active = torch.where(
            accepts.unsqueeze(dim=-1), params_prop, params[active_clusters]
        )

        # Sample new inactive parameters
        params_inactive = samp_base(torch.sum(~active_clusters))

        # Update parameters
        params[active_clusters] = params_active
        params[~active_clusters] = params_inactive


        # # Early stopping check
        # with torch.no_grad():
        #     logging.info(f"Current Params: {params_active}, Prev Params: {prev_params}")
        #     avg_abs_change = (params.abs() - prev_params.abs()).abs().mean().item() 
        #     logging.info(f"Average absolute parameter change: {avg_abs_change:.4f}")
        #     if avg_abs_change < conv_tol:
        #         stable_iters += 1
        #     else:
        #         stable_iters = 0
        #     prev_params = params.clone()

        # Save samples
        active_cluster_ids = active_cluster_ids.clone().cpu().numpy()
        params_active = params_active.clone().cpu().numpy()

        # Update params_active_list for early stopping
        params_active_list.append(params_active)
       
        # Print and log output
        iter_time = time.time() - t
        logging.info(
            f"iter: #{i:3d} | time: {iter_time:4.2f} | num clusters: {len(params_active)} "
            f"| counts: {str(np.bincount(active_cluster_ids))} | accept prob: {num_accept / num_total:.3f}"
        )
        logging.info(f"             params: {np.array2string(params_active.T[0].round(2))}")
        logging.info(f"                     {np.array2string(params_active.T[1].round(2))}")

        # Write outputs
        if out_prefix is not None:
            with open(out_prefix + "_assigns.csv", "a") as fassign:
                fassign.write(",".join([str(x) for x in active_cluster_ids]))
                fassign.write("\n")

            with open(out_prefix + "_params.tsv", "a") as fcluster:
                fcluster.write(
                    "\t".join([f"{p1:.5f},{p2:.5f}" for p1, p2 in params_active])
                )
                fcluster.write("\n")

        # Early stopping check
        last_n = int(conv_patience*5)

        # Only check for early stopping if we have enough iterations
        if early_stopping and len(params_active_list) >= last_n:
            last_n_entries = params_active_list[-last_n:]
            cluster_counts_last_n = [len(p) for p in last_n_entries]

            # If the number of clusters has been stable for the last n iterations within a tolerance of cluster_count_tol
            # Check if the number of clusters has not changed by more than cluster_count_tol in the last n iterations
            last_last_n = int(conv_patience)
            if max(cluster_counts_last_n) - min(cluster_counts_last_n) <= cluster_count_tol:
                last_last_n_entries = params_active_list[-last_last_n:]
                cluster_counts_last_last_n = [len(p) for p in last_last_n_entries]

                # If the number of clusters has been exactly the same for the last last_n iterations
                if max(cluster_counts_last_last_n) - min(cluster_counts_last_last_n) == 0:
                    with torch.no_grad():
                        #recursively find the difference between params in last_last_n_entries
                        avg_abs_changes = []
                        for j in range(1, len(last_last_n_entries)):
                            mean_abs_change = (torch.tensor(last_last_n_entries[j]).abs() - torch.tensor(last_last_n_entries[j-1]).abs()).abs().mean().item()
                            avg_abs_changes.append(mean_abs_change)
                        avg_abs_change = np.mean(avg_abs_changes)
                        if avg_abs_change < conv_tol:
                            logging.info(f"===============================")
                            logging.info(f"Early stopping at iteration {i}. Number of clusters has been stable for {last_n} iterations within a tolerance of {cluster_count_tol} clusters and exactly the same for the last {last_last_n} iterations.")
                            logging.info(f"Average absolute jump and phasicity parameter change over last {last_last_n} iterations: {avg_abs_change:.4f}")
                            logging.info(f"No significant parameter changes for {last_last_n} consecutive iterations. Tolerance: {conv_tol}")
                            logging.info(f"===============================")
                            break

    return out_prefix
