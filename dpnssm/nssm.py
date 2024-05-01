from typing import Optional, Union

import torch
from torch import Tensor
from torch.distributions import Normal


def particle_filter(
    obs: Tensor,
    var: Tensor,
    inputs: Optional[Tensor] = None,
    A: Optional[Tensor] = None,
    B: Optional[Tensor] = None,
    C: Optional[Tensor] = None,
    num_particles: int = 32,
    mean_init: Union[Tensor, float] = 0,
    var_init: Union[Tensor, float] = 1,
    num_bin_trials: Union[Tensor, int] = 0,
):
    batch_size, num_states = obs.shape
    settings = {"dtype": obs.dtype, "device": obs.device}
    curr_log_likelihood = 0

    # Define variables that will be stored and returned
    particles = torch.zeros((batch_size, num_states, num_particles), **settings)
    log_weights = torch.zeros((batch_size, num_states, num_particles), **settings)
    log_likelihoods = torch.zeros((batch_size, num_states), **settings)
    log_state_trans_norms = torch.zeros(
        (batch_size, num_states, num_particles), **settings
    )
    eff_samp_sizes = torch.zeros((batch_size, num_states), **settings)

    # If inputs not supplied, then set them equal to 0
    if inputs is None:
        inputs = torch.zeros((batch_size, num_states), **settings)

    # If A, B, C are not supplied, then run bootstrap particle filter
    if A is None:
        A = torch.zeros((batch_size, num_states), **settings)
    if B is None:
        B = torch.zeros((batch_size, num_states), **settings)
    if C is None:
        C = torch.zeros((batch_size, num_states), **settings)

    if isinstance(mean_init, float):
        mean_init = mean_init * torch.ones(batch_size, **settings)

    if isinstance(var_init, float):
        var_init = var_init * torch.ones(batch_size, **settings)

    if isinstance(num_bin_trials, int):
        num_bin_trials = num_bin_trials * torch.ones(batch_size, **settings)

    # Intialize state t=0
    curr_twisted_var = 1 / (1 / var_init + 2 * A[:, 0])  # B
    curr_twisted_means = (
        (mean_init + inputs[:, 0]) / var_init - B[:, 0]
    ) * curr_twisted_var  # B
    curr_twisted_means = curr_twisted_means.unsqueeze(dim=0).expand(
        num_particles, -1
    )  # S x B

    log_init_prior_norm = (
        0.5 * torch.log(curr_twisted_var)
        - 0.5 * torch.log(var_init)
        + (0.5 * curr_twisted_var)
        * ((mean_init + inputs[:, 0]) / var_init - B[:, 0]) ** 2
        - 0.5 * (mean_init + inputs[:, 0]) ** 2 / var_init
        - C[:, 0]
    )  # B

    for t in range(num_states):
        # Sample new particles and compute new weights
        # curr_particles = Normal(
        #     curr_twisted_means, torch.sqrt(curr_twisted_var)
        # ).sample()  # S x B
        curr_particles = curr_twisted_means + torch.sqrt(
            curr_twisted_var
        ) * torch.randn_like(
            curr_twisted_means
        )  # S x B

        # curr_log_state_dep_likes = -(
        #     (num_bin_trials - obs[:, t]) * curr_particles
        # ) - num_bin_trials * torch.log(
        #     1 + torch.exp(-curr_particles)
        # )  # S x B
        curr_log_state_dep_likes = -(
            (num_bin_trials - obs[:, t]) * curr_particles
        ) - num_bin_trials * torch.logsumexp(
            torch.stack([torch.zeros_like(curr_particles), -curr_particles], dim=0),
            dim=0,
        )  # S x B

        curr_log_weights = (
            curr_log_state_dep_likes
            + A[:, t] * curr_particles**2
            + B[:, t] * curr_particles
            + C[:, t]
        )  # S x B

        if t == 0:
            curr_log_weights += log_init_prior_norm

        if t == num_states - 1:
            curr_log_state_trans_norms = torch.zeros(
                (num_particles, batch_size), **settings
            )  # S x B
        else:
            next_twisted_var = 1 / (1 / var + 2 * A[:, t + 1])  # B
            curr_log_state_trans_norms = (
                0.5 * torch.log(next_twisted_var)
                - 0.5 * torch.log(var)
                + (0.5 * next_twisted_var)
                * ((curr_particles + inputs[:, t + 1]) / var - B[:, t + 1]) ** 2
                - 0.5 * (curr_particles + inputs[:, t + 1]) ** 2 / var
                - C[:, t + 1]
            )  # S x B

        curr_log_weights += curr_log_state_trans_norms  # S x B

        # Compute cumulative log likelihood and effective sample size
        max_log_weights = torch.max(curr_log_weights, dim=0).values
        curr_weights = torch.exp(curr_log_weights - max_log_weights)  # S x B
        curr_log_likelihood += max_log_weights + torch.log(
            torch.mean(curr_weights, dim=0)
        )  # B
        curr_weights_norm = curr_weights / curr_weights.sum(dim=0)  # S x B
        if curr_weights_norm.isnan().any():
            import pdb

            pdb.set_trace()
        curr_ess = 1 / ((curr_weights_norm**2).sum(dim=0))  # B

        # Save necessary information
        particles[:, t] = curr_particles.T
        log_weights[:, t] = curr_log_weights.T
        log_likelihoods[:, t] = curr_log_likelihood
        log_state_trans_norms[:, t] = curr_log_state_trans_norms.T
        eff_samp_sizes[:, t] = curr_ess

        # Set up next iteration
        if t < num_states - 1:
            curr_ancestors = systematic_resampling(
                curr_particles.T, curr_weights_norm.T
            ).T  # S x B
            curr_twisted_var = next_twisted_var  # B
            curr_twisted_means = (
                (curr_ancestors + inputs[:, t + 1]) / var - B[:, t + 1]
            ) * curr_twisted_var  # S x B

    return (
        particles,
        log_weights,
        log_likelihoods,
        log_state_trans_norms,
        eff_samp_sizes,
    )


# Systematic resampling scheme
def systematic_resampling(particles: Tensor, weights: Tensor):
    batch_size, num_particles = particles.shape
    boundaries = torch.cumsum(weights, dim=-1)  # B x S
    positions = (
        torch.rand(batch_size).unsqueeze(dim=-1) + torch.arange(num_particles)
    ) / num_particles
    indices = torch.searchsorted(boundaries, positions)
    try:
        resampled = torch.gather(particles, dim=-1, index=indices)
    except:
        import pdb

        pdb.set_trace()
    return resampled


# Approximate backwards recursion to find approximation to optimal twisting procedure
def approx_back_recursion(
    var: Tensor,
    particles: Tensor,
    inputs: Tensor,
    log_weights: Tensor,
    log_state_trans_norms: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    var_init: Union[Tensor, float] = 1.0,
):
    batch_size, num_states, num_particles = particles.shape

    # Define variables that will be stored and returned
    A_new = torch.zeros_like(A)
    B_new = torch.zeros_like(B)
    C_new = torch.zeros_like(C)

    for t in range(num_states - 1, -1, -1):
        # Define variables (X, y) for regression
        x = particles[:, t].T  # S x B
        ones = torch.ones(
            num_particles, batch_size, dtype=particles.dtype, device=particles.device
        )  # S x B
        X = torch.stack([x**2, x, ones], dim=0)  # 3 x S x B
        y = -log_weights[:, t].T  # S x B
        if t < num_states - 1:
            next_twisted_var_new = 1 / (1 / var + 2 * A_new[:, t + 1])  # B
            curr_log_state_trans_norms_new = (
                0.5 * torch.log(next_twisted_var_new)
                - 0.5 * torch.log(var)
                + (0.5 * next_twisted_var_new)
                * ((x + inputs[:, t + 1]) / var - B_new[:, t + 1]) ** 2
                - 0.5 * (x + inputs[:, t + 1]) ** 2 / var
                - C_new[:, t + 1]
            )  # S x B
            y += (
                -curr_log_state_trans_norms_new + log_state_trans_norms[:, t].T
            )  # S x B

        # Run linear regression to get new twisting coefficients A_new, B_new, C_new

        output = (
            (torch.linalg.lstsq(X.transpose(0, -1), y.T.unsqueeze(dim=-1)))
            .solution.squeeze(dim=-1)
            .T
        )  # 3 x B
        a, b, c = output

        A_new[:, t] = A[:, t] + a
        B_new[:, t] = B[:, t] + b
        C_new[:, t] = C[:, t] + c

        # Check that twisting coefficients do not violate constraints
        if t == 0:
            if torch.any(A_new[:, t] < -1 / (2 * var_init)):
                raise ValueError(
                    "Constraint violated: A_new is less than it should be."
                )
        else:
            if torch.any(A_new[:, t] < -1 / (2 * var)):
                raise ValueError(
                    "Constraint violated: A_new is less than it should be."
                )

    return A_new, B_new, C_new


def nssm_log_likelihood(
    obs: Tensor,
    var: Tensor,
    num_bin_trials: Union[int, Tensor],
    inputs: Optional[Tensor] = None,
    num_particles: int = 128,
    mean_init: Union[float, Tensor] = 0,
    var_init: Union[float, Tensor] = 1,
    ess_threshold: int = 80,
    max_iters: int = 3,
):
    if len(obs) != len(var):
        raise ValueError("Note that obs and var must have same batch size.")

    batch_size, num_states = obs.shape
    settings = {"dtype": obs.dtype, "device": obs.device}

    A = torch.zeros((batch_size, num_states), **settings)
    B = torch.zeros((batch_size, num_states), **settings)
    C = torch.zeros((batch_size, num_states), **settings)

    # If inputs not supplied, then set them equal to 0
    if inputs is None:
        inputs = torch.zeros((batch_size, num_states), **settings)

    # Run bootstrap particle filter
    (
        particles,
        log_weights,
        log_likelihoods,
        log_state_trans_norms,
        eff_samp_sizes,
    ) = particle_filter(
        obs,
        var,
        inputs,
        A,
        B,
        C,
        num_particles,
        mean_init,
        var_init,
        num_bin_trials,
    )
    rel_ess = eff_samp_sizes / num_particles * 100

    # Run controlled SMC until all ess are below threshold
    iters = 0
    while ((rel_ess < ess_threshold).any()) & (iters < max_iters):
        A, B, C = approx_back_recursion(
            var,
            particles,
            inputs,
            log_weights,
            log_state_trans_norms,
            A,
            B,
            C,
            var_init,
        )

        (
            particles,
            log_weights,
            log_likelihoods,
            log_state_trans_norms,
            eff_samp_sizes,
        ) = particle_filter(
            obs,
            var,
            inputs,
            A,
            B,
            C,
            num_particles,
            mean_init,
            var_init,
            num_bin_trials,
        )
        rel_ess = eff_samp_sizes / num_particles * 100
        iters += 1

    return log_likelihoods[:, -1]
