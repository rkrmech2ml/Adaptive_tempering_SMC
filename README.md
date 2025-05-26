feat: Implement Adaptive Tempering for parameter estimation

This commit introduces an Adaptive Tempering algorithm to estimate an unknown physical parameter (e.g., a particle's value) by fitting a simulation to noisy experimental data.

Key features include:
- Iterative refinement of an initial particle distribution using an adaptive tempering schedule.
- Importance sampling and resampling to propagate promising parameter values.
- Markov chain Monte Carlo (MCMC) kernel for exploring the parameter space.
- Visualization of particle distribution convergence at each tempering step.

This approach is crucial for solving inverse problems where direct analytical solutions are infeasible, providing a robust method for uncertainty quantification of the estimated parameter.
