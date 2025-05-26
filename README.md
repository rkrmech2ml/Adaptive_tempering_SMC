# Adaptive Tempering SMC: Estimating Unknown Parameters

This project implements an **Adaptive Tempering Sequential Monte Carlo (SMC)** algorithm to solve inverse problems by estimating an unknown parameter (referred to as a "particle") from noisy experimental data. It leverages a powerful Bayesian inference approach to explore and converge on the most probable parameter values.

---

## How It Works: The Core Components

At its heart, the algorithm iteratively refines a distribution of potential parameter values through a series of intelligent steps:

* **Likelihood Definition (Implicit in `phi_i`):**
    The algorithm quantifies how well a simulated outcome, generated using a given "particle" value, matches the actual observed experimental data. This "goodness of fit" is represented by a **potential function** (derived from a Gaussian likelihood), where a smaller squared error between simulation and experiment indicates a better match.

* **Adaptive Tempering Schedule (`beta` parameter):**
    A crucial element is the **annealing schedule**, controlled by the `beta` parameter, which gradually increases from near zero to one.
    * **Early Stages (low `beta`):** The influence of the likelihood on the particle weights is minimal, allowing for broad exploration of the entire parameter space. This prevents the algorithm from getting stuck in local optima.
    * **Later Stages (high `beta`):** As `beta` approaches one, the likelihood term becomes dominant. This forces the particles to concentrate around the regions of highest probability, where the simulated data best aligns with the experimental observations.

* **Importance Sampling (Weighting):**
    Each potential "particle" value is assigned an **unnormalized weight (`phi_i`)** based on how well it explains the experimental data at the current `beta` level. These weights are then normalized to create a **probability distribution (`mu_list`)** over the current set of particles.

* **Resampling (`resample` function):**
    Particles are selected from the current set based on their calculated weights. This step ensures that particles with higher weights (i.e., those that better explain the data) are more likely to be carried forward to the next iteration. After resampling, the weights for the new set of particles are equalized.

* **Markov Kernel (`markov_kernel` function):**
    The resampled particles are then slightly perturbed or "wiggled" using a **Markov Chain Monte Carlo (MCMC) kernel**. This local exploration around promising parameter values is essential for preventing the algorithm from converging prematurely and for efficiently exploring the local parameter space.

---

## Visualization

The project includes plotting capabilities to visualize the probability distribution of the estimated particle value at different `beta` steps, demonstrating the convergence of the algorithm towards the true parameter value.
