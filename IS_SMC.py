from laplace_solve import laplace_solver
from resampling import resample
from markov_kernal import markov_kernel
from regula_falsi import NewBeta
from regula_falsiiTest import new_beta_test
from Potential import compute_potentials

import numpy as np
import matplotlib.pyplot as plt

#testing the version in git

# ---------------------------------
# Configuration & Initialization
# ---------------------------------

n = 10
true_particle = 1.0  # Ground truth (unknown parameter)
experiment_exact, h = laplace_solver(n, true_particle)

# Add noise to simulate measurement errors
num_noisy_versions = 5
noise_levels = np.random.uniform(0, 1, size=num_noisy_versions)
print(f"Noise levels: {noise_levels}")

experiment_noisy_versions = [
    experiment_exact + np.random.normal(0, sigma, size=experiment_exact.shape)
    for sigma in noise_levels
]
experiment = np.mean(experiment_noisy_versions, axis=0)

# Particle initialization
num_particles = 10
particles = np.random.uniform(low=-6.0, high=6.0, size=num_particles)
particles_initial = particles.copy()  # Store initial particles for reference
weights = [1.0 / num_particles] * num_particles  # Equal initial weights

# ----------------------------------
# Potential Calculation Function
# ----------------------------------

# Compute initial potential values
potentials = compute_potentials(n, particles, experiment, h)
#print(f"Potential values: {potentials}")
#print(f"Initial weights: {weights}")

# -----------------------------
# Adaptive Tempering SMC Loop
# -----------------------------

beta = 0.0

while beta < 0.9999:


    #-------------------------
    #step 1: calculating new beta
    #-------------------------
    #delta = new_beta_test(potentials, weights, beta)
    delta = NewBeta(potentials, weights, beta)
    #delta = 0.5  # Limit the step size to avoid large jumps
    beta = min(beta + delta, 1.0)
    print(f"Current beta: {beta:.3f}")

    

    # Update weights using tempered likelihood
    #-------------------------
    #step 2: importance sampling 
    #-------------------------
    likelihoods = [np.exp(-beta * pot) for pot in potentials]
    norm_factor = sum(likelihoods)
    weights = [lk / norm_factor for lk in likelihoods]
    #-------------------------
    #step 3: resampling 
    #-------------------------
    resampled_particles = resample(particles, weights)
    weights = [1.0 / num_particles] * num_particles  # Reset weights to  1/num_particles

    #-------------------------
    #step 4: Markov kernel  
    #-------------------------
    particles = markov_kernel(particles, resampled_particles, experiment, n, h, beta)

    #-------------------------
    #step 5: recalculating potentials for new particles  
    #-------------------------

    potentials = compute_potentials(n, particles, experiment, h)

    #print("length of particles after MCMC step:", (particles))

    # -----------------------------
    # Visualization
    # -----------------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].hist(particles, bins=50, density=True, alpha=0.7, color='orange')
    axs[0].set_title(f'Resampled Particle PDF (β = {beta:.3f})')
    axs[0].set_xlabel('Particle Value')
    axs[0].set_ylabel('Density')

    axs[1].hist(particles_initial, bins=50, density=True, alpha=0.7, color='green')
    axs[1].set_title('Particles After MCMC Step')
    axs[1].set_xlabel('Particle Value')
    axs[1].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(f'particle_resampled_pdf_beta_{beta:.3f}.png')
    plt.close()
