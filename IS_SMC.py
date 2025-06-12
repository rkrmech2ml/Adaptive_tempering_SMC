from laplace_solve import laplace_solver
from resampling import resample
from markov_kernal import markov_kernel
from regula_falsi import NewBeta

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration & Initialization
# -----------------------------

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
weights = [1.0 / num_particles] * num_particles  # Equal initial weights

# -----------------------------
# Potential Calculation Function
# -----------------------------

def compute_potentials(n, particles, experiment, h):
    """Calculate potential  for each particle."""
    potentials = []
    for p in particles:
        sim_output, _ = laplace_solver(n, p)
        l2_error = h**2 * np.sum((sim_output - experiment)**2) / 2
        potentials.append(l2_error)
    return potentials

# Compute initial potential values
potentials = compute_potentials(n, particles, experiment, h)
#print(f"Potential values: {potentials}")
#print(f"Initial weights: {weights}")

# -----------------------------
# Adaptive Tempering SMC Loop
# -----------------------------

beta = 0.0

while beta < 1.0:
    delta = NewBeta(potentials, weights, beta)
    beta += delta
    print(f"Updated beta: {beta:.4f}")

    # Update weights using tempered likelihood
    likelihoods = [np.exp(-beta * pot) for pot in potentials]
    norm_factor = sum(likelihoods)
    weights = [lk / norm_factor for lk in likelihoods]

    # Resampling step
    resampled_particles = resample(particles, weights)
    weights = [1.0 / num_particles] * num_particles  # Reset weights to  1/num_particles

    # MCMC step using Markov kernel
    particles = markov_kernel(particles, resampled_particles, experiment, n, h, beta)

    # -----------------------------
    # Visualization
    # -----------------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].hist(resampled_particles, bins=50, density=True, alpha=0.7, color='orange')
    axs[0].set_title(f'Resampled Particle PDF (β = {beta:.3f})')
    axs[0].set_xlabel('Particle Value')
    axs[0].set_ylabel('Density')

    axs[1].hist(particles, bins=50, density=True, alpha=0.7, color='green')
    axs[1].set_title('Particles After MCMC Step')
    axs[1].set_xlabel('Particle Value')
    axs[1].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(f'particle_resampled_pdf_beta_{beta:.3f}.png')
    plt.close()
