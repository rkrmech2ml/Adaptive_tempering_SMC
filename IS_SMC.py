from laplace_solve import laplace_solver
from resampling import resample
from markov_kernal import markov_kernel
from regula_falsi import CoeffVariation
from regula_falsiiTest import new_beta_test
from Potential import compute_potentials

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#testing the version in git

# ---------------------------------
# Configuration & Initialization
# ---------------------------------

n = 10
true_particle1 = [1.0,3.0]
 # Ground truth (unknown parameter)
experiment_exact, h = laplace_solver(n, true_particle1)

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
num_particles = 10000
particles = np.random.uniform(
    low=[-5.0, -10.0], 
    high=[5.0, 10.0], 
    size=(num_particles, 2)
)

print(f"Initial particles: {particles}")
particles_initial = particles.copy()  # Store initial particles for reference
weights = [1.0 / num_particles] * num_particles
 # Equal initial weights

# ----------------------------------
# Potential Calculation Function
# ----------------------------------

# Compute initial potential values
potentials = compute_potentials(n, particles, experiment, h)
print(f"Initial potentials: {np.shape(potentials)}")
print(f"Initial particles shape: {particles.shape}")
plt.figure(figsize=(8, 4))
plt.plot(particles[:,1], potentials, 'bo', alpha=0.6)  # Plot first dimension of particles
plt.xlabel('Particle Value (0th column)')
plt.ylabel('Potential')
plt.title('Potential vs Particle Value (0th column)')
plt.grid(True)
plt.tight_layout()
plt.show()
print(f"Potential values: {np.shape(potentials)}")
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
    #delta = NewBeta(potentials, weights, beta)
    delta = 0.1  # Limit the step size to avoid large jumps
    beta = min(beta + delta, 1.0)
    CV_cal= CoeffVariation(potentials, weights, beta)
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
    #print(f"Resampled particles: {resampled_particles}")
 
    weights = [1.0 / num_particles] * num_particles  # Reset weights to  1/num_particles 
    #print(f"weights reset: {weights}")
    #-------------------------
    #step 4: Markov kernel  
    #-------------------------
    particles = markov_kernel(particles, resampled_particles, experiment, n, h, beta)
    #print(f"Particles after Markov kernel: {particles}")

    #-------------------------
    #step 5: recalculating potentials for new particles  
    #-------------------------

    potentials = compute_potentials(n, particles, experiment, h)
    #print(f"Potentials after recalculation: {potentials}")

    #print("length of particles after MCMC step:", (particles))

    # -----------------------------
    # Visualization
    # -----------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    particles = np.array(particles)
    particles_initial = np.array(particles_initial)

    # Top-left: Histogram of particles[:, 0]
    axs[0, 0].hist(particles[:, 0], bins=50, density=True, alpha=0.7, color='orange', label='Current')

    # Overlay Gaussian PDF for particles[:, 0]
    mu0, std0 = np.mean(particles[:, 0]), np.std(particles[:, 0])
    x0 = np.linspace(np.min(particles[:, 0]), np.max(particles[:, 0]), 200)
    axs[0, 0].plot(x0, norm.pdf(x0, mu0, std0), 'k--', label=f'Gaussian PDF\nμ={mu0:.2f}, σ={std0:.2f}')

    axs[0, 0].set_title(f'Particles[:, 0] (β = {beta:.3f})')
    axs[0, 0].set_xlabel('Particle Value (0th column)')
    axs[0, 0].set_ylabel('Density')
    axs[0, 0].legend()

    # Top-right: Histogram of initial particles[:, 0]
    axs[0, 1].hist(particles_initial[:, 0], bins=50, density=True, alpha=0.7, color='green', label='Initial')

    # Overlay Gaussian PDF for initial particles[:, 0]
    min1_init = np.min(particles_initial[:, 0])
    max1_init = np.max(particles_initial[:, 0])
    x1_init = np.linspace(min1_init, max1_init, 200)
    from scipy.stats import uniform
    axs[0, 1].plot(
        x1_init,
        uniform.pdf(x1_init, loc=min1_init, scale=max1_init - min1_init),
        'k--',
        label=f'Uniform PDF\nmin={min1_init:.2f}, max={max1_init:.2f}'
    )
    axs[0, 1].set_title('Initial Particles[:, 0]')
    axs[0, 1].set_xlabel('Particle Value (0th column)')
    axs[0, 1].set_ylabel('Density')
    axs[0, 1].legend()

    # Bottom-left: Histogram of particles[:, 1]
    axs[1, 0].hist(particles[:, 1], bins=50, density=True, alpha=0.7, color='blue', label='Current')

    # Overlay Gaussian PDF for particles[:, 1]
    mu1, std1 = np.mean(particles[:, 1]), np.std(particles[:, 1])
    x1 = np.linspace(np.min(particles[:, 1]), np.max(particles[:, 1]), 200)
    axs[1, 0].plot(x1, norm.pdf(x1, mu1, std1), 'k--', label=f'Gaussian PDF\nμ={mu1:.2f}, σ={std1:.2f}')

    axs[1, 0].set_title(f'Particles[:, 1] (β = {beta:.3f})')
    axs[1, 0].set_xlabel('Particle Value (1st column)')
    axs[1, 0].set_ylabel('Density')
    axs[1, 0].legend()

    # Bottom-right: Histogram of initial particles[:, 1]
    axs[1, 1].hist(particles_initial[:, 1], bins=50, density=True, alpha=0.7, color='red', label='Initial')

    # Overlay Gaussian PDF for initial particles[:, 1]
    min1_init = np.min(particles_initial[:, 1])
    max1_init = np.max(particles_initial[:, 1])
    x1_init = np.linspace(min1_init, max1_init, 200)
    from scipy.stats import uniform
    axs[1, 1].plot(
        x1_init,
        uniform.pdf(x1_init, loc=min1_init, scale=max1_init - min1_init),
        'k--',
        label=f'Uniform PDF\nmin={min1_init:.2f}, max={max1_init:.2f}'
    )
    axs[1, 1].set_title('Initial Particles[:, 1]')
    axs[1, 1].set_xlabel('Particle Value (1st column)')
    axs[1, 1].set_ylabel('Density')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'particle_histograms_beta_{beta:.3f}.png')
    plt.close()

    # Track beta and CV history for plotting
    if 'beta_history' not in locals():
        beta_history = []
        CV_history = []

    beta_history.append(beta)
    CV_history.append(CV_cal)

    # After the SMC loop, plot the variation of beta and CV
    plt.figure(figsize=(7, 4))
    plt.plot(beta_history, CV_history, marker='o')
    plt.xlabel('Beta')
    plt.ylabel('Coefficient of Variation')
    plt.title('Coefficient of Variation vs Beta')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cv_vs_beta.png')
    plt.close()

