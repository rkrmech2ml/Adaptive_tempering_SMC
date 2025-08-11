from laplace_solve import laplace_solver
from resampling import resample
from markov_kernal import markov_kernel
from regula_falsi import  NewBeta,CoeffVariation
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
x = np.arange(0, 1.00001, h)
    #print((x))
y = np.arange(0, 1.00001, h)
X, Y = np.meshgrid(x, y)
plt.figure()
plt.contourf(X, Y, experiment_exact, cmap='viridis')
plt.colorbar(label='u(x,y)')
plt.title('2D Contour Plot of the Solution to Laplace Equation')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.show()



# Add noise to simulate measurement errors
num_noisy_versions = 5
noise_levels = np.random.uniform(0, 0.05, size=num_noisy_versions)
print(f"Noise levels: {noise_levels}")

experiment_noisy_versions = [
    experiment_exact + np.random.normal(0, sigma, size=experiment_exact.shape)
    for sigma in noise_levels
]




experiment = np.mean(experiment_noisy_versions, axis=0)
plt.figure()
plt.contourf(X, Y, experiment, cmap='viridis')
plt.colorbar(label='u(x,y)')
plt.title('2D Contour Plot of the Solution to Laplace Equation')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.show()

# Plot experiment (noisy average) and experiment_exact (ground truth)
plt.figure(figsize=(8, 4))
plt.plot(experiment_exact, label='(ground truth)', marker='o')
plt.plot(experiment, label=' (noisy average)', marker='x')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison of experiment_exact and experiment')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Particle initialization
num_particles = 5000
particles = np.random.uniform(
    low=[-5.0, -1.0], 
    high=[4.0, 7.0], 
    size=(num_particles, 2)
)

print(f"Initial particles: {particles}")
particles_initial = particles.copy()
particles = np.array(particles)
particles_initial = np.array(particles_initial)  # Store initial particles for reference
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
plt.plot(particles[:,0], potentials, 'bo', alpha=0.6)  # Plot first dimension of particles
plt.xlabel('Particle Value (0th column)')
plt.ylabel('Potential')
plt.title('Potential vs Particle Value (0th column)')
plt.grid(True)
plt.tight_layout()
plt.show()
print(f"Potential values: {np.shape(potentials)}")

# -----------------------------
# Adaptive Tempering SMC Loop
# -----------------------------

beta = 0.000

while beta < 0.9999:


    #-------------------------
    #step 1: calculating new beta
    #-------------------------
    #delta = new_beta_test(potentials, weights, beta)
    #delta = NewBeta(potentials, weights, beta)
    delta = 0.01  # Limit the step size to avoid large jumps
    beta = min(beta + delta, 1.0)
    CV_cal= CoeffVariation(potentials, weights, beta,delta)
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
    particles = markov_kernel(particles,particles_initial, resampled_particles, experiment, n, h, beta)
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
    # Calculate mean and standard deviation of both particle columns



    # Top-left: Histogram of particles[:, 0]
    axs[0, 0].hist(particles[:, 0], bins=50, density=True, alpha=0.7, color='orange', label='Current')
    axs[0, 0].set_title(f'Particles[:, 0] (β = {beta:.3f})')
    axs[0, 0].set_xlabel('Particle Value (0th column)')
    axs[0, 0].set_ylabel('Density')
    axs[0, 0].legend()

    # Top-right: Histogram of initial particles[:, 0]
    axs[0, 1].hist(particles_initial[:, 0], bins=50, density=True, alpha=0.7, color='green', label='Initial')
    axs[0, 1].set_title('Initial Particles[:, 0]')
    axs[0, 1].set_xlabel('Particle Value (0th column)')
    axs[0, 1].set_ylabel('Density')
    axs[0, 1].legend()

    # Bottom-left: Histogram of particles[:, 1]
    axs[1, 0].hist(particles[:, 1], bins=50, density=True, alpha=0.7, color='blue', label='Current')
    axs[1, 0].set_title(f'Particles[:, 1] (β = {beta:.3f})')
    axs[1, 0].set_xlabel('Particle Value (1st column)')
    axs[1, 0].set_ylabel('Density')
    axs[1, 0].legend()

    # Bottom-right: Histogram of initial particles[:, 1]
    axs[1, 1].hist(particles_initial[:, 1], bins=50, density=True, alpha=0.7, color='red', label='Initial')
    axs[1, 1].set_title('Initial Particles[:, 1]')
    axs[1, 1].set_xlabel('Particle Value (1st column)')
    axs[1, 1].set_ylabel('Density')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'results/BetaStep0_001/particle_histograms_beta_{beta:.3f}.png')
    plt.close()

    
    
    #Track beta and CV history for plotting
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
    plt.savefig(f'results/BetaStep0_001/cv_vs_beta for beta step{delta}.png')
    plt.close()


mean_0 = np.mean(particles[:, 0])
std_0 = np.std(particles[:, 0])
mean_1 = np.mean(particles[:, 1])
std_1 = np.std(particles[:, 1])
print(f"Particles[:, 0] Mean: {mean_0:.4f}, Std: {std_0:.4f}")
print(f"Particles[:, 1] Mean: {mean_1:.4f}, Std: {std_1:.4f}")