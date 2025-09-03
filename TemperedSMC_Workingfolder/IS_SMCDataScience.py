# from laplace_solve import laplace_solver
from resampling import resample
from markov_kernalData import markov_kernel
# from regula_falsi import NewBeta
# from regula_falsiiTest import new_beta_test
from PotentialDataS import compute_potentials

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import numpy as np

from sklearn.preprocessing import StandardScaler

# 1. Create scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# 2. Fit and transform



# Load dataset
california = fetch_california_housing()
X, y = california.data[:10], california.target[:10]


X_scaled = scaler_X.fit_transform(X)        # shape: (n_samples, 8)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # shape: (n_samples,)

# 3. Use these going forward
X = X_scaled
y = y_scaled


# Normalize target y to have zero mean and unit variance


# Print basic info
print("Shape of features:", X.shape)        # (20640, 8)
print("target:", y)          # (20640,)
print("Feature names:", california.feature_names)

# Plot all 8 features as scatter subplots (4 plots, each with 2 features)
# Plot all 8 features as scatter subplots (4 plots, each with 2 features)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
feature_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

for ax, (i, j) in zip(axs.flatten(), feature_pairs):
    sc = ax.scatter(
        X[:, i], X[:, j],
        c=y,
        cmap='viridis',
        alpha=0.5
    )
    ax.set_xlabel(california.feature_names[i])
    ax.set_ylabel(california.feature_names[j])
    ax.set_title(f'{california.feature_names[i]} vs {california.feature_names[j]}')
    ax.grid(True, linestyle='--', alpha=0.3)

fig.tight_layout(pad=4.0)
cbar = fig.colorbar(sc, ax=axs, orientation='vertical', label='Target value', shrink=0.8)
plt.show()


n_particles = 100
n_features = X.shape[1]  # should be 8

# Random initialization of particles from standard normal
# Find min and max for each feature (column-wise)
X_min = X.min(axis=0)
X_max = X.max(axis=0)

# Sample uniformly within the feature-wise min-max range
np.random.seed(42)
particles = X_min + (X_max - X_min) * np.random.rand(n_particles, n_features)
particles_initial = particles.copy()  # Store initial particles for reference
print("the intial particles:", particles.shape)


# Uniform initial weights
weights = np.ones(n_particles) / n_particles


potentials = compute_potentials(X, particles, y, sigma=0.5)

beta=0.5

while beta < 0.9999:


    #-------------------------
    #step 1: calculating new beta
    #-------------------------
    delta = 0.1  # Limit the step size to avoid large jumps
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
 
    weights = [1.0 / n_particles] * n_particles  # Reset weights to  1/num_particles 
    #print(f"weights reset: {weights}")
    #-------------------------
    #step 4: Markov kernel  
    #-------------------------
    particles = markov_kernel(particles, resampled_particles,X, y, beta)
    

    #-------------------------
    #step 5: recalculating potentials for new particles  
    #-------------------------

    potentials = compute_potentials(X, particles, y, sigma=0.5)




    #-----------------------------
    #Visualization
    #-----------------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].hist(particles, bins=50, density=True, alpha=0.7)
    axs[0].set_title(f'Resampled Particle PDF (β = {beta:.3f})')
    axs[0].set_xlabel('Particle Value')
    axs[0].set_ylabel('Density')

    axs[1].hist(particles_initial, bins=50, density=True, alpha=0.7)
    axs[1].set_title('Particles After MCMC Step')
    axs[1].set_xlabel('Particle Value')
    axs[1].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(f'particle_resampled_pdf_beta_{beta:.3f}.png')
    plt.close()





#print("Initial potentials:", potentials.shape)


plt.hist(potentials, bins=30, alpha=0.7)
plt.xlabel('Potential (Negative log-likelihood)')
plt.ylabel('Number of particles')
plt.title('Initial Potentials Distribution')
plt.grid(True)
plt.show()






# ---------------------------------
# Configuration & Initialization
# ---------------------------------

# n = 10
# true_particle = 1.0  # Ground truth (unknown parameter)
# experiment_exact, h = laplace_solver(n, true_particle)

# # Add noise to simulate measurement errors
# num_noisy_versions = 5
# noise_levels = np.random.uniform(0, 1, size=num_noisy_versions)
# print(f"Noise levels: {noise_levels}")

# experiment_noisy_versions = [
#     experiment_exact + np.random.normal(0, sigma, size=experiment_exact.shape)
#     for sigma in noise_levels
# ]
# experiment = np.mean(experiment_noisy_versions, axis=0)

# # Particle initialization
# num_particles = 100
# particles = np.random.uniform(low=-6.0, high=6.0, size=num_particles)
# particles_initial = particles.copy()  # Store initial particles for reference
# weights = [1.0 / num_particles] * num_particles  # Equal initial weights

# # ----------------------------------
# # Potential Calculation Function
# # ----------------------------------

# # Compute initial potential values
# potentials = compute_potentials(n, particles, experiment, h)
# plt.figure(figsize=(8, 4))
# plt.plot(particles, potentials, 'bo', alpha=0.6)
# plt.xlabel('Particle Value')
# plt.ylabel('Potential')
# plt.title('Potential vs Particle Value')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# #print(f"Potential values: {potentials}")
# #print(f"Initial weights: {weights}")

# # -----------------------------
# # Adaptive Tempering SMC Loop
# # -----------------------------

# beta = 0.0

# while beta < 0.9999:


#     #-------------------------
#     #step 1: calculating new beta
#     #-------------------------
#     #delta = new_beta_test(potentials, weights, beta)
#     delta = NewBeta(potentials, weights, beta)
#     #delta = 0.1  # Limit the step size to avoid large jumps
#     beta = min(beta + delta, 1.0)
#     print(f"Current beta: {beta:.3f}")

    

#     # Update weights using tempered likelihood
#     #-------------------------
#     #step 2: importance sampling 
#     #-------------------------
#     likelihoods = [np.exp(-beta * pot) for pot in potentials]
#     norm_factor = sum(likelihoods)
#     weights = [lk / norm_factor for lk in likelihoods]
#     #-------------------------
#     #step 3: resampling 
#     #-------------------------
#     resampled_particles = resample(particles, weights)
 
#     weights = [1.0 / num_particles] * num_particles  # Reset weights to  1/num_particles 
#     #print(f"weights reset: {weights}")
#     #-------------------------
#     #step 4: Markov kernel  
#     #-------------------------
#     particles = markov_kernel(particles, resampled_particles, experiment, n, h, beta)

#     #-------------------------
#     #step 5: recalculating potentials for new particles  
#     #-------------------------

#     potentials = compute_potentials(n, particles, experiment, h)

    #print("length of particles after MCMC step:", (particles))

    # -----------------------------
    # Visualization
    # -----------------------------
    # fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # axs[0].hist(particles, bins=50, density=True, alpha=0.7, color='orange')
    # axs[0].set_title(f'Resampled Particle PDF (β = {beta:.3f})')
    # axs[0].set_xlabel('Particle Value')
    # axs[0].set_ylabel('Density')

    # axs[1].hist(particles_initial, bins=50, density=True, alpha=0.7, color='green')
    # axs[1].set_title('Particles After MCMC Step')
    # axs[1].set_xlabel('Particle Value')
    # axs[1].set_ylabel('Density')

    # plt.tight_layout()
    # plt.savefig(f'particle_resampled_pdf_beta_{beta:.3f}.png')
    # plt.close()
