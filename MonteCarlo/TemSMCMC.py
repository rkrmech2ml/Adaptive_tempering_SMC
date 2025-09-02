import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Path to the dataset ---
dataset_path = r"C:\Users\RAHUL K R\.cache\kagglehub\datasets\burnoutminer\heights-and-weights-dataset\versions\1"
file_path = os.path.join(dataset_path, "SOCR-HeightWeight.csv")

# --- Load data ---
df = pd.read_csv(file_path)
data = df['Height(Inches)'].values
n_data = len(data)
print("First 5 heights:", data[:5])

# --- Log-likelihood for μ and σ ---
def log_likelihood_mu_sigma(params):
    mu, sigma = params
    if sigma <= 0:  # σ must be positive
        return -np.inf
    return -0.5 * np.sum(((data - mu)/sigma)**2) - n_data*np.log(sigma*np.sqrt(2*np.pi))

# --- 2D MCMC sampler ---
def metropolis_hastings_2d(log_likelihood, num_samples=10000, proposal_std=[1.0,0.5], x0=None):
    if x0 is None:
        x0 = [np.mean(data), np.std(data)]
    samples = np.empty((num_samples, 2))
    x = np.array(x0)
    fx = log_likelihood(x)
    
    for t in range(num_samples):
        x_prop = x + np.random.normal(0, proposal_std, size=2)
        f_prop = log_likelihood(x_prop)
        if np.log(np.random.rand()) < f_prop - fx:
            x, fx = x_prop, f_prop
        samples[t] = x
    return samples



# --- SMC with MH rejuvenation ---
def smc_with_mh_2d(log_likelihood, num_particles=2000, mh_steps=5, mh_step_size=[0.2,0.1]):
    # Initialize particles uniformly around data mean/std
    mu_min, mu_max = np.min(data), np.max(data)
    sigma_min, sigma_max = 0.1, np.std(data)*3
    particles = np.column_stack((
        np.random.uniform(mu_min, mu_max, size=num_particles),
        np.random.uniform(sigma_min, sigma_max, size=num_particles)
    ))
    weights = np.ones(num_particles)/num_particles
    num_steps = 500
    betas = np.linspace(0, 1, num_steps)
    ess_history = []

    for i in range(1, num_steps):
        beta_diff = betas[i] - betas[i-1]
        log_w = beta_diff * np.array([log_likelihood(p) for p in particles])
        log_w -= np.max(log_w)  # prevent overflow
        weights *= np.exp(log_w)
        weights /= np.sum(weights)
        
        ess = 1.0 / np.sum(weights**2)
        ess_history.append(ess)

        # Resample if ESS low
        if ess < num_particles/2:
            idx = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
            particles = particles[idx]
            weights = np.ones(num_particles)/num_particles

    # MH rejuvenation
    for i in range(num_particles):
        x = particles[i]
        fx = log_likelihood(x)
        for _ in range(mh_steps):
            x_prop = x + np.random.normal(0, mh_step_size)
            f_prop = log_likelihood(x_prop)
            if np.log(np.random.rand()) < f_prop - fx:
                x, fx = x_prop, f_prop
        particles[i] = x

    return particles, ess_history


# --- Run MCMC ---
np.random.seed(0)
samples_mcmc = metropolis_hastings_2d(log_likelihood_mu_sigma, num_samples=20000, proposal_std=[0.5,0.2])



# --- Run SMC ---
particles_smc, ess_history = smc_with_mh_2d(log_likelihood_mu_sigma, num_particles=4000, mh_steps=5, mh_step_size=[0.2,0.1])



# --- Plots ---




