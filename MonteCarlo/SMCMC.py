import numpy as np
import matplotlib.pyplot as plt

# --- Target distribution: Mixture of 2 Gaussians ---
def target_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2*np.pi)

# ----------------------------------------------------------------
# 1) MCMC (Metropolis-Hastings)
# ----------------------------------------------------------------
def metropolis_hastings(num_samples=5000, proposal_std=3.0):
    samples = []
    x = 0.0  # start point
    for _ in range(num_samples):
        x_new = x + np.random.normal(0, proposal_std)
        accept_ratio = target_pdf(x_new) / target_pdf(x)
        if np.random.rand() < accept_ratio:
            x = x_new
        samples.append(x)
    return np.array(samples)

# ----------------------------------------------------------------
# 2) SMC (Importance Sampling + Resampling, no tempering)
# ----------------------------------------------------------------
def smc(num_particles=2000):
    # Proposal distribution: Uniform[-6, 6]
    particles = np.random.uniform(-6, 6, size=num_particles)
    weights = np.array([target_pdf(x) for x in particles])
    weights /= np.sum(weights)

    # Resample (multinomial)
    idx = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
    resampled_particles = particles[idx]
    return resampled_particles

# ----------------------------------------------------------------
# Run both methods
# ----------------------------------------------------------------
mcmc_samples = metropolis_hastings()
smc_samples = smc()

# Plotting
x = np.linspace(-6, 6, 1000)
plt.figure(figsize=(12,5))

# True distribution
plt.plot(x, target_pdf(x), 'k-', lw=2, label="Target PDF")

# MCMC samples
plt.hist(mcmc_samples, bins=50, density=True, alpha=0.5, label="MCMC samples")

# SMC samples
plt.hist(smc_samples, bins=50, density=True, alpha=0.5, label="SMC samples")

plt.legend()
plt.title("MCMC vs SMC (no tempering)")
plt.show()
