from laplace_solve import laplace_solver
from resampling import resample
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
def markov_kernel(particle_init, particle_resampled, experiment, n, h, beta):
    gamma = 0.1

    for i in range(len(particle_resampled)):
        # Generate a new candidate value for particle using the Markov kernel
        v = (np.sqrt(1 - gamma**2) * particle_resampled[i] + 
            gamma * (max(particle_init) - min(particle_init)) * np.random.uniform(-1, 1))
        
        # Calculate likelihoods for the new candidate and the current value
        like_hood_v = np.exp( -(h**2 * np.sum((laplace_solver(n, v)[0] - experiment)**2) / 2))
        like_hood_ui = np.exp(-(h**2 * np.sum((laplace_solver(n, particle_resampled[i])[0] - experiment)**2) / 2))
        
        # Accept or reject the new candidate based on the likelihood ratio
        if (like_hood_v / like_hood_ui)**beta > np.random.uniform(0, 1):
            particle_resampled[i] = v  # Update the parameter with the new value
    return particle_resampled
