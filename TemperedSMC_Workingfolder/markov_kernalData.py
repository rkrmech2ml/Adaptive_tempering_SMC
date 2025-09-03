from ModelDef import model_linreg
import numpy as np



def markov_kernel(particles, resampled_particles,X, y, beta):
    # print("length of particles:", len(particles))
    # print("length of resampled particles:", len(resampled_particles))

    gamma = 0.1
    i=0
    for i in range(len(resampled_particles)):
        # Generate a new candidate value for particle using the Markov kernel
        v = (np.sqrt(1 - gamma**2) * resampled_particles[i] + 
            gamma * (np.max(particles, axis=0) - np.min(particles, axis=0)) * np.random.uniform(-1, 1))
        
        # Calculate likelihoods for the new candidate and the current value
        like_hood_v = np.exp(-(np.sum((model_linreg(X, v) - y) ** 2) / 2))
        like_hood_ui = np.exp(-(np.sum((model_linreg(X, resampled_particles[i]) - y) ** 2) / 2))
        # Accept or reject the new candidate based on the likelihood ratio
        if (like_hood_v / like_hood_ui)**beta < np.random.uniform(0, 1):
            resampled_particles[i] = v  # Update the parameter with the new value
    return resampled_particles
