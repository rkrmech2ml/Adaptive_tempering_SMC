from laplace_solve import laplace_solver
from resampling import resample
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm




def markov_kernel(particles,particles_initial, resampled_particles, experiment, n, h, beta):
    # print("length of particles:", len(particles))
    # print("length of resampled particles:", len(resampled_particles))

    gamma = 0.05
    particles = np.array(particles)
    resampled_particles = np.array(resampled_particles)
    spread1 = np.max(resampled_particles[:, 0]) - np.min(resampled_particles[:, 0])
    #print(f"Spread in first dimension: {spread1}")
    spread2 = np.max(resampled_particles[:, 1]) - np.min(resampled_particles[:, 1])
    #print(f"Spread in second dimension: {spread2}")
    i=0
    for i in range(len(resampled_particles)):

        v1, v2 = resampled_particles[i]
        #print(f"Current particle: {v1}, {v2}")

        # Generate a new candidate value for particle using the Markov kernel or random walk
        new_v1 = np.sqrt(1 - gamma**2) * v1 + gamma * spread1 * np.random.uniform(-1, 1)
        new_v2 = np.sqrt(1 - gamma**2) * v2 + gamma * spread2 * np.random.uniform(-1, 1)


        
        # Calculate likelihoods for the new candidate and the current value
        like_hood_v = np.exp( -(h**2 * np.sum((laplace_solver(n, [new_v1,new_v2])[0] - experiment)**2) / 2))
    
        like_hood_ui = np.exp(-(h**2 * np.sum((laplace_solver(n, resampled_particles[i])[0] - experiment)**2) / 2))
 
        
        # Accept or reject the new candidate based on the likelihood ratio
        if (like_hood_v / like_hood_ui)**beta > np.random.uniform(0, 1):
            resampled_particles[i] = [new_v1,new_v2] # Update the parameter with the new value
            
    return resampled_particles
