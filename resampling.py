import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def resample(particle_init, mu_list):

    particle_resampled = []
    for k in range(len(mu_list)):
        r = np.random.uniform(0, 1)
        if r < np.cumsum(mu_list)[k]:
           #print("Selected particle: ", particle_init[k])
            if k < len(particle_init):
                particle_resampled.append(particle_init[k])
            else:
                raise IndexError(f"Index {k} is out of bounds for particle_init with size {len(particle_init)}")
   #print("Resampled particle: ", particle_resampled)
    return particle_resampled