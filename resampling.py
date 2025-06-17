import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def resample(particles, w_list):
    N = len(particles)
    w = np.array(w_list)
    w /= np.sum(w)  # Ensure weights sum to 1

    cdf = np.cumsum(w)
    particle_resampled = []

    for _ in range(N):
        r = np.random.rand()
        for k in range(N):
            if r < cdf[k]:
                particle_resampled.append(particles[k])
                break

    return particle_resampled