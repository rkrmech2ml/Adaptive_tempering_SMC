import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def resample(particles, w_list):
    N = len(particles)
    w = np.array(w_list)


    cdf = np.cumsum(w)
    particle_resampled = []

    for _ in range(N):
        r = np.random.rand()
        idx = np.searchsorted(cdf, r)
        particle_resampled.append(particles[idx])

    return particle_resampled