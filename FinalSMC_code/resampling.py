import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def resample(particles, w_list, visualize=True):
    N = len(particles)
    w = np.array(w_list)  
    cdf = np.cumsum(w)
    particle_resampled = []

    for _ in range(N):
        r = np.random.rand()
        idx = np.searchsorted(cdf, r)
        particle_resampled.append(particles[idx])

    particle_resampled = np.array(particle_resampled)

    # if visualize:
    #     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # #     # --- Before Resampling ---
    #     print(particles[:,0].shape, w.shape)
    #     axes[0].vlines(particles[:,0], 0, w, colors="blue", lw=2, alpha=0.7)
    #     axes[0].set_title("Before Resampling (weights)")
    #     axes[0].set_xlabel("Particle")
    #     axes[0].set_ylabel("Weight")
    #     axes[0].set_ylim(min(w), max(w))
    #     axes[0].grid(True)

    #     # --- After Resampling ---
    #     # Count how many times each particle was picked

    #     axes[1].vlines(particle_resampled[:,0], 0, w, colors="orange", lw=2, alpha=0.7)
    #     axes[1].set_title("After Resampling")
    #     axes[1].set_xlabel("Particle")
    #     axes[1].set_ylabel("weight")
    #     axes[1].set_ylim(min(w), max(w))
    #     axes[1].grid(True)

    #     plt.tight_layout()
    #     plt.show()

    return particle_resampled



