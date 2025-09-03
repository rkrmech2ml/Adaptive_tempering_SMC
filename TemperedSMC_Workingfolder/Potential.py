import numpy as np
from laplace_solve import laplace_solver

def compute_potentials(n, particles, experiment, h):
    """Calculate potential  for each particle."""
    potentials = []
    for p in particles:
        #print(f"Particle: {p}")
        sim_output, _ = laplace_solver(n, p)
        l2_error = h**2 * np.sum((sim_output - experiment)**2) / 2
        potentials.append(l2_error)
    return potentials