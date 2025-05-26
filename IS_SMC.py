from laplace_solve import laplace_solver
from resampling import resample
from markov_kernal import markov_kernel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
n=10
particle=1.0 # we need to find the value of the particle or a distribution of this particle (considering this is our unknown parameter u)
experiment_exact,h = laplace_solver(n, particle) #this will be our desired / actual value from the experiment (assume)
#---------------
#add noise to the experiment
#---------------

experiment_total=0
for _ in range(5):
    # adding noise to the experiment

    noise=(np.random.normal(0, 0.1, size=experiment_exact.shape))
    experiment_total+=noise
    experiment=experiment_exact+experiment_total
    
    #---------------
experiment=experiment/5 #mean of the recorded values





particle_init = np.random.uniform(low=-6.0, high=6.0, size=1000) # initial guess of the particle using uniform distribution

#print("Initial guess of the particle: ", particle_init)


#starting beta from 0.0 to 1.0 with 0.1 step size
#beta = np.arange(0.0, 1.1, 0.1)




#--------------
# step1&2
# --------------
m = 1
beta = 0.0

while beta < 1.0:
    # --------------
    # adaptive tempering
    # --------------
    beta = np.exp(0.7 * m / 10) - 1
    beta = min(beta, 1.0)
    print("Beta value: ", beta)
    phi = []  # phi = 1/2||desired-simualtion(u)||^2 in L^2 norm.
    error = []
    normzng_factor = 0
    mu_list = []  # mu list is the weight according to the potential phi

    for p in particle_init:
        phi_i = np.exp(-beta * (h ** 2 * (np.sum(((laplace_solver(n, p)[0] - experiment) ** 2)) / 2)))
        phi.append(phi_i)
        normzng_factor += phi_i
    for k in range(0, len(phi)):
        mu_list.append(phi[k] / normzng_factor)

    # --------------
    # step3
    # --------------
    particle_resampled = resample(particle_init, mu_list)
    # -------
    # equalizing the weights
    # -------
    mu_list = [1 / len(mu_list)] * len(mu_list)

    # --------------
    # step4
    # --------------
    wiggled_particle = markov_kernel(particle_init, particle_resampled, experiment, n, h, beta)

    # Plot the updated particle_resampled values
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot the resampled particle distribution
    axs[0].hist(particle_resampled, bins=50, density=True, alpha=0.6, color='orange')
    axs[0].set_xlabel('Particle Resampled')
    axs[0].set_ylabel('Probability Density')
    axs[0].set_title(f'PDF of Resampled Particle (beta={beta:.3f})')

    # Plot the probability distribution function (PDF) of the initial particle distribution
    axs[1].hist(particle_init, bins=50, density=True, alpha=0.6, color='green')
    axs[1].set_xlabel('Initial Particle')
    axs[1].set_ylabel('Probability Density')
    axs[1].set_title('PDF of Initial Particle')
    plt.savefig(f'particle_resampled_pdf_beta_{beta:.3f}.png')

    m += 1












