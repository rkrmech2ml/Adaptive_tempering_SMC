from laplace_solve import laplace_solver
from resampling import resample
from markov_kernal import markov_kernel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

##################
# note this is different version of IS-SMC code with weight multiplied to the particle distribution.


n=10
particle=1.0 # we need to find the value of the particle or a distribution of this particle (considering this is our unknown parameter u)
experiment,h = laplace_solver(n, particle) #this will be our desired / actual value from the experiment (assume)
#print("Experiment value: ", experiment,h)

#first we start with an initial distribution of the particle varying from -10 to 10 under uniform distribution

particle_init = np.random.uniform(-5, 5, 500) #initial guess of the particle
#print("Initial guess of the particle: ", particle_init)


#starting beta from 0.0 to 1.0 with 0.1 step size
beta = np.arange(0.0, 1.1, 0.1)
print("Beta values: ", beta)



#######step1&2#######
for m in range (1,len(beta)):

    phi=[] #phi = 1/2||desired-simualtion(u)||^2 in L^2 norm.
    error=[] 
    normzng_factor = 0
    mu_list=[]#ä mu list is the weight according to the potential phi

    for p in particle_init:
        err_i=(((laplace_solver(n, p)[0] - experiment)**2))
    #print("Error: ", np.sum(err_i))
        phi_i=np.exp(-beta[m] * ( h**2*(np.sum(err_i )/2)))
        phi.append(phi_i)
        normzng_factor += phi_i
    for k in range(0, len(phi)):
        mu_list.append(phi[k] / normzng_factor)
    #print("Mu list: ", mu_list)



    # print("Phi values: ", phi)
    # print("Phi values shape: ", np.shape(phi))
    # print("Normalization factor: ", normzng_factor)
    # print("Mu values: ", mu_list)

    # plt.figure(figsize=(10, 6))
    # plt.hist(particle_init, bins=50, alpha=0.6, color='b')
    # plt.xlabel('Mu values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram initial particle values')
    # plt.show()



    #######step3#######
    particle_init = np.multiply(mu_list, particle_init)
    particle_resampled= resample(particle_init, mu_list)
    #equalizing the weights
    mu_list = [1/len(mu_list)] * len(mu_list)
    #print("reweight : ", mu_list)


    # Plot the resampled particle in a distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(particle_resampled, bins=50, alpha=0.6, color='g')
    # plt.xlabel('Resampled Particle')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Resampled Particle')
    # plt.show()


    #now  we have to wiggle the parameter particle to left and right
    #markov kernel to accept the new value of the data



    #######step4#######
    wiggled_particle = markov_kernel(particle_init, particle_resampled, experiment, n,h, beta)



    # Print the updated particle_resampled values
    # Print the updated particle_resampled values
    #print("Updated Particle Resampled: ", particle_resampled)




    # Plot the updated particle_resampled values
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot the resampled particle distribution
    axs[0].hist(particle_resampled, bins=50, alpha=0.6, color='orange')
    axs[0].set_xlabel('Particle Resampled')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(f'Distribution of Updated Particle Resampled beta={beta[m]}')

    # Save the plot for each beta value
    

    # Plot the initial particle distribution
    axs[1].hist(particle_init, bins=50, alpha=0.6, color='green')
    axs[1].set_xlabel('Initial Particle')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Distribution of Initial Particle')
    plt.savefig(f'particle_resampled_beta_{beta[m]:.1f}.png')

    # plt.tight_layout()
    # plt.show()













