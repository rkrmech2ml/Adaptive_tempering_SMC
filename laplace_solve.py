import numpy as np
import scipy.stats as stats

# math functions
import math 
# THIS IS FOR PLOTTING
import matplotlib.pyplot as plt # side-stepping mpl backend
import warnings
warnings.filterwarnings("ignore")
from IPython.display import HTML
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

# Generate normal distribution data
# mu, sigma = 40, 0.1  # mean and standard deviation
# s = np.random.normal(mu, sigma, 100)
# # Plot the histogram of the data
# plt.hist(s, bins=30, density=True, alpha=0.6, color='g')

# Plot the normal distribution curve
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = stats.norm.pdf(x, mu, sigma)
# plt.plot(x, p, 'k', linewidth=2)

# title = "Fit results: mu = %.2f,  sigma = %.2f" % (mu, sigma)
# plt.title(title)

# plt.show()

# print(s[1:100])

def laplace_solver(N, alpha):
    h=1/N
    x = np.arange(0, 1.00001, h)
    #print((x))
    y = np.arange(0, 1.00001, h)
    X, Y = np.meshgrid(x, y)
    # plt.figure()
    # plt.plot(X, Y, marker='.', color='blue', linestyle='none')
    # plt.title('Grid in XY Plane')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()


    nx, ny = N-1, N-1
    N1  = nx*ny
    main_diag = np.ones(N1)*4.0
    side_diag = np.ones(N1-1)*-1
    side_diag[N-2::N-1] = 0
# This is a slicing operation. The syntax for slicing is start:stop:step. Here:

# start = N-2: The slicing starts at the index N-2.
# stop = None: No explicit stop is provided, so it continues to the end of the array.
# step = N-1: The slicing moves in steps of N-1.
# This means the code selects every (N-1)-th element in the array, starting from the (N-2) index.

    up_down_diag = np.ones(N1-3)*-1
    diagonals = [main_diag,side_diag,side_diag,up_down_diag,up_down_diag]
    A = diags(diagonals, [0, -1, 1,nx,-nx], format="csr")
    #print(A.toarray())
   

    # Boundary     
   
    def bc():
        b= np.zeros((N+1, N+1))
        b[0, :] = 0  # Bottom Boundary
        b[N, :] = 0  # Top Boundary
        b[:, 0] = 0  # Left Boundary
        b[:, N] = 0  # Right Boundary
          # bottom Boundary N-1 and N-2 is because we need to skip two rows to reach top boundary  matrix A
        return b
    b=bc()
    # Insert b into u with matching dimension

    # def f():
        
    #     #source = np.sin(np.linspace(0.05, 2*np.pi, ((N-2)*(N-2))))+70
    #     r = np.zeros((N-2, N-2))
    #     for i in range(N-2):
    #         for j in range(N-2):
    #             r[i, j] = h**2 * 1

    #     fig = plt.figure(figsize=(10, 6))
    #     ax = fig.add_subplot(111, projection='3d')
    #     X, Y = np.meshgrid(x[1:N-1], y[1:N-1])
    #     ax.plot_surface(X, Y, r, cmap='viridis')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('r')
    #     ax.set_title('3D Plot of r values')
    #     plt.show()
    #     print("source term is ")
    #     print(r.flatten())

    #     return r.flatten()  # Flatten the array to match the shape of A
    def f(alpha):
        #source = np.sin(np.linspace(0.05, 2*np.pi, ((N-2)*(N-2))))+70
        r = np.zeros((N-1, N-1))
        for i in range(0, N-1):
            for j in range(0, N-1):
                if X[i, j] < 0.45:
                    r[i, j] = h**2 * alpha
                else:
                    r[i, j] = 0

        # Plot the r values
        # fig = plt.figure(figsize=(10, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # X_r, Y_r = np.meshgrid(x[1:N], y[1:N])
        # ax.plot_surface(X_r, Y_r, r, cmap='viridis')
        # ax.set_title('3D Plot of r values')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('r')
        # plt.show()
        # print(r)
        return r.flatten() # Flatten the array to match the shape of A

    def laplace_solve(alpha):
            u = spsolve(A, f(alpha))
            b[1:N,1:N]= np.reshape(u, (N-1, N-1))
            U_solution =b
       
            #print(U_solution[1:N, 1:N])

            # Create a new meshgrid for plotting

            # Plot the solution
            
            # 3D plot of the solution
            # 3D surface plot
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # surf = ax.plot_surface(X, Y, U_solution, cmap='viridis')
            # ax.set_title('3D Plot of the Solution to Laplace Equation')
            # ax.set_xlabel('X-coordinate')
            # ax.set_ylabel('Y-coordinate')
            # ax.set_zlabel('u(x,y)')
            # plt.show()

            # # 2D contour plot
            # plt.figure()
            # plt.contourf(X, Y, U_solution, cmap='viridis')
            # plt.colorbar(label='u(x,y)')
            # plt.title('2D Contour Plot of the Solution to Laplace Equation')
            # plt.xlabel('X-coordinate')
            # plt.ylabel('Y-coordinate')
            # plt.show()
            return U_solution
    
    result=laplace_solve(alpha)
    return result,h



