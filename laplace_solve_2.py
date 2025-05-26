import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Parameters
nx = 20 # Number of cell interfaces (grid points) in x-axis
ny = 20 # Number of cell interfaces (grid points) in x-axis
ncx = nx-1 # Numbers of cells
ncy = ny-1
Lx = 1.0 # Length of domain in x and y axes
Ly = 1.0
dx = Lx / (nx-1) # Size of a cell in x and y axes
dy = Ly / (ny-1)

# Grid and cell center calculation
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
cell_centers_x = 0.5 * (x[1:] + x[:-1])
cell_centers_y = 0.5 * (y[1:] + y[:-1])
X, Y = np.meshgrid(cell_centers_x,cell_centers_y)

def fn(x,y):
    return 1
    #return x**2+y**2

# Source term
f = np.zeros((ncx, ncy))

# Set values for source term function
for i in range(ncx):
    for j in range(ncy):
        f[i,j] = fn(cell_centers_x[i],cell_centers_y[j])

print(f)

# Discretization coefficients
a_w = dy/dx
a_e = dy/dx
a_n = dx/dy
a_s = dx/dy
a_p = -(a_w + a_s + a_e + a_n)

# Initialize solution
phi = np.zeros((ncx, ncy))

# Set Dirichlet boundary conditions
phi[:, 0] = 0.0
phi[:, -1] = 0.0
phi[0, :] = 0.0
phi[-1, :] = 0.0

# Assemble coefficient matrix A and right-hand side b
A = np.zeros((ncx * ncy, ncx * ncy))
for i in range(ncy):
    for j in range(ncx):
        idx = i * ncx + j
        if i > 0:
            A[idx, idx - ncx] = a_n
        if i < ncy - 1:
            A[idx, idx + ncx] = a_s
        if j > 0:
            A[idx, idx - 1] = a_w
        if j < ncx - 1:
            A[idx, idx + 1] = a_e
        A[idx, idx] = a_p

A=A/(dx*dy)

# Compressed Sparse Row (CSR), used for spsolve (Not necessary, but removes warning)
A=csr_matrix(A)

b = f.flatten()

# Solve the linear system
phi = spsolve(A, b).reshape((ncy,ncx))

print(phi)

# 3D Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, phi, cmap='viridis', edgecolor='k')

# Customize the plot
ax.set_title('Poisson Equation Solution using Finite Volume Method')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Phi')

# Uncomment below code to make fixed ratio for axes (Not Recommended)
# ax.set_box_aspect([np.ptp(coord) for coord in [X, Y, phi]])

# Add a colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.show()