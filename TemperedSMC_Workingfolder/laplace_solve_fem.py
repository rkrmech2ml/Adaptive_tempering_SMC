import numpy as np
import matplotlib.pyplot as plt
from skfem import *
from skfem.models.poisson import laplace, unit_load

# Define mesh and basis
mesh = MeshTri.init_sqsymmetric().refined(3)  # refine for resolution
basis = Basis(mesh, ElementTriP1())

# Define alpha
alpha = 1.0  # Example value for alpha

# Source function: step in x < 0.45
@BilinearForm
def a(u, v, w):
    return np.dot(w['grad'].u, w['grad'].v)  # Corrected gradient access

@LinearForm
def l(v, w):
    x = w.x[0]
    return (x < 0.45) * alpha * v

# Assemble system
A = asm(a, basis)
b = asm(l, basis)

# Apply Dirichlet BC
D = basis.get_dofs()
A, b = enforce(A, b, D.all(), 0.0)

# Solve
u = solve(A, b)

# Plot solution
from skfem.visuals.matplotlib import plot
plot(basis, u, shading='gouraud')
plt.colorbar(label='u(x, y)')
plt.title('FEM Solution to Poisson Equation with Step Source')
plt.show()