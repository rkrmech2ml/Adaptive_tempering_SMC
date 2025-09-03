# LIBRARY
# vector manipulation
import numpy as np
# math functions
import math 
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

# THIS IS FOR PLOTTING
#matplotlib inline
import matplotlib.pyplot as plt # side-stepping mpl backend
import warnings
warnings.filterwarnings("ignore")
from IPython.display import HTML
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
N=5
h=1/N
x=np.arange(0,1.0001,h)
y=np.arange(0,1.0001,h)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
plt.plot(x[1],y[1],'ro',label='unknown');
plt.plot(X,Y,'ro');
plt.plot(np.ones(N+1),y,'go',label='Boundary Condition');
plt.plot(np.zeros(N+1),y,'go');
plt.plot(x,np.zeros(N+1),'go');
plt.plot(x, np.ones(N+1),'go');
plt.xlim((-0.1,1.1))
plt.ylim((-0.1,1.1))
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r'Discrete Grid $\Omega_h,$ h= %s'%(h),fontsize=24,y=1.08)
plt.show()
w=np.zeros((N+1,N+1))

for i in range (0,N):
        w[i,0]=0 #left Boundary
        w[i,N]=0#Right Boundary

for j in range (0,N):
        w[0,j]=0 #Lower Boundary
        w[N,j]=0 #Upper Boundary

        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot a basic wireframe.
ax.plot_wireframe(X, Y, w,color='r', rstride=10, cstride=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('w')
plt.title(r'Boundary Values',fontsize=24,y=1.08)
plt.show()
N2=(N-1)*(N-1)
A=np.zeros((N2,N2))
# nx, ny = N-1, N-1
# N1  = nx*ny
# main_diag = np.ones(N1)*-4.0
# side_diag = np.ones(N1-1)
# up_down_diag = np.ones(N1-3)
# diagonals = [main_diag,side_diag,side_diag,up_down_diag,up_down_diag]
# A = diags(diagonals, [0, -1, 1,nx,-nx], format="csr")
# print(A.toarray())
# ## Diagonal            
for i in range (0,N-1):
    for j in range (0,N-1):           
        A[i+(N-1)*j,i+(N-1)*j]=-4

# LOWER DIAGONAL        
for i in range (1,N-1):
    for j in range (0,N-1):           
        A[i+(N-1)*j,i+(N-1)*j-1]=1   
# UPPPER DIAGONAL        
for i in range (0,N-2):
    for j in range (0,N-1):           
        A[i+(N-1)*j,i+(N-1)*j+1]=1   

# LOWER IDENTITY MATRIX
for i in range (0,N-1):
    for j in range (1,N-1):           
        A[i+(N-1)*j,i+(N-1)*(j-1)]=1        
        
        
# UPPER IDENTITY MATRIX
for i in range (0,N-1):
    for j in range (0,N-2):           
        A[i+(N-1)*j,i+(N-1)*(j+1)]=1

print("Matrix A is ")
print(A)
  
# fig = plt.figure(figsize=(12,4));
# plt.subplot(121)
# plt.imshow(A,interpolation='none');
# clb=plt.colorbar();
# clb.set_label('Matrix elements values');
# plt.title('Matrix A ',fontsize=24)
# # plt.subplot(122)
# # plt.imshow(Ainv,interpolation='none');
# # clb=plt.colorbar();
# # clb.set_label('Matrix elements values');
# # plt.title(r'Matrix $A^{-1}$ ',fontsize=24)

# fig.tight_layout()
# plt.show()

r=np.zeros(N2)

# vector r      
for i in range (0,N-1):
    for j in range (0,N-1):
        if x[i]<0.45:
            r[i+(N-1)*j]=1
        else:           
            r[i+(N-1)*j]=1



# Boundary        
b_bottom_top=np.zeros(N2)
for i in range (0,N-1):
    b_bottom_top[i]=0 #Bottom Boundary
    b_bottom_top[i+(N-1)*(N-2)]=0# Top Boundary
      
b_left_right=np.zeros(N2)
for j in range (0,N-1):
    b_left_right[(N-1)*j]=0 # Left Boundary
    b_left_right[N-2+(N-1)*j]=0# Right Boundary
    
b=b_left_right+b_bottom_top
print("source term is ")
print(r)


u=spsolve(A, (r-b))
w[1:N,1:N]= np.reshape(u, (N-1, N-1))
print("Solution is ")
print(w[1:N,1:N])

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d');
# Plot a basic wireframe.
ax.plot_surface(X, Y, w, cmap='viridis', edgecolor='none');
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('w');
plt.title(r'Numerical Approximation of the Poisson Equation',fontsize=24,y=1.08);
plt.show();

plt.figure()
plt.contourf(X, Y, w, cmap='viridis')
plt.colorbar(label='u(x,y)')
plt.title('2D Contour Plot of the Solution to Laplace Equation')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.show()
