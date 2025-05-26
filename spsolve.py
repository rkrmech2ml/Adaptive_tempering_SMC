import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve
A = csc_array([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
B = csc_array([[2, 0], [-1, 0], [2, 0]], dtype=float)
print(A, B)
x = spsolve(A, B)
np.allclose(A.dot(x).toarray(), B.toarray())