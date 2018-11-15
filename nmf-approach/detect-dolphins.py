import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from time import time
import numpy.linalg as LA

# magic numbers
_smallnumber = 1E-5

class GNMF:
    """
    Input:
      -- V: m x n matrix, the dataset

    Optional Input/Output:
      -- l: penalty lambda (trade-off parameter between the regularization term and the loss term.)
	
      -- w_init: basis matrix with size m x r
      -- h_init: weight matrix with size r x n  (want r as small as possible)
      -- tol: tolerance error (stopping condition)
      -- timelimit, maxiter: limit of time and maximum iterations (default 1000)
      -- Output: w, h
    """
    def __init__(self, v, l = None, w_init = None, h_init = None, r = None):
        self.v = v
        self.l = l

        if (r is None):
            self.r = 2
        else:
            self.r = r

        if (w_init is None):
            self.w = np.random.rand(self.v.shape[0], self.r)
            # print('init_w: ', self.w.T, 'with size: ', np.shape(self.w.T))
        else:
            self.w = np.matrix(w_init)

        if (h_init is None):
            self.h = np.random.rand(self.r, self.v.shape[1])
            # print('init_h: ', self.h , 'with size: ', np.shape(self.h))
        else:
            self.h = np.matrix(h_init)

    def frobenius_norm(self):
        """ Euclidean error between v and w*h """

        if hasattr(self, 'h') and hasattr(self, 'w'):  # if it has attributes w and h
            error = LA.norm(self.v - np.dot(self.w, self.h))
        else:
            error = None
        return error

    def kl_divergence(self):
        """ KL Divergence between X and W*H """
    
        if hasattr(self, 'h') and hasattr(self, 'w'):
            error = entropy(self.v, np.dot(self.w, self.h)).sum()
        else:
            error = None
        return error

    def mur_solve(self, tol = None, timelimit = None, max_iter = None, r = None):
            """
            Input:
              -- V: m x n matrix, the dataset

            Optional Input/Output:

              -- tol: tolerance error (stopping condition)
              -- timelimit, maxiter: limit of time and maximum iterations (default 1000)
              -- Output: w, h
              -- r: decompose the marix m x n  -->  (m x r) x (r x n), default 2
            """
            if (tol is None):
                self.tol = _smallnumber
            else:
                self.tol = tol

            if (timelimit is None):
                self.timelimit = 3600
            else:
                self.timelimit = timelimit

            if (max_iter is None):
                self.max_iter = 1000
            else:
                self.max_iter = max_iter
                print(self.max_iter)

            # n_iter = 0
            for n_iter in range(self.max_iter):
                self.h = np.multiply(self.h, (np.dot(self.w.T, self.v) /  (np.dot(np.dot(self.w.T, self.w), self.h) )))
                # denominator = np.dot(np.dot(self.w.T, self.w), self.h) + 2 ** -8
                self.w = np.multiply(self.w, (np.dot(self.v, self.h.T) / (np.dot(self.w, np.dot(self.h, self.h.T)) )))
                # denominator = np.dot(self.w, np.dot(self.h, self.h.T)) + 2**-8

            # print('w', np.shape(self.w), 'h', np.shape(self.h)) # w (10304, 2) h (2, 10)
            # return np.dot(self.w, self.h)
            return self.w


# read .gml
G = nx.read_gml('dolphins-v62-e159/dolphins.gml') # 62 vertices

# for adjacency matrix
A = nx.adjacency_matrix(G)   #(62，62)
A_nmf = GNMF(A, r=2)

print("Start running...")
t0 = time()
reult = A_nmf.mur_solve(max_iter=2000)
t1 = time()

print('Final error is: ', A_nmf.frobenius_norm(), 'Time taken: ', t1 - t0)

# # Print adjacency_matrix
# # You may prefer `nx.from_numpy_matrix`.
# G2 = nx.from_scipy_sparse_matrix(A)
# nx.draw_circular(G2, node_size = 20)
# plt.axis('equal')
# plt.show()