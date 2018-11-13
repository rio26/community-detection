import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt

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
    def __init__(self, v, l, w_init = None, h_init = None, r = None):
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

# read .gml
G = nx.read_gml('dolphins-v62-e159/dolphins.gml')
# print(len(G.node)) 62 vertices

# for adjacency matrix
A = nx.adjacency_matrix(G)


# # Print adjacency_matrix
# # You may prefer `nx.from_numpy_matrix`.
# G2 = nx.from_scipy_sparse_matrix(A)
# nx.draw_circular(G2, node_size = 20)
# plt.axis('equal')
# plt.show()