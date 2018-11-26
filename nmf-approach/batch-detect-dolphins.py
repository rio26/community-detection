import numpy as np
import numpy.linalg as LA
import scipy.io as sio # not working for me
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
#from scipy.stats import entropy
from time import time
import random, math

# magic numbers
_smallnumber = 1E-6

class SNMF():

    """
    Input:
      -- V: m x n matrix, the dataset

    Optional Input/Output:
      -- l: penalty lambda (trade-off parameter between the regularization term and the loss term.)
    
      -- w_init: basis matrix with size m x r
      -- h_init: weight matrix with size r x n  (r is the number of cluster)
      -- Output: w, h
    """
    def __init__(self, x, h_init = None, r = 2, batch_number = 10, max_iter = 100):
        self.x = x
        self.r = r
        self.batch_number = batch_number
        self.mini_batch_size = math.ceil(x.shape[0] / batch_number)
        self.max_iter = max_iter
        print("Batch number is : ", batch_number, " with mini_batch_size: ", self.mini_batch_size)
        print("the matrix's row and column are: ", self.x.shape[0], self.x.shape[1])
        self.h = h_init
        self.errors = np.zeros(self.max_iter)

    def frobenius_norm(self):
        """ Euclidean error between x and h * h.T """

        if hasattr(self, 'h'):  # if it has attributes w and h
            error = LA.norm(self.x - np.dot(self.h, self.h.T))
        else:
            error = None
        return error

    def sgd_solver(self):
        if(self.batch_number == 1):        # normal MUR
            for iter in range(self.max_iter):
                self.errors[iter] = np.linalg.norm(self.x - np.dot(self.h, self.h.T), 'fro') # record error

                tmp_x = self.x.todense()
                self.h = np.multiply(self.h, (np.dot(tmp_x, self.h) / (np.dot(np.dot(self.h, self.h.T) ,self.h) + 2 ** -8)))
        else:
            for iter in range(self.max_iter):  # stochastic MUR     
                self.errors[iter] = np.linalg.norm(self.x - np.dot(self.h, self.h.T), 'fro') # record error
                lo = random.randint(0, (self.x.shape[0] - self.mini_batch_size - 1))
                # print("batch index is: ", lo)
                batch_x = self.x[lo: (lo + self.mini_batch_size), lo: (lo + self.mini_batch_size)].todense()  ## correct
                # print("batch matrix:", np.shape(batch_x), "and its element: ", batch_x)
                batch_h = self.h[lo: (lo + self.mini_batch_size), :] ##correct
                # print("batch_h is: ",  np.shape(batch_h), "and its element: ", batch_h) # dolphin case is (7,7)
                

                # self.h = 0.5 * np.multiply(self.h, (1 + (np.dot(self.x.T, self.h) /  (np.dot(np.dot(self.h, self.h.T), self.h) + 2 ** -8))))
                # tmp = np.multiply(batch_h, ((batch_x, np.dot.(np.dot(batch_h, batch_h.T)) /  (np.dot(np.dot(batch_h, batch_h.T), batch_h)))))
                # tmp = np.dot(np.dot(batch_x, batch_h), batch_h.T) / np.dot(np.dot(np.dot(batch_h, batch_h.T) ,batch_h) ,batch_h.T)
                self.h[lo: (lo + self.mini_batch_size), :] = np.multiply(batch_h, (np.dot(batch_x, batch_h) / (np.dot(np.dot(batch_h, batch_h.T) ,batch_h) + 2 ** -8)))
                # print("Test Tmp shape: ", np.shape(tmp), "and its value: \n", tmp)
 
        # print("updated h, with size: ", np.shape(tmp), "; type: ", type(tmp))
        # print("updated h's element: ", np.shape(tmp[0][0]), "; type: ", type(tmp[0][0]), "value: ", tmp[0][0])
        # # xx = self.x.todense()
        # for c_row in range(xx.shape[0]):
        #     for c_col in range(xx.shape[1]):
        #         if xx[c_row][c_col] < 0:
        #             num_viol += 1
        # print(num_viol)
        return self.h

    def get_error_trend(self):
        return self.errors

# read .gml
G = nx.read_gml('dolphins-v62-e159/dolphins.gml') # 62 vertices
cluster_num = 2


A = nx.adjacency_matrix(G)   #(62，62)                              # get adjacency matrix
initial_h = np.random.rand(A.shape[0], cluster_num)                 # h's initialization
grid1 = np.dot(initial_h,initial_h.T)                               # x's initial value
A_nmf = SNMF(A, r=cluster_num,  h_init = initial_h, batch_number=1, max_iter=50) # call snmf's constructor



print("Staring error is: ", A_nmf.frobenius_norm())
print("Start running...")
t0 = time()
result = A_nmf.sgd_solver()                              # run gd, return h
t1 = time()

print('Final error is: ', A_nmf.frobenius_norm(), 'Time taken: ', t1 - t0)

plt.plot(A_nmf.get_error_trend())
plt.show()

"""
----------------------------------------PLOT----------------------------------------
"""

grid2 = np.dot(result,result.T)     # result matrix

fig = plt.figure()

# subplot 1, initial matrix
ax = fig.add_subplot(121)
im = ax.imshow(grid1)
plt.colorbar(im)

# subplot 2, color map
ax = fig.add_subplot(122)
im = ax.imshow(grid2)
plt.colorbar(im)

plt.tight_layout()
plt.show()