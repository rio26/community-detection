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
        self.x = x.todense()
        self.r = r
        self.max_iter = max_iter
        
        print("Constructor call: The matrix's row and column are: ", self.x.shape[0], self.x.shape[1], "total iteration: ", self.max_iter)
        
        self.batch_number = batch_number
        self.batch_number_range = self.x.shape[0]
        self.mini_batch_size = math.ceil(x.shape[0] / self.batch_number)
        self.batch_x = np.asmatrix(np.zeros((self.mini_batch_size,self.mini_batch_size)))
        print("Constructor call: Batch number is : ", batch_number, " with mini_batch_size: ", self.mini_batch_size, "batch_x has shape:", self.batch_x.shape)

        self.h = h_init
        self.errors = np.zeros(self.max_iter)

    def frobenius_norm(self):
        """ Euclidean error between x and h * h.T """

        if hasattr(self, 'h'):  # if it has attributes w and h
            error = LA.norm(self.x - self.h*self.h.T)
        else:
            error = None
        return error

    def bgd_solver(self, eps = 0.001, debug = None):
        if(self.batch_number == 1):        # normal MUR
            for iter in range(self.max_iter):
                self.errors[iter] = LA.norm(self.x - self.h * self.h.T)

                numerator = self.x*self.h
                denominator = (((self.h*self.h.T)*self.h) + 2 ** -8)
                self.h = np.multiply(self.h, np.divide(numerator, denominator))

                # count = 0
                # for i in range(self.h.shape[0]):
                #     for j in range(self.h.shape[1]):
                #         # print(self.h[i,j])
                #         if self.h[i,j]<0:
                #             count += 1
                #             print("(", i, ",", j, ")")
                # print("negative numbers:", count)

        else:
            batch_h = np.asmatrix(np.zeros((self.mini_batch_size,self.r)))
            for iter in range(self.max_iter):  # stochastic MUR     
                self.errors[iter] = np.linalg.norm(self.x - self.h * self.h.T, 'fro') # record error
                if self.errors[iter] < eps:
                    print("stop condition met!")
                    return self.h

                tmp_list = self.generate_random_numbers(upper_bound = self.batch_number_range, num = self.mini_batch_size)
                # print("tmp_list: ", tmp_list, "type: ", type(tmp_list), "length: ", len(tmp_list))
                
                
                # an ugly matrix to create batch matrix
                i = 0
                while i < len(tmp_list):
                    j = i
                    batch_h[i,:] = self.h[tmp_list[i],:]
                    while j < len(tmp_list):
                        self.batch_x[i,j] = self.x[tmp_list[i],tmp_list[j]]
                        self.batch_x[j,i] = self.x[tmp_list[i],tmp_list[j]]
                        j += 1
                    i += 1
                # print("batch x:", self.batch_x, "size", self.batch_x.shape, "type:", type(self.batch_x))
                # print("batch h:", batch_h, "size", batch_h.shape, "type:", type(batch_h))

                numerator = self.batch_x * batch_h
                denominator = (((batch_h*batch_h.T)*batch_h) + 2 ** -8)
                update = np.multiply(batch_h, np.divide(numerator, denominator))
                # print("line 99, update type: ", type(update), "shape:", update.shape, "update[i,:]", update[1,:])
                i = 0
                while i < len(tmp_list):
                    self.h[tmp_list[i],:] = update[i,:]   
                    i += 1

                # lo = random.randint(0, (self.x.shape[0] - self.mini_batch_size - 1))
                # # print("batch index is: ", lo)
                # batch_x = self.x[lo: (lo + self.mini_batch_size), lo: (lo + self.mini_batch_size)].todense()  ## correct
                # batch_h = self.h[lo: (lo + self.mini_batch_size), :] ##correct
                
                # batch_h = 

                # self.h[lo: (lo + self.mini_batch_size), :] = np.multiply(batch_h, ((batch_x * batch_h) / (((batch_h* batch_h.T)*batch_h) + 2 ** -8)))
                # print("Test Tmp shape: ", np.shape(tmp), "and its value: \n", tmp)
 
        return self.h
 
    def get_error_trend(self):
        return self.errors

    # generate a list of random number from range [0, range], with size num
    def generate_random_numbers(self, upper_bound, num):
        seq = list(range(0,upper_bound))
        return random.sample(seq,num)

# read .gml
G = nx.read_gml('dolphins-v62-e159/dolphins.gml') # 62 vertices
cluster_num = 2


A = nx.adjacency_matrix(G)   #(62，62)                                           # get adjacency matrix
# print("line 126, type of dense A:" , type(A.todense()))                        # <class 'scipy.sparse.csr.csr_matrix'>,  
                                                                                 # to dense: <class 'numpy.matrixlib.defmatrix.matrix'>
initial_h = np.asmatrix(np.random.rand(A.shape[0], cluster_num))                 # h's initialization, as a matrix
# initial_h = np.asmatrix(np.ones((A.shape[0], cluster_num)))
print(type(initial_h))
# print("line 127, type of initial_h" , type(initial_h), "shape: ", initial_h.shape)
# print("initial h [1]::::: ",initial_h[0,0])

grid1 = A.todense()                                                 # initial x
grid2 = np.dot(initial_h,initial_h.T)                               # initial h
A_nmf = SNMF(x=A, r=cluster_num,  h_init = initial_h, batch_number=2, max_iter=1) # call snmf's constructor



print("Staring error is: ", A_nmf.frobenius_norm())
print("Start running...")
t0 = time()
result = A_nmf.bgd_solver()                              # run gd, return h
t1 = time()

# print(result[0,0])
print('Final error is: ', A_nmf.frobenius_norm(), 'Time taken: ', t1 - t0)

dolphins = sio.loadmat('dolphins-v62-e159/dolphins_rlabels')
label = dolphins['labels'].T

correct_count = 0
for i in range(result.shape[0]):
    if result[i,0] < result[i,1]:
        # print(label[i], "-- 1")
        if label[i] == 1:
            correct_count += 1
    else:
        # print(label[i],  "-- 2")
        if label[i] == 2:
            correct_count += 1

print("correct_count: ", correct_count)

"""
----------------------------------------PLOT----------------------------------------
"""


plt.plot(A_nmf.get_error_trend())
plt.show()


grid3 = np.dot(result,result.T)     # result matrix

fig = plt.figure()

# subplot 1, initial matrix
ax = fig.add_subplot(131)
im = ax.imshow(grid1)
plt.colorbar(im)

# subplot 2, color map
ax = fig.add_subplot(132)
im = ax.imshow(grid2)
plt.colorbar(im)

ax = fig.add_subplot(133)
im = ax.imshow(grid3)
plt.colorbar(im)


plt.tight_layout()
plt.show()