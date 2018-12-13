import numpy as np
import numpy.linalg as LA
import scipy.io as sio # not working for me
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
#from scipy.stats import entropy
from time import time
import random, math

# magic numbers
_smallnumber = 1E-6

class SNMF():

    # def __init__(self, V, K, alpha, beta, iterations):
    #     """
    #     Perform matrix factorization to predict empty
    #     entries in a matrix.

    #     Arguments
    #     - V (ndarray)   : user-item rating matrix
    #     - K (int)       : number of latent dimensions
    #     - alpha (float) : learning rate
    #     - beta (float)  : regularization parameter
    #     """

    #     self.V = V
    #     self.num_users, self.num_items = V.shape
    #     self.K = K
    #     self.alpha = alpha
    #     self.beta = beta
    #     self.iterations = iterations
    """
        x: adjancy matrix (symmetric) with dimension (nxn)
        h: result matrix with dimention (nxr)
        r: number of clusters
    """
    def __init__(self, x, h_init = None, r = 2, batch_number = 10):
        self.x = x
        self.r = r
        self.mini_batch_size = math.ceil(x.shape[0] / 10)
        print("the matrix's row and column are: ", self.x.shape[0], self.x.shape[1])

        if (h_init is None):
            self.h = np.random.rand(self.x.shape[0], self.r)
            # print('init_h: ', self.h , 'with size: ', np.shape(self.h))
        else:
            self.h = np.matrix(h_init)


    def frobenius_norm(self):
        """ Euclidean error between x and h * h.T """

        if hasattr(self, 'h'):  # if it has attributes w and h
            error = LA.norm(self.x - np.dot(self.h, self.h.T))
        else:
            error = None
        return error

    def sgd_solver(self, alpha = 0.002, beta = 0.02, max_iter = 100):
        for iter in range(max_iter):
            lo = random.randint(0, (self.x.shape[0] - self.mini_batch_size - 1))
            print("batch index is: ", lo)
            batch_matrix = self.x[lo: (lo + self.mini_batch_size), lo: (lo + self.mini_batch_size)]
            # print("batch_matrix is: \n", batch_matrix.shape) # dolphin case is (7,7)
            



    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.V != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.V[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.V[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.V.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.V[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

# read .gml
print("Reading Data...")
G = nx.read_gml('polical-blogs-v1490-e19090/polblogs.gml') #  vertices

# for adjacency matrix
print("Converting to adjacency matrix...")
A = nx.adjacency_matrix(G)   #(62，62)

A_nmf = SNMF(A, r=2, batch_number=5)
print("A has shape :", A.shape) #(62, 62)

print("Staring error is: ", A_nmf.frobenius_norm())
print("Start running...")
t0 = time()
reult = A_nmf.sgd_solver(max_iter=5)
t1 = time()

# print('Final error is: ', A_nmf.frobenius_norm(), 'Time taken: ', t1 - t0)