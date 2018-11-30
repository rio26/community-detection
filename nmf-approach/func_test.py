import numpy as np
import scipy.io as sio
import random
import networkx as nx
import matplotlib.pyplot as plt
from time import time

G1 = nx.Graph()
# print(type(G))

t0 = time()
with open('email-v1005-e25571-c42/email-Eu-core.txt','r') as f:
    for line in f:
        line=line.split()#split the line up into a list - the first entry will be the node, the others his friends
        # print(len(line), "cont", line[0])
        if len(line)==1:#in case the node has no friends, we should still add him to the network
            if line[0] not in G1:
                nx.add_node(line[0])
        else:#in case the node has friends, loop over all the entries in the list
            focal_node = line[0]#pick your node
            for friend in line[1:]:#loop over the friends
                G1.add_edge(focal_node,friend)#add each edge to the graph
t1 = time()
# nx.write_gml(G, 'email-v1005-e25571-c42/email-Eu-core.gml')

t2 = time()
G2 = nx.read_gml('email-v1005-e25571-c42/email-Eu-core.gml')
t3 = time()

print("import from txt:", t1-t0, "\n import from gml", t3-t2)



# t4 = time()
# print(nx.is_isomorphic(G1, G2))
# t5 = time()
# print("check similarity:", t5 - t4)
# # nx.draw_networkx(G)
# plt.show()
"""
random int
"""

# i = 5
# tmp = random.randint(0, i-1)
# print(tmp)


"""
matrix manipulation
"""
# matrix = np.asmatrix([[1,-2], [-3,4]])

# i = 0
# count = 0
# while i < matrix.shape[0]:
#     j = 0
#     while j < matrix.shape[1]:
#         if matrix[i,j] < 0:
#             matrix[i,j] = 0
#             count += 1
#         j += 1
#     i += 1
# print(count)
# print(matrix, type(matrix))

# print(2**-8)

# a = np.asmatrix(np.zeros((2,2)))
# hi = 5 #(not include this number)
# seq = list(range(0,hi))
# a = random.sample(seq,3)
# print("a: ", a, "type: ", type(a), "\n a's 1,2,3 values: ", a[0], a[1], a[2])
# print("a's value check: ", a[0,0])


# t1 = np.random.rand(2,2)
# print(t1)

# a = np.asmatrix([[1,2], [3,4]])
# b = np.asmatrix([[4,3], [2,1]])


# print(type(a), type(b))
# print(a*b)
# print(np.multiply(a,b))
# print(np.dot(a,b))


# c = [[1,2],[3,4]]
# d = [[4,3],[2,1]]
# print(type(c), type(d))
# # print(c*d)
# print(np.multiply(c,d))
# print(np.dot(c,d))

# dolphin = sio.loadmat('dolphins-v62-e159/dolphins_rlabels')
# label = dolphin['labels'].T
# print(label.shape)

# print(label[1])