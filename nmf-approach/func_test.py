import numpy as np
import scipy.io as sio
import random
i = 5
tmp = random.randint(0, i-1)
print(tmp)



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