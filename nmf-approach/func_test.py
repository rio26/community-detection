import numpy as np

# t1 = np.random.rand(2,2)
# print(t1)

a = np.asmatrix([[1,2], [3,4]])
b = np.asmatrix([[4,3], [2,1]])


print(type(a), type(b))
print(a*b)
print(np.multiply(a,b))
print(np.dot(a,b))


c = [[1,2],[3,4]]
d = [[4,3],[2,1]]
print(type(c), type(d))
# print(c*d)
print(np.multiply(c,d))
print(np.dot(c,d))