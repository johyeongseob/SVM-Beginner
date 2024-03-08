"""
Linear Regression 

Y = W1X + W0 & Error min

"""

from matplotlib import pyplot as plt
import numpy as np

X = np.array([0.8147, 0.9058, 0.127, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649])
Y = np.array([0.8576, 0.9706, 0.2572, 0.8854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595])

ones = np.ones(10)
A = np.column_stack((X, ones))
b = Y.reshape(1, 10).T
W = np.linalg.inv(A.T@A)@A.T@b

W1 = W[0]
W0 = W[1]
print(W)

x = np.linspace(0, 1, 100)
y = 0.8113 * x + 0.1941

plt.plot(x,y, label='y=0.8113*x+0.1941')
plt.plot(X,Y,'o')
plt.legend()
plt.show()



