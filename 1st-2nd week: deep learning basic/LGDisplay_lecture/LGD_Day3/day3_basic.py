"""

Ridge Linear Regression 
w = argmin ||wx-y||^2 + r||w||^2
r = lamda(regularization parameter) 1st~10th

"""

from matplotlib import pyplot as plt
import numpy as np

X = np.array([0.8147, 0.9058, 0.127, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649])
Y = np.array([0.8576, 0.9706, 0.2572, 0.8854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595])

ones = np.ones(10)
A = np.append(X,ones).reshape(2, 10).T
b = Y.reshape(1, 10).T
r =  np.array([100, 10, 1, 0.1, 0.01, 0.001, 0]) # lamda(regularization parameter)
E = np.eye(A.shape[1])

W = []
for i in r:
    w = np.linalg.inv(A.T @ A + i * E) @ A.T @ b
    W.append(w)

W = np.array(W).reshape(r.shape[0], 2)
W1 = W[:, 0]
W0 = W[:, 1]


for i in range(r.shape[0]): 
    x = np.linspace(0,1,100)
    y = W1[i]*x+W0[i]
    plt.plot(x, y, label='lamda = {}'.format(r[i]))
    y=[]

plt.plot(X, Y, 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()