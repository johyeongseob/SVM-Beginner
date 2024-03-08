"""

Support vector machine

min_w,b||w|| s.t.(wx_j+b)y_j >=1, all j

"""

from sklearn import svm
from matplotlib import pyplot as plt
import numpy as np

X = np.array([0.8147, 0.9058, 0.127, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649])
Y = np.array([0.8576, 0.9706, 0.2572, 0.8854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595])
Class = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])

# Reshape X and Y into a 2D array, where each row is a data point
data_points = np.column_stack((X, Y))

# Fit the SVM model
clf = svm.SVC(kernel='linear')
clf.fit(data_points, Class)

# decision boundary ax + by + c = 0 에서 a = clf.coef_[0, 0], b = clf.coef_[0, 1], c = clf.intercept_[0]
a = -clf.coef_[0, 0] / clf.coef_[0, 1]
b = -clf.intercept_[0] / clf.coef_[0, 1]

x_decision = np.linspace(np.min(X), np.max(X), 100)
y_decision = a * x_decision + b

# Plot the decision boundary line
plt.plot(x_decision, y_decision, 'r-')

# Create a scatter plot of the data points
plt.scatter(X, Y, c=Class)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Data Points")
plt.show()