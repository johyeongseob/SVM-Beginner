"""
Classification Algorithm

Observable Data > Modeling(Learning) > Information

"""

from matplotlib import pyplot as plt
import numpy as np
 

X_R = np.array([0.8147, 0.9058, 0.9134, 0.5469, 0.9649])
Y_R = np.array([0.8576, 0.9706, 0.8854, 0.9157, 0.9595])

X_B = np.array([0.127, 0.6324, 0.0975, 0.2785, 0.9575])
Y_B = np.array([0.2572, 0.8003, 0.1419, 0.4218, 0.7922])

x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0 ])
y = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

plt.plot(X_R,Y_R,'ro')
plt.plot(X_B,Y_B,'bo')
plt.plot(x,y, label='y=0.8')
plt.legend()
plt.show()


