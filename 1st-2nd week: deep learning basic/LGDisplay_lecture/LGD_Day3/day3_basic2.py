import numpy as np
import matplotlib.pyplot as plt
# A 10x2
# AT 2x10
# K(A, AT) 10x10
# I 10x10
# y 10x1
# inf 1x2
# K(inf, AT) 1x10
# np.matmul(K(inf, AT), alpha) 1x1
def K(X, Z):
    k = 1 + np.matmul(X, Z)
    return np.square(k)
x=[0.8147, 0.9058, 0.127, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649] #데이터 X의 배열
y=[0.8576, 0.9706, 0.2572, 0.8854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595] #데이터 Y의 배열
# y = ax + b

# plt.plot(x, y, 'o') #데이터 좌표 그리기
# plt.scatter(x, y)
# plt.show()

ones = np.ones(len(x)) # x와 같은 길이의 1로 이루어진 행렬
A = np.stack([x, ones], axis=1) # 상수항 추가 10x2
AT = np.transpose(A) # 전치행렬 2x10
I = np.identity(A.shape[0]) # 단위행렬 I 10x10
Lambda = [100, 10, 1, 0.1, 0.01, 0.001, 0] # lambda

for l in Lambda:
    alpha = np.matmul(np.linalg.inv(K(A, AT) + l * I), y)
    
    plt.scatter(x, y)
    lx = []
    ly = []
    for i in np.linspace(0, 1, 100): # 직선 좌표 수집 0, 0.01, 0.02 ..... 0.99
        
        lx.append(i)
        inf = np.array([i, 1])
        ly.append(np.matmul(K(inf, AT), alpha))

    plt.plot(lx, ly, '--r') # 직선 그리기

plt.title("Kernel Ridge Regression (Lambda : %f)" % l)
plt.xlabel("X")
plt.ylabel("Y")
plt.axis([0, 1, 0, 1])
plt.show()

