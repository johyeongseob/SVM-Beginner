"""
MLP(2layer)를 이용한 XOR problem solve
"""
import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_data = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# 신경망 모델 구성#########
X = tf.constant(x_data, dtype=tf.float32)
Y = tf.constant(y_data, dtype=tf.float32)
# 첫번째 가중치의 차원은 [특성, 히든 레이어의 뉴런갯수] -> [2, 2]
W1 = tf.Variable(tf.random.uniform([2,100], -1.0, 1.0))
# 두번째 가중치의 차원을 [첫번째 히든 레이어의 뉴런 갯수, 분류 갯수]
W2 = tf.Variable(tf.random.uniform([100,2], -1.0, 1.0))
b1 = tf.Variable(tf.zeros([100]))
b2 = tf.Variable(tf.zeros([2]))

def model(X):
    # 신경망의 히든 레이어에 가중치 W1과 편향 b1을 적용합니다
    L1 = tf.add(tf.matmul(X, W1), b1)
    L1 = tf.nn.relu(L1)
    L2 = tf.add(tf.matmul(L1, W2), b2)
    # 최종적인 아웃풋을 계산합니다.# 2개의 출력값 생성
    return tf.nn.softmax(L2)

def cost(model, Y):
    return tf.reduce_mean(-tf.reduce_sum(tf.math.multiply(Y, tf.math.log(model)), axis = 1))
# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)

#########
# 신경망 모델 학습
######
for epoch in range(100):
    with tf.GradientTape() as tape:
        Y_pred = model(X)
        current_loss = cost(Y_pred, Y)
    gradients = tape.gradient(current_loss, [W1, W2, b1, b2])
    optimizer.apply_gradients(zip(gradients, [W1, W2, b1, b2]))
    if (epoch + 1) % 10 == 0:
        print(epoch + 1, current_loss.numpy())

#########
# 결과 확인
######
# tf.argmax: 예측값과 실제값의 행렬에서 tf.argmax 를 이용해 가장 큰 값을 가져옵니다.

target = tf.argmax(Y, 1)
prediction= tf.argmax(model(X),1)
print('예측값:', prediction.numpy())
print('실제값:', target.numpy())
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % (accuracy * 100))