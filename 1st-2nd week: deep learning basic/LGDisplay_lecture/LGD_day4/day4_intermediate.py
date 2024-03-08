import tensorflow as tf
# 텐서플로우에 기본 내장된 mnist 모듈을 이용하여 데이터를 로드합니다.
# 지정한 폴더에 MNIST 데이터가 없는 경우 자동으로 데이터를 다운로드합니다.
# one_hot 옵션은 레이블을 동물 분류 예제에서 보았던 one_hot 방식의 데이터로 만들어줍니다.

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
x_train= tf.reshape(x_train, shape=[-1, 784])
x_test = tf.reshape(x_test, shape=[-1, 784])

#MNIST 입력 이미지를 [0, 1] 범위로 정규화하고, 레이블을 원-핫 인코딩 형식으로 변환합니다.
# 신경망의 레이어는 다음처럼 구성합니다.
# 784(입력 특성값)
#    -> 256 (히든레이어 뉴런 갯수) -> 256 (히든레이어 뉴런 갯수)
#    -> 10 (결과값 0~9 분류)

W1 = tf.Variable(tf.random.normal([784,256], stddev=0.01))
W1 = tf.Variable(tf.cast(W1, dtype=tf.float64))
W2 = tf.Variable(tf.random.normal([256,256], stddev=0.01))
W2 = tf.Variable(tf.cast(W2, dtype=tf.float64))
W3 = tf.Variable(tf.random.normal([256,10], stddev=0.01))
W3 = tf.Variable(tf.cast(W3, dtype=tf.float64))

def model(X):
    # 입력값에 가중치를 곱하고 ReLU 함수를 이용하여 레이어를 만듭니다.
    L1 = tf.nn.relu(tf.matmul(X, W1))
    # L1 레이어의 출력값에 가중치를 곱하고 ReLU 함수를 이용하여 레이어를 만듭니다
    L2 = tf.nn.relu(tf.matmul(L1, W2))
    L3 = tf.matmul(L2, W3)
    # 최종 모델의 출력값은 W3 변수를 곱해 10개의 분류를 가지게 됩니다.
    return L3

def cost(model, Y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)

#########
# 신경망 모델 학습
######
batch_size = 100
total_batch = int(x_train.shape[0]/ batch_size)
splits_x = tf.split(x_train, num_or_size_splits=total_batch)
splits_y = tf.split(y_train, num_or_size_splits=total_batch)

# Print the shape of each split tensor
for epoch in range(1):
    for i in range(0, total_batch):
        with tf.GradientTape() as tape:
            Y_pred = model(splits_x[i])
            current_loss = cost(Y_pred, splits_y[i])
        gradients = tape.gradient(current_loss, [W1, W2, W3])
        optimizer.apply_gradients(zip(gradients, [W1, W2, W3]))
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', (current_loss.numpy() / batch_size))

#########
# # 결과 확인
# ######
# model 로 예측한 값과 실제 레이블인 Y의 값을 비교합니다.
# tf.argmax 함수를 이용해 예측한 값에서 가장 큰 값을 예측한 레이블이라고 평가합니다.
# 예) [0.1 0 0 0.7 0 0.2 0 0 0 0] -> 3

target = tf.argmax(y_test, 1)
prediction= tf.argmax(model(x_test),1)
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % (accuracy*100))
