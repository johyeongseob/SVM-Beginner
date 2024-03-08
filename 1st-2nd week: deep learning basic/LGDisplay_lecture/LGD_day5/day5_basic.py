import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from tensorflow.python.keras.layers import Conv2D

# 훈련 파라미터 설정
EPOCH = 1
batch_size = 512

# 이미지 크기
img_rows = 28
img_cols = 28

# (훈련 이미지, 훈련 레이블), (테스트 이미지, 테스트 레이블) 로 데이터 구분
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# data를 로드한 다음 28X28 크기의 matrix를 input shape으로 정의
# 28X28X1로 데이터의 reshape
input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# 훈련 이미지와 테스트 이미지 픽셀 값의 범위를 (0, 1)로 변경
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# train, test 데이터 갯수 및 shape
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 분류할 class 개수
num_classes = 10

# 레이블을 one-hot encoding 형식으로 변환
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# CNN 모델 구조 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.predict(x_train[:1])

# 모델 구조 출력
print(model.summary())

# Loss function, Optimizer, 검증 척도(metric) 설정
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCH, validation_data=(x_test, y_test))

# 학습 후 Test loss, accuracy 출력
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])