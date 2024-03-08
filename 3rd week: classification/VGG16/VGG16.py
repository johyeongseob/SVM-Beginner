"""

VGG16 Structure

2개의 합성곱 층 (64개의 3x3 필터, ReLU) + 최대 풀링 층 (2x2, 스트라이드 2)
2개의 합성곱 층 (128개의 3x3 필터, ReLU) + 최대 풀링 층 (2x2, 스트라이드 2)
3개의 합성곱 층 (256개의 3x3 필터, ReLU) + 최대 풀링 층 (2x2, 스트라이드 2)
3개의 합성곱 층 (512개의 3x3 필터, ReLU) + 최대 풀링 층 (2x2, 스트라이드 2)
3개의 합성곱 층 (512개의 3x3 필터, ReLU) + 최대 풀링 층 (2x2, 스트라이드 2)
3개의 완전 연결 층 (4096, 4096, 1000 뉴런, ReLU) + Softmax 출력 층

"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.datasets import cifar10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

save_path = "C:/Users/johs/Desktop/Github/SmartVision&Media/onboarding/Model/weight_save/VGG_10EPOCHS.h5"

EPOCH = 10
batch_size = 128
input_shape = (32, 32, 3)

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

def VGG16(input_shape):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# VGG16 = VGG16(input_shape)

# print(VGG16.summary())
# VGG16.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# VGG16.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCH, validation_data=(x_test, y_test))
# VGG16.save_weights(save_path)

# # 테스트 데이터로 평가
# score = VGG16.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

VGG16_test = VGG16((32, 32, 3))
VGG16_test.load_weights(save_path)  # 저장된 가중치 불러오기

# 임의의 이미지와 라벨 가져오기
index = np.random.randint(0, x_test.shape[0])  # 테스트 데이터에서 랜덤한 인덱스 선택
image = x_test[index]
label = y_test[index]

# 모델 예측값 가져오기
predictions = VGG16_test.predict(np.expand_dims(image, axis=0))
predicted_label = np.argmax(predictions)

# 이미지 시각화
fig, axes = plt.subplots(1,2, figsize = (4, 2))

axes[0].imshow(image)
axes[1].imshow(image)

axes[0].set_title('True Label: {}'.format(np.argmax(label)))
axes[1].set_title('Predicted Label: {}'.format(predicted_label))

plt.show()
