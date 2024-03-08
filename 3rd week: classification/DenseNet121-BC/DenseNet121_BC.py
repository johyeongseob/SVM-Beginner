"""

DenseNet121-BC Structure

Convolution:            7x7 conv, stride 2
Pooling:                3x3 max pool, stride 2
Dense Block (1):        [[1x1 conv] [3x3 conv]] x 6
Transition Layer (1):   1x1 conv + 2x2 average pool, stride 2
Dense Block (2):        [[1x1 conv] [3x3 conv]] x 12
Transition Layer (2):   1x1 conv + 2x2 average pool, stride 2
Dense Block (3):        [[1x1 conv] [3x3 conv]] x 24
Transition Layer (3):   1x1 conv + 2x2 average pool, stride 2
Dense Block (4):        [[1x1 conv] [3x3 conv]] x 16
Classification Layer):  7x7 global average pool + 1000D fully connected, softmax

Bottleneck layer: BN, ReLU, conv, BN, ReLU, conv

"""

import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, BatchNormalization, Activation, Dense, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
from keras.datasets import cifar10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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

input_tensor = Input(shape=(32,32,3), dtype='float32', name='input')

x = input_tensor

# DenseNet121 모델 구조 설정

dense_blocks = 4
blocks = [6,12,24,16] # For DenseNet-121
k = 8
th =0.5

# Convolution: 7x7 conv, stride 2
x = Conv2D(16, (7, 7), strides=(2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Pooling: 3x3 max pool, strides 2
x = MaxPooling2D((3, 3), 2, padding='same')(x)

# Dense Block & Transition Layer
for block_idx in range(dense_blocks - 1):
    concat = x 

    # Dense Block: [[1x1 conv] [3x3 conv]]
    for i in range(blocks[block_idx]):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(k, (1, 1))(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(k, (3, 3), padding='same')(x)

        x = concatenate([concat, x], axis=-1)
        concat = x


    # Transition Layer: 1x1 conv + 2x2 average pool, stride 2
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(x.shape[-1]/2), (1, 1))(x)

    x = AveragePooling2D((2, 2), 2, padding='same')(x)
    print(x.shape)

concat = x 

for i in range(blocks[-1]):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3, 3), padding='same')(x)

    x = concatenate([concat, x], axis=-1)
    concat = x
print(x.shape)

x = GlobalAveragePooling2D()(x)

output_tensor = Dense(10, activation='softmax')(x)

DenseNet121 = Model(input_tensor, output_tensor)

# 모델 구조 출력
print(DenseNet121.summary())

# Loss function, Optimizer, 검증 척도(metric) 설정
DenseNet121.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
hist = DenseNet121.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCH, validation_data=(x_test, y_test))

# 학습 후 Test loss, accuracy 출력
score = DenseNet121.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])