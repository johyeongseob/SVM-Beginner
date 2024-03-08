"""

ResNEt-18 Structure

Layer name: conv1, conv2_x, conv3_x, conv4_x, conv5_x

conv1: [7x7, 64, stride 2, padding = 3x3]
conv2_x: [3x3 max pool, stride 2] + [[3x3, 64] [3x3, 64]] X 2
conv3_x: [[3x3, 128] [3x3, 128]] X 2
conv4_x: [[3x3, 256] [3x3, 256]] X 2
conv5_x: [[3x3, 512] [3x3, 512]] X 2
End: [average pool, 1000-d, fc, softmax]

Down-sampling is performed by conv3_1, conv4_1, conv5_1 with a stride of 2.

"""

import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
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

# ResNet18 모델 구조 설정

#Conv1_1
x = ZeroPadding2D(padding=(3, 3))(x)
x = Conv2D(16, (7, 7), strides=(2, 2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = ZeroPadding2D(padding=(1,1))(x)


#Conv2
x = MaxPooling2D((3, 3), 2)(x)

shortcut = x

for i in range(2):

    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    shortcut = x


#Conv3
for i in range(2):

    if (i == 0):
        x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(32, (3, 3), padding='same')(x)
        shortcut = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(shortcut)
        x = BatchNormalization()(x)
        shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)

        shortcut = x
    
    else:
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)

        shortcut = x


#Conv4
for i in range(2):

    if (i == 0):
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        shortcut = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(shortcut)
        x = BatchNormalization()(x)
        shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)

        shortcut = x
    
    else:
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)

        shortcut = x


#Conv5
for i in range(2):

    if (i == 0):
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(128, (3, 3), padding='same')(x)
        shortcut = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(shortcut)
        x = BatchNormalization()(x)
        shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)

        shortcut = x

    else:
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)

x = GlobalAveragePooling2D()(x)
output_tensor = Dense(10, activation='softmax')(x)

resnet18 = Model(input_tensor, output_tensor)

# 모델 구조 출력
print(resnet18.summary())

# Loss function, Optimizer, 검증 척도(metric) 설정
resnet18.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
hist = resnet18.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCH, validation_data=(x_test, y_test))

# 학습 후 Test loss, accuracy 출력
score = resnet18.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])