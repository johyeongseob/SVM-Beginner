"""

SRCNN: https://kevinitcoding.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EA%B5%AC%ED%98%84-SRCNNby-Keras%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%9C-Super-Resolution

"""


import numpy as np
import matplotlib.pyplot as plt
#import h5py
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras import initializers
from keras.optimizers import Adam
import random

import cv2
import os
from glob import glob


n1,n2,n3 = 128, 64, 3
f1,f2,f3 = 9,3,5
upscale_factor = 3
#To avoid border effects during training, all the convolutional layers have no padding, and the network produces a smaller output ((fsub − f1 − f2 − f3 + 3)2 × c).

input_size = 33
output_size = input_size - f1 - f2 - f3 + 3
pad = abs(input_size - output_size) // 2 # 7
stride = 14

batch_size = 128
epochs = 200

path = "C:/Users/johs/image/SRCNN/T91"
save_path = "C:/Users/johs/Desktop/Github/SmartVision&Media/onboarding/Model/weight_save/SRCNN_200EPOCHS.h5"

img_paths = glob(path + '/' + '*.png')

# Check
# img = cv2.imread(img_paths[0], cv2.COLOR_BGR2RGB)
# print(img.shape)
# plt.imshow(img)
# plt.show()

sub_lr_imgs = []
sub_hr_imgs = []

for img_path in img_paths:
  img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
  
  # mod_crop
  h = img.shape[0] - np.mod(img.shape[0], upscale_factor)
  w = img.shape[1] - np.mod(img.shape[1], upscale_factor)
  img = img[:h, :w, :]

  # zoom_img
  label = img.astype('float') / 255
  temp_input = cv2.resize(label, dsize=(0,0), fx = 1/upscale_factor, fy = 1/upscale_factor,
                          interpolation = cv2.INTER_AREA)
  input = cv2.resize(temp_input, dsize=(0,0), fx = upscale_factor, fy = upscale_factor,
                     interpolation = cv2.INTER_CUBIC)
  
  # Crop: img to sub_imgs
  for h in range(0, input.shape[0] - input_size + 1, stride):
    for w in range(0, input.shape[1] - input_size + 1, stride):
      sub_lr_img = input[h:h+input_size, w:w+input_size, :]
      sub_hr_img = label[h+pad:h+pad+output_size, w+pad:w+pad+output_size, :]

      sub_lr_imgs.append(sub_lr_img)
      sub_hr_imgs.append(sub_hr_img)

sub_lr_imgs = np.asarray(sub_lr_imgs)
sub_hr_imgs = np.asarray(sub_hr_imgs)

# Check
# fig, axes = plt.subplots(1,2, figsize = (5,5))
# idx = random.randint(0, sub_lr_imgs.shape[0])

# axes[0].imshow(sub_lr_imgs[idx])
# axes[1].imshow(sub_hr_imgs[idx])

# print(idx)
# axes[0].set_title('lr_img')
# axes[1].set_title('hr_img')
# plt.show()


# 모델 학습: 초기에 학습하고 가중치 저장 후, 주석 처리.
# initializer = initializers.GlorotNormal()

# SRCNN = Sequential()
# SRCNN.add(Input(shape = (33, 33, 3)))
# SRCNN.add(Conv2D(filters = n1, kernel_size = f1, activation = 'ReLU', input_shape = (33,33,3), kernel_initializer = initializer, bias_initializer = 'zeros', name = 'Conv1'))
# SRCNN.add(Conv2D(filters = n2, kernel_size = f2, activation = 'ReLU', kernel_initializer = initializer, bias_initializer = 'zeros', name = 'Conv2'))
# SRCNN.add(Conv2D(filters = n3, kernel_size = f3, activation = 'linear', kernel_initializer = initializer, bias_initializer = 'zeros', name = 'Conv3'))

# print(SRCNN.summary())

# optimizer = Adam(lr = 0.0003)

# SRCNN.compile(optimizer = optimizer, loss ='MSE', metrics = ['MSE'])
# SRCNN.fit(sub_lr_imgs, sub_hr_imgs, batch_size = batch_size, epochs = epochs, verbose=1)

# SRCNN.save(save_path)


initializer = initializers.GlorotNormal()

def predict_model():
  SRCNN = Sequential()
  SRCNN.add(Conv2D(filters = n1, kernel_size = f1, activation = 'ReLU', input_shape = (None,None,3), kernel_initializer = initializer, bias_initializer = 'zeros', name = 'Conv1'))
  SRCNN.add(Conv2D(filters = n2, kernel_size = f2, activation = 'ReLU', kernel_initializer = initializer, bias_initializer = 'zeros', name = 'Conv2'))
  SRCNN.add(Conv2D(filters = n3, kernel_size = f3, activation = 'linear', kernel_initializer = initializer, bias_initializer = 'zeros', name = 'Conv3'))

  return SRCNN

SRCNN_Test = predict_model()
SRCNN_Test.load_weights(save_path)

hr_img_path = 'C:/Users/johs/image/SRCNN/Set5/butterfly.png'

hr_img = cv2.imread(hr_img_path)
hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
print("img shape: {}".format(hr_img.shape))

hr_img = hr_img.astype('float') / 255
temp_img = cv2.resize(hr_img, dsize=(0,0), fx = 1/upscale_factor, fy = 1/upscale_factor, interpolation = cv2.INTER_AREA)
bicubic_img = cv2.resize(temp_img, dsize=(0,0), fx = upscale_factor, fy = upscale_factor, interpolation = cv2.INTER_CUBIC)

input_img = bicubic_img[np.newaxis, :]
srcnn_img = SRCNN_Test.predict(input_img)

fig, axes = plt.subplots(1,3, figsize = (10,5))

axes[0].imshow(hr_img)
axes[1].imshow(bicubic_img)
axes[2].imshow(np.squeeze(srcnn_img))

axes[0].set_title('hr_img')
axes[1].set_title('bicubic_img')
axes[2].set_title('srcnn_img')

plt.show()