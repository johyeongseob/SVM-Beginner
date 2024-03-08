"""

Paper name: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network / from: https://arxiv.org/pdf/1609.04802.pdf

This project aims to implement "generator network G (SRResNet)"

Architecture of Generator Network G is illustrated in Figure 4 in the paper above.

reference: https://github.com/jlaihong/image-super-resolution

"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Add
from tensorflow.python.keras.layers import PReLU
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

from div2k_loader import train_and_valid_dataset


div2k_folder = "C:/Users/johs/image/datasets/div2k"

save_path = "C:/Users/johs/Desktop/Github/SmartVision&Media/onboarding/Model/weights/SRResNet_50EPOCHS.h5"

train_dataset, valid_dataset = train_and_valid_dataset()

valid_dataset_subset = valid_dataset.take(10) # only taking 10 examples here to speed up evaluations during training

EPOCH = 50

def srresnet(num_filters=64, num_res_blocks=16):

    lr = Input(shape=(None, None, 3))
    x = lr / 255.0

    x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(num_res_blocks):
        block_input = x
        x = Conv2D(num_filters, kernel_size=3, padding='same')(block_input)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([block_input, x])

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    for _ in range(2):
        x = Conv2D(num_filters * 4, kernel_size=3, padding='same')(x)
        x = tf.nn.depth_to_space(x, 2)
        PReLU(shared_axes=[1, 2])(x)

    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    sr = (x + 1) * 127.5

    return Model(lr, sr)

def psnr_metric(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)

# generator = srresnet()

# generator.compile(optimizer=Adam(1e-4), loss=MeanSquaredError(), metrics=[psnr_metric])
# generator.fit(train_dataset,validation_data=valid_dataset_subset, epochs=EPOCH)
# generator.save(save_path)

srresnet_test = srresnet()

srresnet_test.load_weights(save_path)

input_path = 'C:/Users/johs/image/datasets/div2k/DIV2K_valid_HR/0886.png'
Input_file = Image.open(input_path)

target_size = (Input_file.width // 4, Input_file.height // 4)
bicubic_X4 = Input_file.resize(target_size, Image.BICUBIC)

sr = (np.array(bicubic_X4))[np.newaxis]
sr = srresnet_test.predict(sr)[0]
sr = tf.clip_by_value(sr, 0, 255)
sr = tf.round(sr)
sr = tf.cast(sr, tf.uint8)
sr = Image.fromarray(sr.numpy())

fig, axes = plt.subplots(1,3, figsize = (20,10))

axes[0].imshow(Input_file)
axes[1].imshow(bicubic_X4)
axes[2].imshow(np.squeeze(sr))

axes[0].set_title('input_img')
axes[1].set_title('bicubic_X4_img')
axes[2].set_title('srresnet_img')

plt.show()