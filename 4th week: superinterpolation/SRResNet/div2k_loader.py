import os
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from tensorflow.python.data.experimental import AUTOTUNE

# hr_train_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
# hr_valid_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
# train_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip"
# valid_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip"

data_directory = "C:/Users/johs/image/datasets/div2k"

mapping = lambda lr, hr: random_crop(lr, hr)

def image_dataset_from_directory(data_directory, image_directory):

    images_path = os.path.join(data_directory, image_directory)
    filenames = sorted(glob.glob(images_path + "/*.png"))

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(tf.io.read_file)
    dataset = dataset.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)

    return dataset


def train_and_valid_dataset():
    training_lr_dataset = image_dataset_from_directory(data_directory, "DIV2K_train_LR_bicubic/X4")
    training_hr_dataset = image_dataset_from_directory(data_directory, "DIV2K_train_HR")
    training_dataset = tf.data.Dataset.zip((training_lr_dataset, training_hr_dataset))
    training_dataset = training_dataset.repeat(2)
    training_dataset = training_dataset.map(mapping, num_parallel_calls=AUTOTUNE)
    training_dataset = training_dataset.batch(16)
    

    valid_lr_dataset = image_dataset_from_directory(data_directory, "DIV2K_valid_LR_bicubic/X4")
    valid_hr_dataset = image_dataset_from_directory(data_directory, "DIV2K_valid_HR")
    valid_dataset = tf.data.Dataset.zip((valid_lr_dataset, valid_hr_dataset))
    valid_dataset = valid_dataset.batch(1)


    # # valid_dataset 1장 이미지 시각화

    # shuffled_dataset = valid_dataset.shuffle(buffer_size=100, reshuffle_each_iteration=False)
    # batch = next(iter(shuffled_dataset.batch(1)))

    # lr_image = batch[0][0]  # LR 이미지
    # hr_image = batch[1][0]  # HR 이미지

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(lr_image.numpy().astype("uint8"))
    # plt.title("LR Image")
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(hr_image.numpy().astype("uint8"))
    # plt.title("HR Image")
    # plt.axis('off')

    # plt.show()

    return training_dataset, valid_dataset


def random_crop(lr_img, hr_img):
    
    lr_shape = tf.shape(lr_img)[:2]

    lr_top = tf.random.uniform(shape=(), maxval=lr_shape[0] - 24 + 1, dtype=tf.int32)
    lr_left = tf.random.uniform(shape=(), maxval=lr_shape[1] - 24 + 1, dtype=tf.int32)

    hr_top = lr_top * 4
    hr_left = lr_left * 4

    lr_crop = lr_img[lr_top:lr_top + 24, lr_left:lr_left + 24]
    hr_crop = hr_img[hr_top:hr_top + 96, hr_left:hr_left + 96]

    return lr_crop, hr_crop