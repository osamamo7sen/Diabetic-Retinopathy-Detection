from random import random
import gin
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from tensorflow.python.ops.gen_image_ops import random_crop
import tensorflow_addons as tfa
from absl import flags


@gin.configurable
def preprocess(image, label, img_height=256, img_width=256, binary_classification=True):
    """Dataset preprocessing: Normalizing and resizing"""

    # Resize image
    image_resized = tf.image.resize(image, size=(img_height, img_width))
    image_orig = tf.image.resize(image, size=(img_height, img_width)) / 255.

    # change labels
    if binary_classification:
        modified_label = tf.where(label > 1, 1, 0)

    return image_resized, modified_label, image_orig


@gin.configurable
def augment(image, label, image_orig,
            skip_probability=0.5,
            horizontal_flip=True,
            vertical_flip=True,
            random_brightness=True,
            random_saturation=True,
            random_hue=True,
            random_contrast=True,
            random_roation=True,
            max_rotation=20,
            random_crop=True,
            intermediate_size=(300, 300)):
    """Data augmentation"""
    if skip_probability < tf.random.uniform(()):

        if horizontal_flip:
            image = tf.image.random_flip_left_right(image)
        if vertical_flip:
            image = tf.image.random_flip_up_down(image)
        if random_brightness:
            image = tf.image.random_brightness(image, max_delta=0.15)
        if random_saturation:
            image = tf.image.random_saturation(image, lower=0.75, upper=1.5)
        if random_hue:
            image = tf.image.random_hue(image, max_delta=0.1)
        if random_contrast:
            image = tf.image.random_contrast(image, lower=0.75, upper=1.5)
        if random_roation:
            angle_rad = max_rotation / 180 * np.pi
            # random_angle = tf.random.uniform([1],-angle_rad,angle_rad)
            # image = tfa.image.rotate(image,random_angle,fill_mode = "nearest",interpolation = "nearest")
        if random_crop:
            original_size = image.shape[-3:]
            image = tf.image.resize(image, intermediate_size)
            image = tf.image.random_crop(image, original_size)

    return image, label, image_orig


def post_augment_preprocess(image, label, image_orig):
    image = preprocess_input(image)
    return image, label, image_orig


@gin.configurable
def augment_batch(X, y,
                  intermediate_size=(300, 300),
                  batch_size=32,
                  min_crop_percent=0.001,
                  max_crop_percent=0.01,
                  crop_probability=0,
                  rotation_range=20):
    batch_size = tf.shape(X)[0]
    with tf.name_scope('transformation'):
        # code borrowed from https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19
        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if rotation_range > 0:
            angle_rad = rotation_range / 180 * np.pi
            angles = tf.random.uniform([batch_size], -angle_rad, angle_rad)
            transforms += [
                tfa.image.angles_to_projective_transforms(angles, intermediate_size[0], intermediate_size[1])]

        #Random cropping with predefined percentages
        if crop_probability > 0:
            crop_pct = tf.random.uniform([batch_size], min_crop_percent, max_crop_percent)
            left = tf.random.uniform([batch_size], 0, intermediate_size[0] * (1.0 - crop_pct))
            top = tf.random.uniform([batch_size], 0, intermediate_size[1] * (1.0 - crop_pct))
            crop_transform = tf.stack([
                crop_pct,
                tf.zeros([batch_size]), top,
                tf.zeros([batch_size]), crop_pct, left,
                tf.zeros([batch_size]),
                tf.zeros([batch_size])
            ], 1)
            coin = tf.expand_dims(tf.less(tf.random.uniform([batch_size], 0, 1.0), crop_probability), axis=-1)
            transforms += [tf.where(coin, crop_transform, tf.tile(tf.expand_dims(identity, 0), [batch_size, 1]))]
        if len(transforms) > 0:
            for trans in transforms:
                X = tfa.image.transform(X,
                                        trans,
                                        interpolation='BILINEAR')  # or 'NEAREST'

            # X = tfa.image.transform(X,
            #       tfa.image.compose_transforms(*transforms),
            #       interpolation='BILINEAR') # or 'NEAREST'

    return X, y


def augment_image(image, label, img_height, img_width, intermediate_trans='scale'):
    if intermediate_trans == 'scale':
        image = tf.image.resize(image, (img_height, img_width))
    elif intermediate_trans == 'crop':
        image = tf.image.resize_with_crop_or_pad(image, img_height, img_width)
    else:
        raise ValueError('Invalid Operation {}'.format(intermediate_trans))
    return image, label
