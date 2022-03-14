import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import io
import matplotlib.pyplot as plt
import cv2
import logging
import gin

@gin.configurable
def load_btgraham(filepath, target_pixel=128):
    with tf.io.gfile.GFile(filepath, mode="rb") as image_fobj:
        image = _btgraham_processing(
            image_fobj=image_fobj,
            filepath=filepath,
            target_pixels=target_pixel,
            crop_to_radius=True)
        image_decoded = cv2.imdecode(
            np.frombuffer(image.read(), dtype=np.uint8), flags=3)
        image_decoded = cv2.resize(image_decoded, (256, 256))
        return image_decoded


def _btgraham_processing(
        image_fobj, filepath, target_pixels, crop_to_radius=False):
    """Process an image as the winner of the 2015 Kaggle competition.
    Args:
      image_fobj: File object containing the original image.
      filepath: Filepath of the image, for logging purposes only.
      target_pixels: The number of target pixels for the radius of the image.
      crop_to_radius: If True, crop the borders of the image to remove gray areas.
    Returns:
      A file object.
    """
    cv2 = tfds.core.lazy_imports.cv2
    # Decode image using OpenCV2.
    image = cv2.imdecode(
        np.frombuffer(image_fobj.read(), dtype=np.uint8), flags=3)
    #image = cv2.resize(image, (256, 256))
    #print(image.shape)
    # Process the image.
    image = _scale_radius_size(image, filepath, target_radius_size=target_pixels)
    #print(image.shape)
    image = _subtract_local_average(image, target_radius_size=target_pixels)
    image = _mask_and_crop_to_radius(
        image, target_radius_size=target_pixels, radius_mask_ratio=0.9,
        crop_to_radius=crop_to_radius)
    # Encode the image with quality=72 and store it in a BytesIO object.
    _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return io.BytesIO(buff.tostring())


def _scale_radius_size(image, filepath, target_radius_size):
    """Scale the input image so that the radius of the eyeball is the given."""
    cv2 = tfds.core.lazy_imports.cv2
    x = image[image.shape[0] // 2, :, :].sum(axis=1)
    r = (x > x.mean() / 10.0).sum() / 2.0
    if r < 1.0:
        # Some images in the dataset are corrupted, causing the radius heuristic to
        # fail. In these cases, just assume that the radius is the height of the
        # original image.
        logging.info("Radius of image \"%s\" could not be determined.", filepath)
        r = image.shape[0] / 2.0
    s = target_radius_size / r
    return cv2.resize(image, dsize=None, fx=s, fy=s)


def _subtract_local_average(image, target_radius_size):
    cv2 = tfds.core.lazy_imports.cv2
    image_blurred = cv2.GaussianBlur(image, (0, 0), target_radius_size / 30)
    image = cv2.addWeighted(image, 4, image_blurred, -4, 128)
    return image


def _mask_and_crop_to_radius(
        image, target_radius_size, radius_mask_ratio=0.9, crop_to_radius=False):
    """Mask and crop image to the given radius ratio."""
    cv2 = tfds.core.lazy_imports.cv2
    mask = np.zeros(image.shape)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    radius = int(target_radius_size * radius_mask_ratio)
    cv2.circle(mask, center=center, radius=radius, color=(1, 1, 1), thickness=-1)
    image = image * mask + (1 - mask) * 128
    if crop_to_radius:
        x_max = min(image.shape[1] // 2 + radius, image.shape[1])
        x_min = max(image.shape[1] // 2 - radius, 0)
        y_max = min(image.shape[0] // 2 + radius, image.shape[0])
        y_min = max(image.shape[0] // 2 - radius, 0)
        image = image[y_min:y_max, x_min:x_max, :]
    return image
