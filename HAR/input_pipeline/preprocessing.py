import gin
import tensorflow as tf

@gin.configurable
def preprocess(timestamp, acc, gyro, activity):

    return timestamp, acc, gyro, activity-1

def augment(image, label):
    """Data augmentation"""

    return image, label