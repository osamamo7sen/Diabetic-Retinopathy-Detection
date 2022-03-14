"""
This file is used to visualize the result for a whole sequence from the test set.
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

COLORMAP = ['w', 'lightcoral', 'chocolate', 'gold', 'g', 'royalblue',
            'plum', 'olive', 'm', 'brown', 'hotpink', 'r', 'grey']
CLASS_NAME = ['UNLABELED', 'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING',
              'STAND_TO_SIT', 'SIT_TO_STAND', ' SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE',
              'LIE_TO_STAND']

import io



def visualize(label, acc,gyro, pred):
    """Visualize the sensors data, and the labels and the predictions. Code is inspired by: https://github.com/LEGO999/Human-Activity-Recognition-HAPT
        Parameters:
            label (int): label of the human activity
            acc (float32): the three acceleromaeter readings
            gyro (float32): The three gyroscope readings
            pred (int): prediction of the human activity
        Returns:
            (image): visualization image
        """

    result_image = plt.figure(figsize=(10, 10))
    index = []
    for i in range(len(label[0])):
        index.append(i)
    #extract sensors data
    acc_01 = list(acc[0,:,0])
    acc_02 = list(acc[0,:,1])
    acc_03 = list(acc[0,:,2])
    gyro_01 = list(gyro[0,:,0])
    gyro_02 = list(gyro[0,:,1])
    gyro_03 = list(gyro[0,:,2])

    #plotting accelerometer data
    plt.subplot(5, 1, 1)
    plt.title('Accelerometer')
    plt.plot(index, acc_01, 'b')
    plt.plot(index, acc_02, 'g')
    plt.plot(index, acc_03, 'r')

    #plotting gyroscope data
    plt.subplot(5, 1, 2)
    plt.title('Gyroscope')
    plt.plot(index, gyro_01, 'b')
    plt.plot(index, gyro_02, 'g')
    plt.plot(index, gyro_03, 'r')

    #plotting labels
    plt.subplot(5, 1, 3)
    plt.title('Label')
    label = list(label[0])
    color_list = list()
    for i in range(len(label)):
        label_index = label[i]
        color = COLORMAP[(label_index)]
        color_list.append(color)

    #plotting predictions
    plt.vlines(index, 0, 5, linewidth=3, color=color_list)
    plt.subplot(5, 1, 4)
    plt.title('Prediction')
    pred = list(pred[0])
    color_list = list()
    for i in range(len(pred)):
        pred_index = pred[i]
        color = COLORMAP[(pred_index)]
        color_list.append(color)

    plt.vlines(index, 0, 5, linewidth=3, color=color_list)

    #plotting colormap
    plt.subplot(5, 1, 5)
    plt.title('Colormap')
    plt.bar(range(13), 5, color=COLORMAP)
    plt.xticks(range(13), CLASS_NAME, rotation=75)

    plt.subplots_adjust(hspace=1)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buffer.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    plt.close(result_image)

    return image


if __name__ == '__main__':
    visualize()