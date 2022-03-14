import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils.generic_utils import to_list
import numpy as np

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, name="confusion_matrix", n_classes=12, **kwargs):
        '''represents a confusion matrix for multiclass classification

        Parameters :
            name(str) : name of the metric
            n_classes(int): number of the classes for the confunsion matrix

        Returns :
            (ConfusionMatrix) : confusion matrix obejct'''

        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.confustion_matrix = self.add_weight('cm', shape=(n_classes, n_classes), dtype=tf.int32,
                                                 initializer='zeros')
        self.n_classes = n_classes

    def update_state(self, y_true, y_pred):
        # self.confustion_matrix.assign_add(tf.math.confusion_matrix(y_true,y_true,num_classes=self.n_classes))

        self.confustion_matrix.assign_add(
            tf.math.confusion_matrix(y_true, tf.argmax(y_pred, axis=-1), num_classes=self.n_classes))

    def result(self):
        return self.confustion_matrix

    def reset_states(self):
        self.confustion_matrix.assign(tf.zeros_like(self.confustion_matrix))


class BalancedAccuarcy(Metric):
    

    def __init__(self, name="balanced accuracy", n_classes=12, **kwargs):
        '''represents balanced accuracy for binary classification task

        Parameters :
            name(str): name of the metric
            n_classes(int): number of classes 

        Returns :
            (BalancedAccuarcy) : balanced accuracy obejct'''
        super(BalancedAccuarcy, self).__init__(name=name, **kwargs)
        self.confustion_matrix = self.add_weight('cm', shape=(n_classes, n_classes), dtype=tf.int32,
                                                 initializer='zeros')
        self.n_classes = n_classes

    def update_state(self, y_true, y_pred):
        # self.confustion_matrix.assign_add(tf.math.confusion_matrix(y_true,y_true,num_classes=self.n_classes))

        self.confustion_matrix.assign_add(
            tf.math.confusion_matrix(y_true, tf.argmax(y_pred, axis=-1), num_classes=self.n_classes))

    def result(self):
        return tf.math.reduce_mean(tf.math.divide(tf.linalg.diag_part(self.confustion_matrix),tf.reduce_sum(self.confustion_matrix,axis = 1)))

    def reset_states(self):
        self.confustion_matrix.assign(tf.zeros_like(self.confustion_matrix))