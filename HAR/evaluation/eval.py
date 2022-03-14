import tensorflow as tf
import tensorflow_addons as tfa
from evaluation.metrics import BalancedAccuarcy, ConfusionMatrix
import logging

#import metrics


class Evaluator():
    def __init__(self,model, checkpoint, ds_test, ds_info, run_paths) :
        '''this class creates a trainer object which is responsible for training in an expirement

        Parameters :
            model(tf.Keras.Model) : model to be trained of type tf.Keras.Model
            checkpoint(str) : path of the run directory incase of evaluating from checkpoint. it has no direct use and can be used only as a sanity check
            ds_test(tf.Data.Dataset): test dataset of type tf.Data.Dataset
            ds_info(dict): a dict various information related to the dataset like image sizes
            run_paths(dict): a dict containing the paths related to the run like the ckpt dir and tensorboard summary dir

        Returns :
            (int) : accuracy of the model at each logging interval and at the end of thr training
        '''

        if checkpoint != None :
            self.model = model
            self.optimizer = tf.keras.optimizers.Adam()
            checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
            manager = tf.train.CheckpointManager(
                checkpoint, directory=run_paths['path_ckpts_train'], max_to_keep=10)
            checkpoint.restore(manager.latest_checkpoint)
            logging.info("loaded model from checkpoint")
        else :
            self.model = model


        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # Metrics
        self.eval_loss = tf.keras.metrics.Mean(name='eval_loss')
        self.eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')
        self.balanced_eval_accuracy = BalancedAccuarcy(name = 'balanced_eval_accuracy')
        self.eval_confustion_matrix = ConfusionMatrix()

        self.ds_test = ds_test


    @tf.function
    def test_step(self, inputs, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        # inputs = tf.expand_dims(inputs, axis=-1)
        predictions = self.model(inputs, training=False) - 1
        indecies = tf.not_equal(labels, -1)
        labels = tf.boolean_mask(labels, indecies)
        predictions = tf.boolean_mask(predictions, indecies)
        t_loss = self.loss_object(labels, predictions)

        # update metrics
        self.eval_loss(t_loss)
        self.eval_accuracy(labels, predictions)
        self.balanced_eval_accuracy(labels, predictions)
        self.eval_confustion_matrix(labels, predictions)

    def eval(self) :

        for test_timestamp, test_acc, test_gyro, test_activity in self.ds_test:

            self.test_step(tf.concat([test_acc, test_gyro], -1), test_activity)

        template = 'Evaluation :- Loss: {}, Accuracy: {}, Balanced Accuracy: {}'
        logging.info(template.format(self.eval_loss.result(),
                                    self.eval_accuracy.result() * 100,
                                    tf.reduce_sum(self.balanced_eval_accuracy.result()) * 100
                                    ))

        template_confustion = 'Evaluation Confusion Matrix : \n {}'
        logging.info(template_confustion.format(self.eval_confustion_matrix.result()))


        # Reset eval metrics
        self.eval_loss.reset_states()
        self.eval_accuracy.reset_states()
        self.balanced_eval_accuracy.reset_states()
        self.eval_confustion_matrix.reset_states()
