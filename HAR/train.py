import datetime
import os
import gin
import tensorflow as tf
import logging
from visualization import visualize

from evaluation.metrics import BalancedAccuarcy, ConfusionMatrix
import tensorflow_addons as tfa

import json


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval):
        """Create a trainer instance.
                Parameters:
                    model (Keras.Models): The model to be trianed
                    ds_train (tf.dataset): the training dataset
                    ds_val (tf.dataset): The validation dataset
                    ds_info (tf.dataset): A dictionary with the dataset info
                    run_paths (string): directories to save the model
                    total_steps (int): the total number of training steps
                    log_interval (int): The number of steps to log the info.
                    ckpt_interval (int): The number of steps to save the current model weights
                Returns:
                    (Trainer): Trainer object
                """
        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

        # Summary Writer
        self.writer = tf.summary.create_file_writer(run_paths['summary'])
        self.writer.set_as_default()

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()

        # Checkpoint Manager
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
        self.manager = tf.train.CheckpointManager(
            checkpoint, directory=self.run_paths['path_ckpts_train'], max_to_keep=10)

        # load an existing model
        if self.manager.latest_checkpoint:
            checkpoint.restore(self.manager.latest_checkpoint)
            if os.path.exists(os.path.join(run_paths['path_model_id'], 'state.json')):
                with open(os.path.join(run_paths['path_model_id'], 'state.json'), 'r') as fp:
                    self.state = json.load(fp)
            else:
                self.state = {'best_validation_balanced_accuracy': 0, 'validation_balanced_accuracy': 0}
            logging.info("loaded model from checkpoint")
        else:
            self.state = {'best_validation_balanced_accuracy': 0, 'validation_balanced_accuracy': 0}
            logging.info("starting model from rnadom init")

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.balanced_train_accuracy = BalancedAccuarcy(name='balanced_train_accuracy')
        self.confustion_matrix = ConfusionMatrix()

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.balanced_test_accuracy = BalancedAccuarcy(name='balanced_test_accuracy')
        self.test_confustion_matrix = ConfusionMatrix()

    # @tf.function
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            # inputs = tf.expand_dims(inputs, axis=-1)
            predictions = self.model(inputs, training=True)
            indecies = tf.not_equal(labels, -1)
            labels = tf.boolean_mask(labels, indecies)
            predictions = tf.boolean_mask(predictions, indecies)

            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # update metrics
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        self.balanced_train_accuracy(labels, predictions)
        self.confustion_matrix(labels, predictions)

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
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        self.balanced_test_accuracy(labels, predictions)
        self.test_confustion_matrix(labels, predictions)

    def train(self):
        for idx, (timestamp, acc, gyro, activity) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(tf.concat([acc, gyro], -1), activity)

            # if step == profiler_start_step :
            #     options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3,
            #                                        python_tracer_level = 1,
            #                                        device_tracer_level = 1)
            #     tf.profiler.experimental.start(logdir = self.run_paths['summary'],options= options)
            # elif step == profiler_stop_step :
            #     tf.profiler.experimental.stop(save = True)

            if step % self.log_interval == 0:

                vis_images = []

                for (val_timestamp, val_acc, val_gyro, val_activity) in self.ds_val:
                    self.test_step(tf.concat([val_acc, val_gyro], -1), val_activity)
                    #Visualize the data and the predictions
                    predictions = tf.argmax(self.model(tf.concat([val_acc, val_gyro], -1), training=False), axis=-1) + 1
                    vis_images.append(visualize(val_activity + 1, val_acc, val_gyro, predictions))

                #logging info
                template = 'Step {}, Loss: {}, Accuracy: {}, Balanced Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Test Balanced Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             (self.balanced_train_accuracy.result()) * 100,
                                             self.test_loss.result(),
                                             self.test_accuracy.result() * 100,
                                             (self.balanced_test_accuracy.result() * 100),
                                             ))

                template_confustion = 'Confusion Matrix : \n {} \n Test Confusion Matrix : \n {}'
                logging.info(
                    template_confustion.format(self.confustion_matrix.result(), self.test_confustion_matrix.result()))

                # Write summary to tensorboard
                tf.summary.scalar('Training Loss', self.train_loss.result(), step=step)
                tf.summary.scalar('Training Accuracy', self.train_accuracy.result(), step=step)
                tf.summary.scalar('Training Balanced Accuracy', (self.balanced_train_accuracy.result()) * 100,
                                  step=step)

                tf.summary.scalar('Validation Loss', self.test_loss.result(), step=step)
                tf.summary.scalar('Validation Accuracy', self.test_accuracy.result(), step=step)
                tf.summary.scalar('Validation Balanced Accuracy', (self.balanced_test_accuracy.result()) * 100,
                                  step=step)

                #visualize the data in the tensorboard
                for i, image in enumerate(vis_images):
                    tf.summary.image('val image {}'.format(i), image, step=step, )

                self.state['validation_balanced_accuracy'] = float((self.balanced_test_accuracy.result()).numpy() * 100)

                #Computing best validation balanced accuracy
                if self.state['validation_balanced_accuracy'] > self.state['best_validation_balanced_accuracy']:
                    self.state['best_validation_balanced_accuracy'] = self.state['validation_balanced_accuracy']
                    self.model.save(os.path.join(self.run_paths["path_ckpts_train"], "best_model"))
                    logging.info('best validation balanced accuracy : {}'.format(
                        self.state['best_validation_balanced_accuracy']))
                    with open(os.path.join(self.run_paths['path_model_id'], 'state.json'), 'w') as fp:
                        json.dump(self.state, fp)

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()
                self.balanced_train_accuracy.reset_states()
                self.confustion_matrix.reset_states()

                # Reset test metrics
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()
                self.balanced_test_accuracy.reset_states()
                self.test_confustion_matrix.reset_states()

                yield self.test_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                self.manager.save()

            #Final training step
            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                self.manager.save()
                self.model.save(self.run_paths["path_ckpts_train"])
                return self.test_accuracy.result().numpy()
