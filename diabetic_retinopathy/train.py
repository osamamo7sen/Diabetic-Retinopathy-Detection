import os
import gin
import tensorflow as tf
import logging

from evaluation.metrics import BalancedAccuarcy, ConfusionMatrix
import tensorflow_addons as tfa

from visualization import visualize_integrated_gradients

import json

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval, transfer_model = False ,inital_lr = 1e-4,frozen_steps = 1000, transfer_lr = 1e-6):
        '''this class creates a trainer object which is responsible for training in an expirement

        Parameters :
            model(tf.Keras.Model) : model to be trained of type tf.Keras.Model
            ds_train(tf.Data.Dataset): training dataset of type tf.Data.Dataset
            ds_val(tf.Data.Dataset): validation_dataset of type tf.Data.Dataset
            ds_info(dict): a dict various information related to the dataset like image sizes
            run_paths(dict): a dict containing the paths related to the run like the ckpt dir and tensorboard summary dir
            total_steps(int): number of the total training steps the model should undergo
            log_interval(int): interval for logging and validating the model
            ckpt_interval(int): interval for checkpointing the model during training
            transfer_model(bool): a flag for tranfer learning training
            inital_lr(float): the initial learning rate for the optimizer during transfer learning
            frozen_steps(int): number of steps during which the backbone is frozen during tranfer leanring
            transfer_lr(float): learning rate for the model after unfreezing the backbone

        Returns :
            (Trainer) : trainer object
        '''

        # Summary Writer 
        self.writer = tf.summary.create_file_writer(run_paths['summary'])
        self.writer.set_as_default()

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # freezes the backbone at the begining of the trainging in case of transfer learning. only supported backbone is inception_v3
        if transfer_model :
            logging.info("freezing model")
            self.optimizer = tf.keras.optimizers.Adam(inital_lr)
            model.get_layer('inception_v3').trainable = False
            model.compile(optimizer = self.optimizer)
            self.transfer_model = True
            self.frozen_steps = frozen_steps
            self.transfer_lr = transfer_lr
        else :
            self.transfer_model = False
            self.frozen_steps = frozen_steps
            self.transfer_lr = transfer_lr
            self.optimizer = tf.keras.optimizers.Adam(inital_lr)


        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.balanced_train_accuracy = BalancedAccuarcy(name = 'balanced_train_accuracy')
        self.confustion_matrix = ConfusionMatrix()
        self.f1score = tfa.metrics.F1Score(num_classes=2,average = 'macro')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.balanced_test_accuracy = BalancedAccuarcy(name = 'balanced_test_accuracy')
        self.test_confustion_matrix = ConfusionMatrix()
        self.test_f1score = tfa.metrics.F1Score(num_classes=2,average = 'macro')


        #model and dataseets
        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info

        #training log params
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

        # Checkpoint Manager
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
        self.manager = tf.train.CheckpointManager(
            checkpoint, directory=self.run_paths['path_ckpts_train'], max_to_keep=10)

        #loading an existing model and load the privious best accuracy else set best accuracy to 0
        if self.manager.latest_checkpoint and not self.transfer_model:
            checkpoint.restore(self.manager.latest_checkpoint)
            if os.path.exists(os.path.join(run_paths['path_model_id'],'state.json')):
                with open(os.path.join(run_paths['path_model_id'],'state.json'), 'r') as fp:
                    self.state = json.load(fp)
            else :
                self.state = {'best_validation_balanced_accuracy':0,'validation_balanced_accuracy':0}
            logging.info("loaded model from checkpoint")
        else :
            self.state = {'best_validation_balanced_accuracy':0,'validation_balanced_accuracy':0}
            logging.info("starting model from rnadom init")



    # @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            model_params = self.model.trainable_variables
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in model_params
                               if 'bias' not in v.name]) * 0.0000000
            loss = self.loss_object(labels, predictions)+lossL2
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        #update metrics
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        self.balanced_train_accuracy(labels,predictions)
        self.confustion_matrix(labels,predictions)
        self.f1score(tf.one_hot(labels,2),predictions)


    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)


        #update metrics
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        self.balanced_test_accuracy(labels,predictions)
        self.test_confustion_matrix(labels,predictions)
        self.test_f1score(tf.one_hot(labels,2),predictions)


    @gin.configurable
    def train(self,profiler_start_step=100,profiler_stop_step=200):
        '''this class creates a trainer object which is responsible for training in an expirement

        Parameters :
            profiler_start_step(int) : Indicates when to start the profiler
            profiler_stop_step(int): Indicates when to end the profiler
        
        Returns :
            (float) : Accuracy each loggin interval and at the end
        '''
        for idx, (images, labels,images_original) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images, labels)
            
            # uncomment for tensorboard profiler to improve the input pipelines and the trainging time , however here it's commented because it's not supported on the server runs

            # if step == profiler_start_step :
            #     options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3,
            #                                        python_tracer_level = 1,
            #                                        device_tracer_level = 1)
            #     tf.profiler.experimental.start(logdir = self.run_paths['summary'],options= options)
            # elif step == profiler_stop_step :
            #     tf.profiler.experimental.stop(save = True)

            #start fine tuning when the classification head is trained in case of transfer learning
            if self.transfer_model and step == self.frozen_steps :
                self.model.get_layer('inception_v3').trainable = True
                self.optimizer.learning_rate = self.transfer_lr
                self.model.compile(optimizer = self.optimizer)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()

                #visualizing
                visualize_next_batch = True
                for test_images, test_labels, test_images_original in self.ds_val:
                    self.test_step(test_images, test_labels)
                    if visualize_next_batch :
                        vis_images = visualize_integrated_gradients(self.model,test_images[0:2],test_labels[0:2],test_images_original[0:2])
                        visualize_next_batch = False

                #logging info
                template = 'Step {}, Loss: {}, Accuracy: {}, Balanced Accuracy: {}, F1 score : {}, Test Loss: {}, Test Accuracy: {}, Test Balanced Accuracy: {},Test F1 score : {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             (self.balanced_train_accuracy.result()) * 100,
                                             self.f1score.result(),
                                             self.test_loss.result(),
                                             self.test_accuracy.result() * 100,
                                             (self.balanced_test_accuracy.result() * 100),
                                             self.test_f1score.result()
                                             ))

                template_confustion = 'Confusion Matrix : \n {} \n Test Confusion Matrix : \n {}'
                logging.info(template_confustion.format(self.confustion_matrix.result(),self.test_confustion_matrix.result()))

                # Write summary to tensorboard
                tf.summary.scalar('Training Loss',self.train_loss.result(),step=step)
                tf.summary.scalar('Training Accuracy',self.train_accuracy.result(),step=step)
                tf.summary.scalar('Training Balanced Accuracy',(self.balanced_train_accuracy.result()) * 100,step=step)
                tf.summary.scalar('Training F1 Score',self.f1score.result(),step=step)

                tf.summary.scalar('Validation Loss',self.test_loss.result(),step=step)
                tf.summary.scalar('Validation Accuracy',self.test_accuracy.result(),step=step)
                tf.summary.scalar('Validation Balanced Accuracy',(self.balanced_test_accuracy.result()) * 100,step=step)
                tf.summary.scalar('Validation F1 Score',self.f1score.result(),step=step)

                #visualize image to tensorboard
                for i,image in enumerate(vis_images) :
                    tf.summary.image('val image {}'.format(i),image,step=step,)

                self.state['validation_balanced_accuracy'] = float((self.balanced_test_accuracy.result()).numpy() * 100)

                #keeping track of best balidation balanced accuracy
                if self.state['validation_balanced_accuracy'] > self.state['best_validation_balanced_accuracy'] :
                    self.state['best_validation_balanced_accuracy'] = self.state['validation_balanced_accuracy']
                    self.model.save(os.path.join(self.run_paths["path_ckpts_train"],"best_model"))
                    logging.info('best validation balanced accuracy : {}'.format(self.state['best_validation_balanced_accuracy']))
                    with open(os.path.join(self.run_paths['path_model_id'],'state.json'), 'w') as fp:
                        json.dump(self.state, fp)

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()
                self.balanced_train_accuracy.reset_states()
                self.confustion_matrix.reset_states()
                self.f1score.reset_states()

                # Reset test metrics
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()
                self.balanced_test_accuracy.reset_states()
                self.test_confustion_matrix.reset_states()
                self.test_f1score.reset_states()

                yield self.test_accuracy.result().numpy()

            #@checkpoint interval..
            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                self.manager.save()

            # Final training step.
            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                self.manager.save()
                self.model.save(self.run_paths["path_ckpts_train"])
                return self.test_accuracy.result().numpy()
