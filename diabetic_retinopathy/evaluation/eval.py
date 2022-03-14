import tensorflow as tf
import tensorflow_addons as tfa
from evaluation.metrics import BalancedAccuarcy, ConfusionMatrix
import logging

#import metrics


class Evaluator():
    def __init__(self,model, checkpoint, ds_test, ds_info, run_paths,saved_model) :
        '''this class creates a trainer object which is responsible for training in an expirement

        Parameters :
            model(tf.Keras.Model) : model to be trained of type tf.Keras.Model
            checkpoint(str) : path of the run directory incase of evaluating from checkpoint. it has no direct use and can be used only as a sanity check
            ds_test(tf.Data.Dataset): test dataset of type tf.Data.Dataset
            ds_info(dict): a dict various information related to the dataset like image sizes
            run_paths(dict): a dict containing the paths related to the run like the ckpt dir and tensorboard summary dir
            saved_model(str): path of model of type saved_model to evaluate if provided

        Returns :
            (int) : accuracy of the model at each logging interval and at the end of thr training
        '''

        if saved_model != None :
            #load saved model
            self.model = tf.saved_model.load(saved_model)
        else :
            #evaluate current model
            self.model = model
            self.optimizer = tf.keras.optimizers.Adam()
            checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
            manager = tf.train.CheckpointManager(
                checkpoint, directory=run_paths['path_ckpts_train'], max_to_keep=10)
            checkpoint.restore(manager.latest_checkpoint)
            logging.info("loaded model from checkpoint")

        #loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # Metrics
        self.eval_loss = tf.keras.metrics.Mean(name='train_loss')
        self.eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.eval_balanced_accuracy = BalancedAccuarcy(name = 'balanced_train_accuracy')
        self.eval_confustion_matrix = ConfusionMatrix()
        self.eval_f1score = tfa.metrics.F1Score(num_classes=2,average = 'macro')

        self.ds_test = ds_test


    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        #update metrics
        self.eval_loss(t_loss)
        self.eval_accuracy(labels, predictions)
        self.eval_balanced_accuracy(labels,predictions)
        self.eval_confustion_matrix(labels,predictions)
        self.eval_f1score(tf.one_hot(labels,2),predictions)

    def eval(self) :

        for test_images, test_labels, test_images_original in self.ds_test:

            self.test_step(test_images, test_labels)
        #logging info
        template = 'Evaluation :- Loss: {}, Accuracy: {}, Balanced Accuracy: {}, F1 score : {}'
        logging.info(template.format(self.eval_loss.result(),
                                    self.eval_accuracy.result() * 100,
                                    tf.reduce_sum(self.eval_balanced_accuracy.result()) * 100,
                                    self.eval_f1score.result()
                                    ))

        template_confustion = 'Evaluation Confusion Matrix : \n {}'
        logging.info(template_confustion.format(self.eval_confustion_matrix.result()))


        # Reset eval metrics
        self.eval_loss.reset_states()
        self.eval_accuracy.reset_states()
        self.eval_balanced_accuracy.reset_states()
        self.eval_confustion_matrix.reset_states()
        self.eval_f1score.reset_states()

