import gin
import logging
from absl import app, flags

from train import Trainer
from evaluation.eval import Evaluator
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import build_model
import tensorflow as tf
import matplotlib.pyplot as plt
from input_pipeline.datasets import load

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('checkpoint', None , 'path of checkpoint to continue training from or evaluate')
flags.DEFINE_string('saved_model', None , 'path of tf.savedModel to evaluate')
flags.DEFINE_string('transfer_model', None , 'path of tf.savedModel to evaluate')


def main(argv):

    # check that no checkpoint and saved model are provided together for evaluation
    if not FLAGS.train and FLAGS.checkpoint is not None and FLAGS.saved_model is not None :
        logging.info('error : only one of checkpoint or saved model can be provided during evaluation' )

    
    # generate folder structures from checpoint if provided else a new folder structure
    if FLAGS.checkpoint is not None :
        run_paths = utils_params.gen_run_folder(FLAGS.checkpoint)
    else :
        run_paths = utils_params.gen_run_folder()


    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup input pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()


    #building models
    model = build_model(input_shape=ds_info['image_shape'], n_classes=ds_info['num_classes'])

    #model summary
    model.summary()
    print('finished model summary',flush=True)


    if FLAGS.train and FLAGS.transfer_model is not None :
        #training with transfer learning
        model = tf.keras.models.load_model(FLAGS.transfer_model)
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths,transfer_model = True)
        for _ in trainer.train():
            continue

    elif FLAGS.train:
        #training base model
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue
    else:
        #Evaluate an existing model
        Evaluator(model,
                 FLAGS.checkpoint,
                 ds_test,
                 ds_info,
                 run_paths,
                 FLAGS.saved_model).eval()

if __name__ == "__main__":
    app.run(main)