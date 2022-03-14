import gin
from absl import app, flags

from train import Trainer
from evaluation.eval import Evaluator
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import build_model
from dataset_tools.dataset_tool import prepare_HAPT
import tensorflow as tf

import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_boolean('prepare', False, 'Specify wether to prepare the HAPT dataset and consturct TFRecords')
flags.DEFINE_string('checkpoint', None , 'path of checkpoint to continue training from or evaluate')
flags.DEFINE_string('saved_model', None , 'path of tf.savedModel to evaluate or continue training from')

def main(argv):

    #load an existin trained model
    # error if multiple checkpoints provided
    if FLAGS.checkpoint is not None and FLAGS.saved_model is not None :
        print('error : only one of checkpoint or saved model can be provided during evaluation')
        exit()

    # generate folder structures from previous run if provided
    if FLAGS.checkpoint is not None :
        run_paths = utils_params.gen_run_folder(FLAGS.checkpoint)
    else :
        run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], 20)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())
    
    # check if run is to prepare dataset inform of tensorrecords
    if FLAGS.prepare:
        prepare_HAPT()

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(cache_dir = run_paths['cache_dir'])


    # model
    model = build_model(input_shape = ds_info['input_shape'])

    if FLAGS.saved_model is not None :
        model = tf.keras.models.load_model(FLAGS.saved_model)
        
    print("Model Summary")
    model.summary()
    print("End Model Summary",flush=True)
        
    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue
    else:
        Evaluator(model,
                 FLAGS.checkpoint,
                 ds_test,
                 ds_info,
                 run_paths,
                 ).eval()

if __name__ == "__main__":
    app.run(main)