import os
import glob
import tensorflow as tf
import gin 
import logging
import tensorflow as tf
import numpy as np

@gin.configurable
def make_HAPT_TFR_Dataset(data_dir,window_size,shift,warm_up):
    tfrecords_paths = glob.glob(os.path.join(data_dir,"*.tfrecords"))
    dataset = tf.data.Dataset.from_tensor_slices(tfrecords_paths)


    def pasre_tfrecord_file(filename):

        file_dataset = tf.data.TFRecordDataset(filename).map(parse_tfrecord,num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def parse_window(timestamp,acc,gyro,activity):
            timestamp = tf.data.experimental.get_single_element(timestamp.batch(window_size))
            acc = tf.data.experimental.get_single_element(acc.batch(window_size))
            gyro = tf.data.experimental.get_single_element(gyro.batch(window_size))
            activity = tf.data.experimental.get_single_element(activity.batch(window_size))
            activity = tf.concat([tf.zeros_like(activity[0:warm_up],dtype = tf.int64),activity[warm_up:]],axis=0)
            return timestamp,acc,gyro,activity

        file_dataset = file_dataset.window(window_size,shift,1,True).map(parse_window)#,num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return file_dataset

    tfrecords_full_dataset = dataset.interleave(pasre_tfrecord_file,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_info = {'window_size':window_size,'n_classes' : 12,'input_shape' : (window_size,6) }
    

    return tfrecords_full_dataset,ds_info

@gin.configurable
def make_HAPT_TFR_Dataset_test(data_dir,window_size = 300000,warm_up = 50 ):
    tfrecords_paths = glob.glob(os.path.join(data_dir,"*.tfrecords"))
    dataset = tf.data.Dataset.from_tensor_slices(tfrecords_paths)


    def pasre_tfrecord_file(filename):

        file_dataset = tf.data.TFRecordDataset(filename).map(parse_tfrecord,num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def parse_window(timestamp,acc,gyro,activity):
            timestamp = tf.data.experimental.get_single_element(timestamp.batch(window_size))
            acc = tf.data.experimental.get_single_element(acc.batch(window_size))
            gyro = tf.data.experimental.get_single_element(gyro.batch(window_size))
            activity = tf.data.experimental.get_single_element(activity.batch(window_size))
            activity = tf.concat([tf.zeros_like(activity[0:warm_up],dtype = tf.int64),activity[warm_up:]],axis=0)
            return timestamp,acc,gyro,activity

        file_dataset = file_dataset.window(window_size,window_size,1,False).map(parse_window)#,num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return file_dataset

    tfrecords_full_dataset = dataset.interleave(pasre_tfrecord_file,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_info = {'window_size':window_size,'n_classes' : 12,'input_shape' : (window_size,6) }
    

    return tfrecords_full_dataset,ds_info

@gin.configurable
def make_HAPT_TFR_Dataset2(data_dir,mode,window_size,shift,cache_dir = None):
    tfrecords_paths = glob.glob(os.path.join(data_dir,"*.tfrecords"))
    tfrecords_full_dataset = None
    for tfrecord_path in tfrecords_paths :

        path_dataset = tf.data.TFRecordDataset(tfrecord_path)
        path_dataset = path_dataset.map(parse_tfrecord,num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def parse_window(timestamp,acc,gyro,activity):
            timestamp = tf.data.experimental.get_single_element(timestamp.batch(window_size))
            acc = tf.data.experimental.get_single_element(acc.batch(window_size))
            gyro = tf.data.experimental.get_single_element(gyro.batch(window_size))
            activity = tf.data.experimental.get_single_element(activity.batch(window_size))
            return timestamp,acc,gyro,activity
            

        path_dataset = path_dataset.window(window_size,shift,1,True).map(parse_window)#,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if tfrecords_full_dataset == None :
            tfrecords_full_dataset = path_dataset
        else :
            tfrecords_full_dataset = tfrecords_full_dataset.concatenate(path_dataset)
    if cache_dir != None :
        # pass
        filename = os.path.join(cache_dir,mode+'_cahce')
        print(filename)
        # tfrecords_full_dataset = tfrecords_full_dataset.cache(filename = filename)
        # tfrecords_full_dataset = tfrecords_full_dataset.cache()

    tfrecords_full_dataset.shuffle(buffer_size = 256, reshuffle_each_iteration = True)
    tfrecords_full_dataset.batch(batch_size = 32)
    return tfrecords_full_dataset


def parse_tfrecord(serialized_example):
    feature_description = {
        'timestap': tf.io.FixedLenFeature((), tf.int64),
        'acc': tf.io.FixedLenFeature((), tf.string),
        'gyro': tf.io.FixedLenFeature((), tf.string),
        'activity': tf.io.FixedLenFeature((), tf.int64),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    timestamp = example['timestap']
    acc = tf.io.parse_tensor(example['acc'], out_type = tf.float64)
    gyro = tf.io.parse_tensor(example['gyro'], out_type = tf.float64)
    activity = example['activity']
  
    return timestamp, acc, gyro, activity



