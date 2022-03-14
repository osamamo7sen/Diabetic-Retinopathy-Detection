## tool which prepares datasets before training for each dataset a method should be implemented

import os
import gin
import logging
import glob
import numpy as np 
import tensorflow as tf
import logging

@gin.configurable
def prepare_HAPT(dataset_dir,out_dir):
    # check if dir exists
    if not os.path.exists(dataset_dir):
        logging.info("dataset directory doesn't exit")
        logging.info("exiting...")
        exit(0)

    # load labels for all experiements into numpy array
    labels_path = os.path.join(dataset_dir,"RawData","labels.txt")
    with open(labels_path) as labels_file:
        labels_all_array = np.array([[int(digit) for digit in line.split()] for line in labels_file],dtype=int)

    for user in range(1,31):
        
        # get the user's expirements numbers 
        expirement_acc = os.path.join(dataset_dir,"RawData","acc_exp*_user{:02d}.txt".format(user))
        expirements_acc = glob.glob(expirement_acc)
        expirement_numbers = [os.path.split(file_path)[1][7:9] for file_path in expirements_acc]

        for expirement_number in expirement_numbers :
            # load the acc and gyro raw data files
            acc_file_name = os.path.join(dataset_dir,"RawData","acc_exp{}_user{:02d}.txt".format(expirement_number,user))
            gyro_file_name = os.path.join(dataset_dir,"RawData","gyro_exp{}_user{:02d}.txt".format(expirement_number,user))

            with open(acc_file_name) as acc_file:
                acc_array = np.array([[float(digit) for digit in line.split()] for line in acc_file])

            with open(gyro_file_name) as gyro_file: 
                gyro_array = np.array([[float(digit) for digit in line.split()] for line in gyro_file])

            # unfold the labels array resulting in a label for each data row in acc and gyro

            labels_folded = labels_all_array[labels_all_array[:,0]==int(expirement_number)]
            labels_array = np.zeros(len(acc_array),dtype=int)

            for label_row in labels_folded :
                labels_array[ label_row[3]-1 : label_row[4]-1 ]=label_row[2]

            # normalize each col of acc and gyro alone to remove user biases and errors due to placement of the sensors

            acc_array_normalized = (acc_array - np.mean(acc_array,axis=0))/np.std(acc_array,axis=0)
            gyro_array_normalized = (gyro_array- np.mean(gyro_array,axis=0)/np.std(gyro_array,axis=0))

            # name tfr with the same name as experiement and user and write arrays to tfr with added timestamp which is the same as index since our data in still ordered
            tfr_name = os.path.split(acc_file_name)[1][4:16]
            # depending on the user number subfolder is selected for train test eval
            if user < 22 :
                subfolder = 'train'
            elif user < 28 :
                subfolder = 'test'
            else :
                subfolder = 'val' 
            
            write_exp_TFRecord(np.arange(0,len(acc_array_normalized),dtype=int),acc_array_normalized,gyro_array_normalized,labels_array,tfr_name,
                                os.path.join(out_dir,subfolder))


def write_exp_TFRecord(timestamps,acc,gyro,labels,name,out_dir) :
    if not os.path.isdir(out_dir) :
        logging.info("creating output directory : {}".format(out_dir))
        os.makedirs(out_dir)
    file_path = os.path.join(out_dir,"{}.tfrecords".format(name))
    logging.info("writing file : "+name)
    with tf.io.TFRecordWriter(file_path) as writer :
        for  timestamp,acc_point,gyro_point,label in zip(timestamps,acc,gyro,labels):
            serialized_example = serialize_HAPT_example(timestamp,tf.io.serialize_tensor(acc_point),tf.io.serialize_tensor(gyro_point),label)
            writer.write(serialized_example)

    return


def serialize_HAPT_example(feature0, feature1, feature2, feature3):
    feature = {
      'timestap': _int64_feature(feature0),
      'acc': _bytes_feature(feature1),
      'gyro': _bytes_feature(feature2),
      'activity': _int64_feature(feature3),
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))




            
            


         

