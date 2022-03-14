import os
import glob
import random

import gin
import logging
import numpy as np
import tensorflow as tf
from tensorflow._api.v2 import image
import tensorflow_datasets as tfds
import pandas as pd

from input_pipeline.btgraham_preprocessing import load_btgraham
from input_pipeline.preprocessing import post_augment_preprocess, preprocess, augment

@gin.configurable
def load(name, data_dir):
    #IDRID dataset
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        #reading data from directory
        train_images = glob.glob(os.path.join(data_dir,'IDRID_dataset','images','train','*.jpg'))
        train_images = sorted(train_images)
        train_labels = pd.read_csv(os.path.join(data_dir,'IDRID_dataset','labels','train.csv'))['Retinopathy grade'].tolist()
        test_images = glob.glob(os.path.join(data_dir,'IDRID_dataset','images','test','*.jpg'))
        test_images = sorted(test_images)
        test_labels = pd.read_csv(os.path.join(data_dir,'IDRID_dataset','labels','test.csv'))['Retinopathy grade'].tolist()
        # Retinopathy grade

        #creating tensorflow datasets
        ds_train, ds_val, ds_test, ds_info = loadIDRID(train_images,train_labels,test_images,test_labels)

        #visualize a sample
        # iterator = iter(ds_train)
        # sample_data = iterator.__next__()
        # sample_image = sample_data[0]
        # sample_label = sample_data[1]
        # print(sample_image.shape)
        # print(sample_label)



        logging.info(f"finished preparing dataset {name}...")
        return prepare(ds_train, ds_val, ds_test, ds_info)
    #EYEPACS Dataset

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=os.path.join(data_dir,'tensorflow_datasets')
        )

        #preprocessing
        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']
        #creating datasets
        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds_info_dict = dict() 
        ds_info_dict["image_shape"] = ds_info.features["image"].shape
        ds_info_dict['num_classes'] = ds_info.features["label"].num_classes
        ds_info_dict['train_num'] = ds_info.splits['train'].num_examples

        return prepare(ds_train, ds_val, ds_test, ds_info_dict)

    #mnist dataset
    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=os.path.join(data_dir,'tensorflow_datasets')
        )

        ds_info_dict = dict() 
        ds_info_dict["image_shape"] = ds_info.features["image"].shape
        ds_info_dict['num_classes'] = ds_info.features["label"].num_classes
        ds_info_dict['train_num'] = ds_info.splits['train'].num_examples

        return prepare(ds_train, ds_val, ds_test, ds_info_dict)

    else:
        raise ValueError

def count(features, labels):

    #count the number of instances for each class in case of binary classification for IDRID dataset
    counts = dict()
    labels = np.array(labels)

    class_1 = labels >= 2
    class_1 = tf.cast(class_1, tf.int32)

    class_0 = labels < 2
    class_0 = tf.cast(class_0, tf.int32)

    counts['class_0'] = tf.reduce_sum(class_0).numpy()
    counts['class_1'] = tf.reduce_sum(class_1).numpy()

    return counts

@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching,img_height, img_width,balance_dataset = True,binary_classification = True):
    # Prepare training dataset
    # ds_test = ds_train
    if binary_classification:
        ds_info['num_classes'] = 2
    #Apply preprocessing
    preprocess_lambda = lambda image,label: preprocess(image,label,img_height,img_width,binary_classification)
    # augment_batch_lambda = lambda images,labels: augment_batch(images,labels)
    # augment_unbatch_lambda = lambda image,label: augment_image(image,label,img_height,img_width)
    ds_train = ds_train.map(
        preprocess_lambda, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #caching
    if caching:
        ds_train = ds_train.cache()

    #balancing the dataset in case it is not balanced to oversample the less frequent class
    if balance_dataset:
        ds_train = make_dataset_balanced(ds_train)
    else :
        ds_train = ds_train.repeat(-1)

    #Augmenting the dataset
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #post-augmentation processing for the images
    ds_train = ds_train.map(post_augment_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #Shuffeling the dataset
    ds_train = ds_train.shuffle(ds_info['train_num'] // 10)
    #Batching the dataset
    ds_train = ds_train.batch(batch_size)

    # ds_train = ds_train.map(augment_batch_lambda, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds_train = ds_train.unbatch()
    # ds_train = ds_train.map(augment_unbatch_lambda,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)

    #prefetching
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess_lambda, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(post_augment_preprocess,  num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess_lambda, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(post_augment_preprocess,  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    from tensorflow_datasets.core.features.image_feature import Image

    ds_info["image_shape"] = (img_width,img_height,ds_info["image_shape"][2])
    return ds_train, ds_val, ds_test, ds_info



def make_dataset_balanced(ds):
    #Balancing the dataset to account for classes with much less samples than the other classes.

    negative_filter = lambda features, label,orig_feat: tf.reduce_all(label==0)
    positive_filter = lambda features, label,orig_feat: tf.reduce_all(label==1)
    negative_ds = (
        ds
        .filter(negative_filter)
        .repeat(-1))
    positive_ds = (
        ds
        .filter(positive_filter)
        .repeat(-1))

    balanced_ds = tf.data.experimental.sample_from_datasets(
    [negative_ds, positive_ds], [0.5, 0.5])

    return balanced_ds





def build_dataset(files, labels,btgraham):
    # Create tf data set

    if btgraham :
        #Applying Ben Graham Preprocessing for the images
        images = [load_btgraham(file) for file in files]
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        ds = ds.cache()

    else :
        ds = tf.data.Dataset.from_tensor_slices((files, labels))  # Create dataset of files and labels

        def _parse_func(filename,label):
            image_string = tf.io.read_file(filename)  # Read the image
            image_decoded = tf.io.decode_jpeg(image_string, channels=3)  # Decode the image and normalize it
            image = tf.image.resize(image_decoded,(256,256)) # Resizing the images
            return image, label

        ds = ds.map(_parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds


@gin.configurable
def loadIDRID(train_images,train_labels,test_images,test_labels,train_percentage,btgraham = False) :
        #Splitting the training data to training and validation
        split_index = len(train_images) * train_percentage // 100
        training_data = list(zip(train_images, train_labels))

        #shuffeling the data to make sure the validation set is representative of the classes
        random.shuffle(training_data)
        train_images, train_labels = zip(*training_data)

        train_images = list(train_images)
        train_labels = list(train_labels)

        #Validation set
        validation_ds = build_dataset(train_images[split_index:], train_labels[split_index:],btgraham)

        #Training set
        train_images = train_images[0:split_index]
        train_labels = train_labels[0:split_index]
        train_ds = build_dataset(train_images,train_labels,btgraham)

        #Test set
        test_ds = build_dataset(test_images, test_labels,btgraham)

        #Creating Dictionary for the tensorflow dataset
        ds_info_dict = dict()
        ds_info_dict["image_shape"] = (256,256,3)
        ds_info_dict["num_classes"] = 2
        ds_info_dict["train_num"] = split_index
        return train_ds,validation_ds,test_ds,ds_info_dict

