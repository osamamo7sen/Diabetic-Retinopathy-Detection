import os
import glob

import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

from input_pipeline.preprocessing import preprocess, augment
from input_pipeline.HAPT_TFR_dataset import make_HAPT_TFR_Dataset,make_HAPT_TFR_Dataset_test

@gin.configurable
def load(name, data_dir, cache_dir):
    if name == "idird":
        print(data_dir)
        logging.info(f"Preparing dataset {name}...")

        train_images = glob.glob(os.path.join(data_dir,'IDRID_dataset','images','train','*.jpg'))
        test_images = glob.glob(os.path.join(data_dir,'IDRID_dataset','images','test','*.jpg'))
        train_labels = pd.read_csv(os.path.join(data_dir,'IDRID_dataset','labels','train.csv'))['Retinopathy grade'].tolist()
        test_labels = pd.read_csv(os.path.join(data_dir,'IDRID_dataset','labels','test.csv'))['Retinopathy grade'].tolist()

        ds_train, ds_val, ds_test, ds_info = loadIDRID(train_images,train_labels,test_images,test_labels)


        logging.info(f"finished preparing dataset {name}...")
        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds_info_dict = dict() 
        ds_info_dict["image_shape"] = ds_info.features["image"].shape
        ds_info_dict['num_classes'] = ds_info.features["label"].num_classes
        ds_info_dict['train_num'] = ds_info.splits['train'].num_examples

        return prepare(ds_train, ds_val, ds_test, ds_info_dict)

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

    elif name == "HAPT_TFR":
        ds_train,ds_info_dict = make_HAPT_TFR_Dataset(os.path.join(data_dir,name,"train"))
        ds_val,_ = make_HAPT_TFR_Dataset_test(os.path.join(data_dir,name,"val"))
        ds_test,_ = make_HAPT_TFR_Dataset_test(os.path.join(data_dir,name,"test"))


        return prepare(ds_train, ds_val, ds_test, ds_info_dict,cache_dir)
    else :
        raise ValueError

@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info,cache_dir,batch_size,caching, schuffle_size = 256):
    preprocess_function = lambda timestamp, acc, gyro, activity: preprocess(timestamp, acc, gyro, activity)
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching == 2:
        if cache_dir != None :
            filename = os.path.join(cache_dir,'train_cache')
            ds_train = ds_train.cache(filename = filename)

    elif caching == 1 :
        ds_train = ds_train.cache()
    
    ds_train = ds_train.shuffle(schuffle_size)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_val = ds_val.batch(1)

    if caching == 2:
        if cache_dir != None :
            filename = os.path.join(cache_dir,'val_cache')
            ds_val = ds_val.cache(filename = filename)

    elif caching == 1 :
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(1)

    if caching == 2:
        if cache_dir != None :
            filename = os.path.join(cache_dir,'test_cache')
            ds_test = ds_test.cache(filename = filename)

    elif caching == 1 :
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info





def build_dataset(files, labels):
    # Create tf data set
    ds = tf.data.Dataset.from_tensor_slices((files, labels))  # Create data set of files and labels

    def _parse_func(filename,label):
        print(filename)
        image_string = tf.io.read_file(filename)  # Read the image
        # print(image_string)
        image_decoded = tf.io.decode_jpeg(image_string, channels=3)  # Decode the image and normalize it
        # print(image_decoded)
        image = image_decoded
        # print(image.shape)
        image = tf.image.resize(image,(224,224))
        label = tf.expand_dims(tf.cast(label, tf.float32), axis=-1)
        return image, label

    ds = ds.map(_parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


@gin.configurable
def loadIDRID(train_images,train_labels,test_images,test_labels,train_percentage) :
        train_labels = [0 if label < 2 else 1 for label in train_labels]
        test_labels = [0 if label < 2 else 1 for label in test_labels]
        split_index = len(train_images) * train_percentage // 100
        print(train_labels[split_index:])
        train_ds = build_dataset(train_images[0:split_index], train_labels[0:split_index])
        validation_ds = build_dataset(train_images[split_index:], train_labels[split_index:])
        test_ds = build_dataset(test_images, test_labels)
        ds_info_dict = dict()
        ds_info_dict["image_shape"] = (224,224,3)
        ds_info_dict["num_classes"] = 2
        ds_info_dict["train_num"] = split_index
        return train_ds,validation_ds,test_ds,ds_info_dict

