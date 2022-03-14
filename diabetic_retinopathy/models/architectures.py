from input_pipeline.preprocessing import preprocess
import gin
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.backend import dropout
from models.layers import vgg_block, residual_block
from models.inception_with_squeeze_and_ignore import inception_with_squeeze_and_ignore

@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)
    out = vgg_block(inputs, base_filters)
    for i in range(2, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')



@gin.configurable
def resnet(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a ResNet architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """
    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)
    out = inputs
    out = residual_block(out, base_filters)
    for i in range(2, n_blocks):
        out = residual_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dense(dense_units // 2, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='resnet')

@gin.configurable
def mobilenetV2(input_shape, n_classes,dropout_rate):
    """Defines a MobileNetV2 architecture.

        Parameters:
            input_shape (tuple: 3): input shape of the neural network
            n_classes (int): number of classes, corresponding to the number of output neurons
            dropout_rate (float): dropout rate

        Returns:
            (keras.Model): keras model object
        """

    inputs = tf.keras.Input(input_shape)
    IMG_SHAPE = input_shape
    preprocess =  tf.keras.applications.mobilenet.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(n_classes)
    

    out = preprocess(inputs)
    out = base_model(out)
    out = global_average_layer(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    out = prediction_layer(out)
    return tf.keras.Model(inputs=inputs, outputs=out, name='mobilenetv2_frozen_backbone')

@gin.configurable
def kaggle_challange_model(input_shape):
    """Defines a pretrained model.

        Parameters:
            input_shape (tuple: 3): input shape of the neural network

        Returns:
            (keras.Model): keras model object
        """
    inputs = tf.keras.Input(input_shape)    
    base_model = tf.keras.models.load_model('./pretrained_models/full_retina_model.h5')
    base_model.trainable = False
    out = base_model(inputs)
    return tf.keras.Model(inputs=inputs, outputs=out, name='kaggle_challange_model')




@gin.configurable
def build_model(model_name,input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Creates a model

        Parameters:
            model_name (string): model name to be created
            input_shape (tuple: 3): input shape of the neural network
            n_classes (int): number of classes, corresponding to the number of output neurons
            base_filters (int): number of base filters, which are doubled for every VGG or residual block
            n_blocks (int): number of blocks in case the model consists of predefined blocks
            dense_units (int): number of dense units
            dropout_rate (float): dropout rate

        Returns:
            (keras.Model): keras model object
        """
    if model_name =='vgg_like' :
        return vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate)
    elif model_name == 'resnet' :
        return resnet(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate)
    elif model_name == 'mobilenetV2' :
        return mobilenetV2(input_shape, n_classes, dropout_rate)
    elif model_name == 'kaggle_challange_model' :
        return kaggle_challange_model(input_shape)
    elif model_name == 'inception_with_squeeze_and_ignore' :
        return  inception_with_squeeze_and_ignore(input_shape,n_classes)
    else :
        print("Undefined model")
        exit(0)




