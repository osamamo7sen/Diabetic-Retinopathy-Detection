import gin
import tensorflow as tf
import tensorflow.keras as k

tf.keras.backend.set_floatx('float64')
from models.layers import vgg_block


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
def build_model(model_name, n_outputs,input_shape):
    if model_name == 'lstm':
        model = lstm(n_outputs,input_shape)
    elif model_name == 'lstm_double':
        model = lstm_double(n_outputs,input_shape)
    elif model_name == 'cnn':
        model = cnn(n_outputs,input_shape)
    elif model_name == 'gru_double' :
        model = gru_double(n_outputs,input_shape)
    elif model_name == 'cnn_lstm':
        model = cnn_lstm(n_outputs,input_shape)
    else :
        print("model {} is not defined".format(model_name))
    
    return model


@gin.configurable
def lstm(n_outputs,input_shape):
    model = k.models.Sequential()
    model.add(k.layers.LSTM(100, return_sequences=True,input_shape = input_shape))
    model.add(k.layers.Dense(100, activation='relu'))
    model.add(k.layers.Dense(n_outputs))
    return model


@gin.configurable
def lstm_double(n_outputs,input_shape,units = 100,dropout = 0.2):
    model = k.models.Sequential()
    model.add(k.layers.Input(input_shape))
    model.add(k.layers.LSTM(units, return_sequences=True))
    model.add(k.layers.Dropout(dropout))
    model.add(k.layers.LSTM(2*units, return_sequences=True))
    model.add(k.layers.Dropout(dropout))
    model.add(k.layers.Dense(2*units, activation='relu'))
    model.add(k.layers.Dropout(dropout))
    model.add(k.layers.Dense(32, activation='relu'))
    model.add(k.layers.Dropout(dropout))
    model.add(k.layers.Dense(n_outputs))
    return model


@gin.configurable
def cnn(n_outputs,input_shape):
    model = k.models.Sequential()
    model.add(k.layers.Reshape((input_shape+(1,)),input_shape = input_shape))
    model.add(k.layers.Conv2D(64, kernel_size=(5, 1), padding='same'))
    model.add(k.layers.Activation("relu"))
    model.add(k.layers.Conv2D(64, kernel_size=(5, 1), padding='same'))
    model.add(k.layers.Activation("relu"))
    model.add(k.layers.Conv2D(64, kernel_size=(5, 1), padding='same'))
    model.add(k.layers.Activation("relu"))
    model.add(k.layers.Conv2D(64, kernel_size=(5, 1), padding='same'))
    model.add(k.layers.Activation("relu"))
    model.add(k.layers.Reshape((250, 6 * 64)))
    model.add(k.layers.LSTM(128, activation="tanh", return_sequences=True))
    model.add(k.layers.Dropout(0.5, seed=0))
    model.add(k.layers.LSTM(128, activation="tanh", return_sequences=True))
    model.add(k.layers.Dropout(0.5, seed=1))
    model.add(k.layers.Dense(n_outputs))
    model.add(k.layers.Activation("softmax"))
    return model

@gin.configurable
def gru_double(n_outputs,input_shape,unit=16,dropout_percent = 0.2):
    model = k.models.Sequential()
    model.add(tf.keras.layers.GRU(unit * 2, dropout=dropout_percent, return_sequences=True,input_shape = input_shape))
    model.add(tf.keras.layers.GRU(unit, dropout=dropout_percent, return_sequences=True))
    model.add(tf.keras.layers.Dense(unit, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_percent))
    model.add(tf.keras.layers.Dense(n_outputs))
    return model

@gin.configurable
def cnn_lstm(n_outputs,input_shape):
    model = k.models.Sequential()
    model.add(k.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=-1),input_shape = input_shape))

    # model.add(k.layers.Reshape((input_shape+(1,)),input_shape = input_shape))
    model.add(k.layers.Conv2D(4, kernel_size=(21, 1), padding='same'))
    model.add(k.layers.Activation("relu"))
    model.add(k.layers.Conv2D(4, kernel_size=(21, 1), padding='same'))
    model.add(k.layers.Activation("relu"))
    model.add(k.layers.Conv2D(4, kernel_size=(21, 1), padding='same'))
    model.add(k.layers.Activation("relu"))
    model.add(k.layers.Conv2D(4, kernel_size=(21, 1), padding='same'))
    model.add(k.layers.Activation("relu"))
    model.add(k.layers.Reshape((-1, 6 * 4)))
    model.add(k.layers.LSTM(32, activation="tanh", return_sequences=True))
    model.add(k.layers.Dropout(0.5, seed=0))
    model.add(k.layers.LSTM(16, activation="tanh", return_sequences=True))
    model.add(k.layers.Dropout(0.5, seed=1))
    model.add(k.layers.Dense(n_outputs))
    return model