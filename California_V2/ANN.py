import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import size
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
from tensorflow.keras import layers, initializers, regularizers
# from tcn.tcn import TCN

def CNN(x, n_out, layer, layer_size, rate, weight_decay):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(x.shape[1], x.shape[2])))
    for i in np.arange(layer-1):
        units = int(layer_size/(2**i))
        model.add(layers.Conv1D(
            filters=units, kernel_size=3, strides=2, 
            # kernel_initializer=initializers.he_normal(),
            # kernel_regularizer=regularizers.l2(weight_decay), 
            activation='relu', data_format='channels_first'))
        # model.add(layers.MaxPool1D(
        #     pool_size=2, strides=2, 
        #     data_format='channels_first'))
    model.add(layers.GlobalMaxPooling1D())
    # model.add(layers.Flatten())
    # model.add(layers.Dropout(rate))
    model.add(layers.Dense(n_out, 
        # kernel_initializer=initializers.he_normal(),
        activation='relu'
        # activation='sigmoid'
        ))
    return model

def LSTM(x, n_out, layer, layer_size, rate, weight_decay):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(x.shape[1],x.shape[2])))
    for i in np.arange(layer-1):
        if i==layer-1: return_sequences = False
        else: return_sequences = True
        units = int(layer_size/(2**i))
        model.add(layers.LSTM(units=units,
            activation='relu',
            # kernel_initializer=initializers.he_normal(),
            # kernel_initializer=initializers.glorot_normal(),
            # kernel_regularizer=regularizers.l2(weight_decay), 
            return_sequences=return_sequences))
    model.add(layers.Dense(units=n_out,         
        # kernel_initializer=initializers.he_normal(),
        # kernel_initializer=initializers.glorot_normal(),
        # kernel_regularizer=regularizers.l2(weight_decay), 
        activation='relu'))
    return model

def lstm_cnn(x, n_out, layer, layer_size, rate, weight_decay):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(x.shape[1], x.shape[2])))
    if layer==3:
        model.add(layers.LSTM(layer_size, 
            activation='relu', 
            return_sequences=True))
        model.add(layers.Conv1D(filters=int(layer_size/2), 
            kernel_size=3, strides=2, 
            activation='relu', 
            data_format='channels_first'))
    elif layer==4:
        model.add(layers.LSTM(layer_size, 
            activation='relu', 
            return_sequences=True))
        model.add(layers.LSTM(int(layer_size/2), 
            activation='relu', 
            return_sequences=True))
        model.add(layers.Conv1D(filters=int(layer_size/4), 
            kernel_size=3, strides=2, 
            activation='relu', 
            data_format='channels_first'))
    elif layer==5:
        model.add(layers.LSTM(layer_size, 
            activation='relu', 
            return_sequences=True))
        model.add(layers.LSTM(int(layer_size/2), 
            activation='relu', 
            return_sequences=True))
        model.add(layers.LSTM(int(layer_size/4), 
            activation='relu', 
            return_sequences=True))
        model.add(layers.Conv1D(filters=int(layer_size/6), 
            kernel_size=3, strides=2, 
            activation='relu', 
            data_format='channels_first'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(n_out, 
        activation='relu'))
    return model





def cnn_lstm(x, n_out, layer, layer_size, rate, weight_decay):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(x.shape[1], x.shape[2])))
    layer_size = 128
    model.add(layers.Conv1D(filters=int(layer_size/1), 
        kernel_size=2, strides=2, 
        activation='relu', 
        data_format='channels_first'))
    model.add(layers.LSTM(int(layer_size/2), 
        activation='relu', 
        return_sequences=True))
    model.add(layers.LSTM(int(layer_size/4), 
        activation='relu', 
        return_sequences=True))
    model.add(layers.LSTM(int(layer_size/8), 
        activation='relu', 
        return_sequences=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(n_out, 
        activation='relu'))
    return model

# def lstm_cnn(x, n_out, layer, layer_size, rate, weight_decay):
#     model = tf.keras.Sequential()
#     model.add(layers.Input(shape=(x.shape[1], x.shape[2])))
#     model.add(layers.LSTM(layer_size, 
#         activation='relu', 
#         return_sequences=True))
#     model.add(layers.LSTM(int(layer_size/2), 
#         activation='relu', 
#         return_sequences=True))
#     model.add(layers.Conv1D(filters=int(layer_size/4), 
#         kernel_size=3, strides=2, 
#         activation='relu', 
#         data_format='channels_first'))
#     model.add(layers.GlobalMaxPooling1D())
#     model.add(layers.Dense(n_out, 
#         activation='relu'))
#     return model

def tcn(x, n_out, layer, rate, layer_size, weight_decay):
    inputs = layers.Input(shape=(x.shape[1], x.shape[2]))
    t = TCN(return_sequences=False, nb_filters=32, kernel_size=5, 
        activation='relu', 
        # kernel_initializer=initializers.he_normal(),
        dilations=[2**i for i in range(9)])(inputs)
    # t = TCN(return_sequences=False, nb_filters=16, kernel_size=5, 
    #     activation='relu', 
    #     # kernel_initializer=initializers.he_normal(),
    #     dilations=[2**i for i in range(10)])(t)
    outputs = layers.Dense(n_out,
        activation='relu', 
        # kernel_initializer=initializers.he_normal(),
        )(t)
    tcn_model = tf.keras.Model(inputs, outputs) 
    return tcn_model

def tcn_lstm(x, n_out, layer, rate, layer_size, weight_decay):
    inputs = layers.Input(shape=(x.shape[1], x.shape[2]))
    t = TCN(return_sequences=True, nb_filters=16, kernel_size=5, 
        activation='relu',
        kernel_initializer=initializers.he_normal(),
        dilations=[2**i for i in range(10)])(inputs)
    t = layers.LSTM(132, 
        activation='relu', 
        kernel_initializer=initializers.he_normal(),
        return_sequences=False)(t)
    outputs = layers.Dense(n_out, 
        kernel_initializer=initializers.he_normal(),
        activation='relu')(t)
    tcn_model = tf.keras.Model(inputs, outputs) 
    return tcn_model

def tcn_cnn(x, n_out, layer, rate, layer_size, weight_decay):
    inputs = layers.Input(shape=(x.shape[1], x.shape[2]))
    t = TCN(return_sequences=True, nb_filters=16, kernel_size=5, 
        activation='relu',
        kernel_initializer=initializers.he_normal(),
        dilations=[2**i for i in range(10)])(inputs)
    t = layers.Conv1D(filters=128, kernel_size=3, strides=2, 
        activation='relu', 
        kernel_initializer=initializers.he_normal(),
        data_format='channels_first')(t)
    # t = layers.MaxPooling1D(pool_size=2, strides=2, 
    #     data_format='channels_first')(t)
    t = layers.GlobalMaxPooling1D()(t)
    # model.add(layers.Flatten())
    outputs = layers.Dense(n_out, 
        kernel_initializer=initializers.he_normal(),
        activation='relu')(t)
    tcn_model = tf.keras.Model(inputs, outputs) 
    return tcn_model


def tcn_cnn_lstm(x, n_out, layer, layer_size, rate, weight_decay):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(x.shape[1], x.shape[2])))
    model.add(TCN(return_sequences=True, nb_filters=16, kernel_size=5, 
        activation='relu',
        kernel_initializer=initializers.he_normal(),
        dilations=[2**i for i in range(10)]))
    model.add(layers.Conv1D(filters=132, kernel_size=2, strides=2, 
        activation='relu', 
        kernel_initializer=initializers.he_normal(),
        data_format='channels_first'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, 
        data_format='channels_first'))
    # model.add(layers.MaxPool1D(pool_size=2, strides=2, 
    #     data_format='channels_first'))
    model.add(layers.LSTM(132, 
        activation='relu', 
        kernel_initializer=initializers.he_normal(),
        return_sequences=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(n_out, 
        kernel_initializer=initializers.he_normal(),
        activation='relu'))
    return model

def tcn_lstm_cnn(x, n_out, layer, layer_size, rate, weight_decay):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(x.shape[1], x.shape[2])))
    model.add(TCN(return_sequences=True, nb_filters=16, kernel_size=5, 
        activation='relu',
        kernel_initializer=initializers.he_normal(),
        dilations=[2**i for i in range(10)]))
    model.add(layers.LSTM(132, 
        activation='relu', 
        kernel_initializer=initializers.he_normal(),
        return_sequences=True))
    model.add(layers.Conv1D(filters=132, kernel_size=2, strides=2, 
        activation='relu', 
        kernel_initializer=initializers.he_normal(),
        data_format='channels_first'))
    # model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, 
        data_format='channels_first'))
    model.add(layers.GlobalMaxPooling1D())
    # model.add(layers.Flatten())
    model.add(layers.Dense(n_out, 
        kernel_initializer=initializers.he_normal(),
        activation='relu'))
    return model