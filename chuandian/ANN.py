import time
import numpy as np
import tensorflow as tf
import config

np = config.np
tf = config.tf
layers = config.layers
layer_size = config.layer_size
rate = config.rate
weight = config.weight
batch_size = config.batch_size
initializers = config.initializers
regularizers = config.regularizers

def stateless_lstm(x, output_node, layer=2, layer_size=128, rate=0.20, weight=1e-6):
    model = tf.keras.Sequential()
    for i in np.arange(1, layer+1, 1):
        if i == layer: 
            model.add(layers.Dense(units=output_node, 
                activation='relu',
                # kernel_initializer=initializers.he_normal(),
                # kernel_initializer=initializers.glorot_normal(),
                kernel_regularizer=regularizers.l2(weight), 
                name='output_layer'))
        else:
            if i == layer-1:    
                return_sequences = False
            else:   
                return_sequences = True
            units = int(layer_size/(2**(i-1)))
            model.add(layers.LSTM(units=units, 
                input_shape=(x.shape[1], x.shape[2]), 
                activation='relu',
                # kernel_initializer=initializers.he_normal(),
                # kernel_initializer=initializers.glorot_normal(),
                kernel_regularizer=regularizers.l2(weight), 
                return_sequences=return_sequences, 
                name='hidden_layer{}'.format(i), 
            ))
            model.add(layers.Dropout(rate)),
            # model.add(layers.BatchNormalization())
    return model


def stateless_lstm_more(x, output_node, hidden_layers=4, layer_size=128):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(units=layer_size, 
        input_shape=(x.shape[1], x.shape[2]), \
        return_sequences=True))
    for i in range(hidden_layers-2):  # hidden_layers>=2
        model.add(layers.LSTM(units=layer_size, 
            return_sequences=True))
    model.add(layers.LSTM(units=layer_size, 
        return_sequences=False))
    model.add(layers.Dense(units=output_node, 
        kernel_initializer=initializers.he_normal(),
        activation='relu'))
    return model

def cnn_lstm(x, output_node, layer=2, layer_size=128, rate=0.20, weight=1e-6):
    # model = tf.keras.Sequential()
    # model.add(layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', \
    #     input_shape=(x.shape[1], x.shape[2]), activation='relu',data_format="channels_last"))
    # # model.add(layers.BatchNormalization())
    # model.add(layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='valid', \
    #     activation='relu', data_format="channels_last"))
    # # model.add(layers.BatchNormalization())
    # model.add(layers.Flatten())
    # model.add(layers.Dropout(rate))
    
    # model.add(layers.Dense(units=output_node, 
    #     activation='relu',
    #     kernel_initializer=initializers.he_normal(),
    #     # kernel_initializer=initializers.glorot_normal(),
    #     kernel_regularizer=regularizers.l2(weight), 
    #     name='output_layer', 
    # ))
    # return model

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(x.shape[1], x.shape[2])))
    # model.add(layers.Input(shape=(time_step, 128))
    model.add(layers.Dropout(rate))
    model.add(layers.Conv1D(
        filters=64, kernel_size=3, strides=1, padding='valid', \
        kernel_regularizer=regularizers.l2(weight), 
        activation='relu', kernel_initializer=initializers.he_normal(), data_format="channels_first"))
    model.add(layers.MaxPooling1D(pool_size=3))
    model.add(layers.LSTM(128, 
        kernel_regularizer=regularizers.l2(weight), 
        kernel_initializer=initializers.he_normal(),
        activation='relu', 
        return_sequences=True))
    model.add(layers.Dropout(rate))
    model.add(layers.LSTM(64, 
        kernel_regularizer=regularizers.l2(weight), 
        kernel_initializer=initializers.he_normal(),
        activation='relu'))
    model.add(layers.Dropout(rate))
    model.add(layers.Dense(output_node, 
        kernel_regularizer=regularizers.l2(weight), 
        kernel_initializer=initializers.he_normal(),
        activation='relu'))
    return model

def stateful_lstm(x, output_node, batch_size=32, 
        layer=3, layer_size=128, rate=0.20, weight=1e-6):
    model = tf.keras.Sequential()
    for i in np.arange(1, layer+1, 1):
        if i == 1:
            model.add(layers.LSTM(units=layer_size, 
                batch_input_shape=(batch_size, x.shape[1], x.shape[2]),
                activation='relu',
                # kernel_initializer=initializers.glorot_normal(),
                kernel_initializer=initializers.he_normal(),
                kernel_regularizer=regularizers.l2(weight), 
                return_sequences=True, stateful=True, 
                name='input_layer{}'.format(i)))
            model.add(layers.Dropout(rate)),
            # model.add(layers.BatchNormalization())
        elif i > 1 and i < layer-1:
            model.add(layers.LSTM(units=layer_size, 
                activation='relu',
                kernel_initializer=initializers.he_normal(),
                kernel_regularizer=regularizers.l2(weight), 
                stateful=True, return_sequences=True, 
                name='input_layer{}'.format(i)))
            model.add(layers.Dropout(rate)),
            # model.add(layers.BatchNormalization())
        elif i == layer-1:
            model.add(layers.LSTM(units=layer_size, 
                activation='relu',
                kernel_initializer=initializers.he_normal(),
                kernel_regularizer=regularizers.l2(weight), 
                stateful=True, return_sequences=False, 
                name='input_layer{}'.format(i)))
            model.add(layers.Dropout(rate)),
            # model.add(layers.BatchNormalization())
        else:
            model.add(layers.Dense(units=output_node, 
                kernel_regularizer=regularizers.l2(weight), 
                kernel_initializer=initializers.he_normal(),
                activation='relu', 
                name='input_layer{}'.format(i)))
    return model