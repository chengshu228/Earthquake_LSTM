import config

np = config.np
tf = config.tf
layers = config.layers
initializers = config.initializers
regularizers = config.regularizers

def stateless_lstm(x, output_node, layer=2, layer_size=128, rate=0.20, weight=1e-6):
    model = tf.keras.Sequential()
    for i in np.arange(1, layer, 1):
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
    model.add(layers.Dense(units=output_node, 
        activation='relu',
        # kernel_initializer=initializers.he_normal(),
        # kernel_initializer=initializers.glorot_normal(),
        kernel_regularizer=regularizers.l2(weight), 
        name='output_layer', 
    ))
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


def stateful_lstm(x, batch_size, layer_size, output_node):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(
        units=layer_size, 
        activation='relu', 
        kernel_initializer=initializers.he_normal(),
        batch_input_shape=(batch_size, x.shape[1], x.shape[2]), 
        stateful=True, return_sequences=True))
    model.add(layers.LSTM(
        units=int(layer_size/2), 
        activation='relu', 
        kernel_initializer=initializers.he_normal(),
        stateful=True, return_sequences=True))
    model.add(layers.LSTM(
        units=int(layer_size/4), 
        activation='relu', 
        kernel_initializer=initializers.he_normal(),
        stateful=True, return_sequences=False))
    model.add(layers.Dense(units=output_node, activation='relu', ))
    return model

