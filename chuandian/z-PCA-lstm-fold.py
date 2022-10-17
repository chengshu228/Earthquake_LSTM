

import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN

from ANN import stateless_lstm, stateless_lstm_more
from utils import checkpoints, save_data, dataset



start_time = time.time()

features = 17
blocks = 36
index = 2
part = 2

file_location = r'C:\Users\cshu\Desktop\shi\data'

value = dataset(blocks=blocks)
value[:, 0::17] = value[:, 0::17] ** index
input_data = value[:, :features*blocks]
output_data = value[:, features*blocks::features]
logE = np.around(1.8*output_data+12, 1)
output_data = 10**(0.5*logE)
print('input_data.shape=', input_data.shape, 'output_data.shape=', output_data.shape)

x_scaler = MinMaxScaler(feature_range=(0, 1))
input_data = x_scaler.fit_transform(input_data).reshape(-1, features*blocks)  
y_scaler = MinMaxScaler(feature_range=(0, 1)) 
output_data = y_scaler.fit_transform(output_data).reshape(-1, blocks)  

n_components = 20 # features*blocks
pca = PCA(n_components=n_components, svd_solver='full')
# kernels = ['linear','poly','rbf','sigmoid']
# pca = KernelPCA(n_components=n_components, kernel=kernels[0])
# pca = KernelPCA(n_components=n_components, kernel=kernels[1])
# pca = KernelPCA(n_components=n_components, kernel=kernels[2])
# pca = KernelPCA(n_components=n_components, kernel=kernels[3])
# pca = IncrementalPCA(n_components=n_components)
# pca = TSNE(n_components=1)  # one-component TSNE
# pca = TSNE(n_components=n_components)  # two-component TSNE
# pca = MDS(n_components=1)  # one-component MDS
# pca = MDS(n_components=2)  # two-component MDS
# pca = DBSCAN()
# pca = pca.fit_predict(x_oringin) # c = y_pred
input_data = pca.fit_transform(input_data)  
print(input_data.shape)

x_train, x_validation = input_data[:12*40, :], input_data[12*40:, :]
y_train, y_validation = output_data[:12*40, :], output_data[12*40:, :]

time_series_length = 1
x_train = x_train.reshape((x_train.shape[0], time_series_length, x_train.shape[1]))
x_validation = x_validation.reshape((x_validation.shape[0], time_series_length, x_validation.shape[1]))
y_train = y_train.reshape(y_train.shape[0], blocks)
y_validation = y_validation.reshape(y_validation.shape[0], blocks)

epochs = 10**2
batch_size = 16
reset_number = epochs
output_node = blocks
learning_rate = 1e-3
# learning_rate = optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=learning_rate, decay_steps=epochs,
# 	decay_rate=0.99, staircase=False)

# model = stateless_lstm(x_train, filter1=64, output_node=output_node)
model = stateless_lstm_more(x_train, output_node)
print('model.summary(): \n{} '.format(model.summary()))
print('layer nums:', len(model.layers))

# RMSprop Adam sgd adagrad adadelta adamax nadam
optimizer = optimizers.Adam(learning_rate=learning_rate) 

model.compile(optimizer=optimizer, loss='mean_squared_error',
    metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

# for i in range(20):
# 	print('steps: ', i)
# 	global history
# 	# history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_validation, y_validation), \
# 	# 						batch_size=batch_size, verbose=2, shuffle=False)
# 	history = model.fit(x_train, y_train, epochs = epochs, shuffle=False, batch_size=batch_size,
#                  			verbose=2, validation_data = (x_validation, y_validation), callbacks = callbacks)
# 	model.reset_states()
history = model.fit(x_train, y_train, epochs=epochs, shuffle=False, batch_size=batch_size, 
    verbose=2, validation_data=(x_validation, y_validation), callbacks=checkpoints())

test_loss, test_mae, test_rmse = model.evaluate(x_validation, y_validation, verbose=2)
print('Evaluate on test data: ')
print('  test_mse={0:.4f}%, test_mae={1:.4f}%, test_rmse={2:.4f}%'.\
	format(test_loss*100, test_mae*100, test_rmse*100))

save_data(file_location=file_location, 
    name='pca-loss-M{}-P{}'.format(index, part), value=history.history['loss'])    
save_data(file_location=file_location, 
    name='pca-mae-M{}-P{}'.format(index, part), value=history.history['mae'])    
save_data(file_location=file_location, 
    name='pca-rmse-M{}-P{}'.format(index, part), value=history.history['rmse'])  
save_data(file_location=file_location, 
    name='pca-val_loss-M{}-P{}'.format(index, part), value=history.history['val_loss'])    
save_data(file_location=file_location, 
    name='pca-val_mae-M{}-P{}'.format(index, part), value=history.history['val_mae'])    
save_data(file_location=file_location, 
    name='pca-val_rmse-M{}-P{}'.format(index, part), value=history.history['val_rmse'])  

plt.plot(history.history['mae'], label='mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(loc='upper left')
# plt.savefig(r'.\figure\train-cnn-p.pdf')
plt.show()






