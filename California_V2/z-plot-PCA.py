
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu
import tensorflow as tf
assert tf.__version__.startswith("2.") 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN

import config
from ANN import stateless_lstm, stateful_lstm, stateless_lstm_more
from utils import series_to_supervised, \
    checkpoints, save_data, dataset



start_time = time.time()
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

span_lat = config.span_lat
span_lon = config.span_lon
time_window = config.time_window
blocks = config.blocks 
features = config.features
index = config.index

n_splits = config.n_splits
epochs = config.epochs
learning_rate = config.learning_rate
filename = config.filename
catolog_name = config.catolog_name

file_location = r'C:\Users\cshu\Desktop\shi\data'
value = dataset(blocks=blocks)
value[:, 0::features] = value[:, 0::features] ** index
input_data = value[:, :features*blocks]
output_data = M = value[:, features*blocks::features]
if config.energy:  # 能量平方根
    output_data = np.around(np.sqrt(np.power(10, 1.8*output_data+12)), 0)
print('input_data', input_data.shape, 'output_data', output_data.shape)

x_scaler = MinMaxScaler(feature_range=(0, 1))
input_data = x_scaler.fit_transform(input_data).reshape(-1, features*blocks)  
y_scaler = MinMaxScaler(feature_range=(0, 1)) 
output_data = y_scaler.fit_transform(output_data).reshape(-1, blocks)  

n_components = config.n_components # features*blocks 
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

series_length = 1
# x = input_data.reshape((-1, series_length, features*blocks))
x = input_data.reshape((-1, series_length, input_data.shape[1]))
y = output_data.reshape(-1, blocks)
# x = np.concatenate((x_train, x_test), axis=0)

fold = TimeSeriesSplit(n_splits=n_splits, max_train_size=None)

loss_per_fold = []
mae_per_fold = []
rmse_per_fold = []
for fold_no, (train_index, test_index) in enumerate(fold.split(x, y)):
    model = stateless_lstm(x, output_node=blocks, 
        layer=config.layer, layer_size=config.layer_size, 
        rate=config.rate, weight=config.weight)
    optimizer = optimizers.Adam(learning_rate=learning_rate) 
    model.compile(optimizer=optimizer, loss='mean_squared_error',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    print(f'Training for fold {fold_no+1} ...')
    history = model.fit(x[train_index], y[train_index], verbose=2, epochs=epochs, 
        batch_size=len(x[train_index]), 
        validation_data=(x[test_index], y[test_index]), 
        callbacks=checkpoints(), 
        shuffle=False)    
    test_loss, test_mae, test_rmse = model.evaluate(x[test_index], y[test_index],
        batch_size=len(x[train_index]), verbose=2)
    print('Evaluate on testing data: ' + \
        'test_mse={0:.2f}%, test_mae={1:.2f}%, test_rmse={2:.2f}%'.\
        format(test_loss*100, test_mae*100, test_rmse*100))    
    loss_per_fold.append(test_loss)
    mae_per_fold.append(test_mae)
    rmse_per_fold.append(test_rmse)

    y_train = y_scaler.inverse_transform(y[train_index])
    y_train_pre = model.predict(x[train_index], batch_size=len(x[train_index]))
    y_train_pre = y_scaler.inverse_transform(y_train_pre)
    y_test = y_scaler.inverse_transform(y[test_index])
    y_test_pre = model.predict(x[test_index], batch_size=len(x[train_index]))
    y_test_pre = y_scaler.inverse_transform(y_test_pre)

    len_train = len(y_train.reshape(-1, ))
    len_test = len(y_test.reshape(-1, ))

    # plt.figure(figsize=(10, 6))
    # plt.grid(True, linestyle='--', linewidth=0.5)
    # plt.scatter(np.arange(len_train), y_train.reshape(-1, ), 
    #     marker='+', c='grey', label='observed')
    # plt.scatter(np.arange(len_train, len_train+len_test, 1), y_test.reshape(-1, ), 
    #     marker='+', c='grey')
    # plt.scatter(np.arange(len_train), y_train_pre.reshape(-1, ), 
    #     marker='+', c='dodgerblue', label='predicted training set')    
    # plt.scatter(np.arange(len_train, len_train+len_test, 1), y_test_pre.reshape(-1, ), 
    #     marker='+', c='indianred', label='predicted validation set')
    # plt.title(f'Training for fold {fold_no+1}', fontsize=20)
    # plt.xlabel('sample index (years + blocks)', fontsize=18)  
    # plt.ylabel('M', fontsize=18)  
    # plt.xticks(size=16)
    # plt.yticks(size=16)
    # plt.ylim(2.4, 8.6)
    # plt.legend(loc='lower left', fontsize=16) 
    # ax = plt.gca()
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['right'].set_linewidth(1.5) #
    # if fold_no+1 == n_splits:
    #     plt.savefig(r".\figure\fold{}-{}.png".format(fold_no+1, filename))
        # plt.show()

model.save(r'.\model_lstm\LSTMT-{}.h5'.format(filename))

fig = plt.figure(figsize=(10, 6))  
plt.text(0.01, 3.3, 'Score per fold (validation set):', va='bottom', fontsize=16)
for i in range(0, len(loss_per_fold)):
    plt.text(0.01, 3.1-i/5, 
        f'  Fold {i+1:.0f} - MSE: {100*loss_per_fold[i]:.2f}%' +
        f'- MAE: {100*mae_per_fold[i]:.2f}%' +
        f'- RMSE: {100*rmse_per_fold[i]:.2f}%', va='bottom', fontsize=16)    
print('Average scores for all folds:')
plt.text(0.01, 0.7, 'Average scores for all folds:', va='bottom', fontsize=16)
plt.text(0.01, 0.5, f'  MSE: {np.mean(loss_per_fold):.4f}', va='bottom', fontsize=16)
plt.text(0.01, 0.3, f'  MAE: {np.mean(mae_per_fold):.4f} (+/- {np.std(mae_per_fold):.4f})', 
    va='bottom', fontsize=16)
plt.text(0.01, 0.1, f'  RMSE: {np.mean(rmse_per_fold):.4f} (+/- {np.std(rmse_per_fold):.4f})', 
    va='bottom', fontsize=16)
plt.ylim(-0.1, 3.6)
plt.xlim(-0.2, 2.2)
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.savefig(r".\figure\6\error-{}.png".format(filename))
plt.show()

# save_data(file_location=file_location, 
#     name='loss-{}'.format(filename), value=history.history['loss'])    
# save_data(file_location=file_location, 
#     name='mae-{}'.format(filename), value=history.history['mae'])    
# save_data(file_location=file_location, 
#     name='rmse-{}'.format(filename), value=history.history['rmse'])  
# save_data(file_location=file_location, 
#     name='val_loss-{}'.format(filename), value=history.history['val_loss'])    
# save_data(file_location=file_location, 
#     name='val_mae-{}'.format(filename), value=history.history['val_mae'])    
# save_data(file_location=file_location, 
#     name='val_rmse-{}'.format(filename), value=history.history['val_rmse'])  

# loss = read_data(file_location=file_location, name="loss")
# mae = read_data(file_location=file_location, name="mae")
# rmse = read_data(file_location=file_location, name="rmse")
# val_loss = read_data(file_location=file_location, name="val_loss")  
# val_mae = read_data(file_location=file_location, name="val_mae")
# val_rmse = read_data(file_location=file_location, name="val_rmse")
# x = np.arange(len(loss))
# fig = plt.figure(figsize=(14, 4))
# plt.subplot(1,3,1) 
# plt.plot(x, loss, color="dodgerblue", label='loss')
# plt.plot(x, val_loss, color="indianred", label='val_loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# # plt.ylim(-0.001, 0.011)
# plt.legend()
# plt.grid(True, linestyle='--')
# plt.subplot(1,3,2) 
# plt.plot(x, mae, color="dodgerblue", label='mae')
# plt.plot(x, val_mae, color="indianred", label='val_mae')
# plt.xlabel('epoch')
# plt.ylabel('MAE')
# # plt.ylim(-0.01, 0.11)
# plt.legend()
# plt.grid(True, linestyle='--')
# plt.subplot(1,3,3) 
# plt.plot(x, rmse, color="dodgerblue", label='rmse')
# plt.plot(x, val_rmse, color="indianred", label='val_rmse')
# plt.xlabel('epoch')
# plt.ylabel('RMSE')
# # plt.ylim(-0.01, 0.11)
# plt.legend()
# plt.grid(True, linestyle='--')
# plt.subplots_adjust(wspace=0.35)
# plt.savefig(r".\figure\loss-{}.png".format(filename))
# # plt.savefig(r".\figure\cnn-p-loss.eps")
# plt.show()

print((time.time()-start_time)/60, ' minutes')

