import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import optimizers
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

import config
import ANN
import utils

matplotlib.rcParams.update(config.config_font)
plt.rcParams['font.sans-serif']=['simsun'] 
plt.rcParams['axes.unicode_minus']=False 

file_name,model_name = config.file_name,config.model_name
data_location,file_location = config.data_location,config.file_location
figure_location = config.figure_location
min_year,max_year = config.min_year,config.max_year 
min_lat,max_lat = config.min_lat,config.max_lat 
min_lon,max_lon = config.min_lon,config.max_lon
min_mag,min_number = config.min_mag,config.min_number
span_lat,span_lon = config.span_lat,config.span_lon
time_window,next_month = config.time_window,config.next_month
blocks,features = config.blocks,config.features
index,energy = config.index,config.energy
m,n = config.m,config.n
learning_rate,batch_size = config.learning_rate,config.batch_size
layer,layer_size = config.layer,config.layer_size
rate,weight_decay = config.rate,config.weight_decay
split_ratio = config.split_ratio
epochs = config.epochs
simsun,times = config.times,config.simsun
loc_block = config.loc_block
# n_out = config.blocks
n_splits = config.n_splits

start_time = time.time()
seed = config.seed
np.random.seed(seed)
tf.random.set_seed(seed)

factor = np.loadtxt(data_location+\
    f'\\factor-{file_name}.txt', delimiter=' ')
print('\n initial factor:', factor.shape, blocks*(features*n+m))

output_data = factor[:,0].reshape(-1,blocks) 
input_data = np.concatenate((
    factor[:, 1].reshape(-1, 1),  # frequency
    factor[:, 2].reshape(-1, 1),  # max_magnitude
    factor[:, 3].reshape(-1, 1),  # mean_magnitude
    factor[:, 4].reshape(-1, 1), # b_lstsq
    factor[:, 5].reshape(-1, 1),  # b_mle
    factor[:, 6].reshape(-1, 1), # a_lstsq
    factor[:, 7].reshape(-1, 1), # max_mag_absence
    factor[:, 8].reshape(-1, 1), # rmse_lstsq
    factor[:, 9].reshape(-1, 1), # total_energy_square
    factor[:, 10].reshape(-1, 1), # mean_lon
    factor[:, 11].reshape(-1, 1), # rmse_lon
    factor[:, 12].reshape(-1, 1), # mean_lat
    factor[:, 13].reshape(-1, 1), # rmse_lat
    factor[:, 14].reshape(-1, 1), # k
    factor[:, 15].reshape(-1, 1), # epicenter_longitude
    factor[:, 16].reshape(-1, 1), # epicenter_latitude
    ), axis=1)
input_data = input_data.reshape(-1, blocks*(features*n+m))
print('input=', input_data.shape, 'output=', output_data.shape)

if loc_block <= 10: n_out = 1
else: n_out = loc_block//10
print('\tn_out={}'.format(n_out))

if loc_block<=10:
    output_data = output_data[:,loc_block-1].reshape(-1,n_out)
elif loc_block==55:
    output_data = output_data[:,1:].reshape(-1,n_out)
elif loc_block==66: output_data = output_data

output_data = np.power(output_data, index)
if energy:
    output_data = np.around(np.sqrt(np.power(10, 1.5*output_data+11.8)), 0)

x_scaler = MinMaxScaler().fit(input_data) 
input_data = x_scaler.transform(input_data)
y_scaler = MinMaxScaler().fit(output_data)
output_data = y_scaler.transform(output_data)

num = int(len(input_data)*split_ratio)
x,y = input_data[:num,:],output_data[:num,:]
x_test,y_test = input_data[num:,:],output_data[num:,:]
x,x_test = np.expand_dims(x,axis=1),np.expand_dims(x_test,axis=1)
print(x.shape, x_test.shape, y.shape, y_test.shape)

fold = TimeSeriesSplit(n_splits=n_splits, max_train_size=None)
# listy = []
# for j in np.arange(0, n_splits+0.1, 1):
#     listy.append(f'Training for Fold '+str(int(j)))
# fig = plt.figure(figsize=(8, 8))
# for i, (train_index, val_index) in enumerate(fold.split(x, y)):
#     l1 = plt.scatter(train_index, [i+1]*len(train_index), 
#         c='dodgerblue', marker='_', lw=14)
#     l2 = plt.scatter(val_index, [i+1]*len(val_index), 
#         c='darkviolet', marker='_', lw=14)
#     plt.legend([l1, l2], ['Training set', 'Validation set'], \
#         prop=times, loc='upper right', fontsize=16)
#     plt.xlabel('Sample Index (Month)', fontproperties='Arial', fontsize=18)  
#     plt.ylabel('CV Iteration', fontproperties='Arial', fontsize=18)  
#     plt.title('Time Series Split', fontproperties='Arial', fontsize=20)  # Blocking
#     plt.text(1.2, i+1.15, '{} | {}'.format(len(train_index), len(val_index)), 
#         fontproperties='Arial', fontsize=16)
#     plt.xticks(size=14)
#     plt.yticks(np.arange(0, n_splits+0.1, 1), listy, size=14)
#     # plt.axvline(x=444, ls=':', c='green')
#     ax = plt.gca()
#     ax.spines['bottom'].set_linewidth(1.5)
#     ax.spines['left'].set_linewidth(1.5)
#     ax.spines['top'].set_linewidth(1.5)
#     ax.spines['right'].set_linewidth(1.5)
#     # ax.set(ylim=[n_splits+0.9, 0.1])
#     ax.set(ylim=[n_splits+0.5, 0.5])
#     ax.yaxis.set_major_locator(plt.MultipleLocator(1))
#     plt.tight_layout()
# plt.savefig(file_location+r'\figure\{}-CV.png'.format(file_name))
# plt.show()

loss_per_fold = []
mae_per_fold = []
rmse_per_fold = []

if model_name=='lstm':
    model = ANN.LSTM(x=x, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='cnn':
    model = ANN.CNN(x=x, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='cnn_lstm':
    model = ANN.cnn_lstm(x=x, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='lstm_cnn':
    model = ANN.lstm_cnn(x=x, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='tcn':
    model = ANN.tcn(x=x, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='tcn_lstm':
    model = ANN.tcn_lstm(x=x, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='tcn_cnn':
    model = ANN.tcn_cnn(x=x, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='tcn_lstm_cnn':
    model = ANN.tcn_lstm_cnn(x=x, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='tcn_cnn_lstm':
    model = ANN.tcn_cnn_lstm(x=x, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
print('\t model.summary(): \n{} '.format(model.summary()))
print('\t layer nums:', len(model.layers))
  
optimizer = optimizers.Adam(learning_rate=learning_rate) 
model.compile(optimizer=optimizer, loss='mean_squared_error',
    metrics=[#'mae', 
    tf.keras.metrics.RootMeanSquaredError(name='rmse')])

for fold_no, (train_index, val_index) in enumerate(fold.split(x, y)):
    print(f'\n\n\nTraining for Fold {fold_no+1} ...')
    # history = model.fit(
    #     x[train_index,:,:], y[train_index,:], 
    #     verbose=2, epochs=epochs, 
    #     # batch_size=len(x[train_index]), 
    #     # validation_data=(x[val_index,:,:], y[val_index,:]), 
    #     # callbacks=utils.checkpoints(model_name=model_name,name=file_name), 
    #     shuffle=False)    
    history = model.fit(x[train_index], y[train_index], 
        validation_data=(x[val_index], y[val_index]),
        epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False, 
        callbacks=utils.checkpoints(model_name=model_name,name=file_name))

    val_loss, val_rmse = model.evaluate(x[val_index], y[val_index],
        batch_size=len(x[train_index]), verbose=2)
    print('Evaluate on validation data: ' + \
        'val_mse={0:.2f}, val_rmse={1:.2f}'.\
        format(val_loss, val_rmse))    
    loss_per_fold.append(val_loss)
    rmse_per_fold.append(val_rmse)

    y_train = y_scaler.inverse_transform(y[train_index])
    y_train_pre = model.predict(x[train_index])
    y_train_pre = y_train_pre.reshape(-1, n_out)
    y_train_pre = y_scaler.inverse_transform(y_train_pre)
    y_val = y_scaler.inverse_transform(y[val_index])
    y_val_pre = model.predict(x[val_index])
    y_val_pre = y_val_pre.reshape(-1, n_out)
    y_val_pre = y_scaler.inverse_transform(y_val_pre)  
    
    len_train = len(y_train.reshape(-1, ))
    len_val = len(y_val.reshape(-1, ))   
    len_test = len(y_test.reshape(-1, ))    

    y_train = np.power(y_train, 1/index).reshape(-1, )
    y_train_pre = np.power(y_train_pre, 1/index).reshape(-1, )
    y_val = np.power(y_val, 1/index).reshape(-1, )
    y_val_pre = np.power(y_val_pre, 1/index).reshape(-1, )

    if config.energy:
        y_train = ((2*np.log10(y_train)-11.8) / 1.5).reshape(-1, )
        y_train_pre = ((2*np.log10(y_train_pre)-11.8) / 1.5).reshape(-1, )
        y_val = ((2*np.log10(y_val)-11.8) / 1.5).reshape(-1, )
        y_val_pre = ((2*np.log10(y_val_pre)-12) / 1.5).reshape(-1, )

    # if fold_no+1 != config.n_splits:
    #     plt.figure(figsize=(8, 5))
    #     plt.grid(True, linestyle='--', linewidth=1.0) 
    #     plt.scatter(np.arange(len_train), y_train, 
    #         marker='o', c='none', edgecolors='grey', label='Observed', s=10)
    #     plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val, 
    #         marker='o', c='none', edgecolors='grey', s=10)
    #     plt.scatter(np.arange(len_train), y_train_pre, 
    #         marker='+', c='dodgerblue', label='Predicted (Training Set)', s=30)    
    #     plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val_pre, 
    #         marker='x', c='darkviolet', label='Predicted (Validation Set)', s=25)
    #     plt.title(f'Training for Fold {fold_no+1}', fontproperties='Arial', fontsize=20)
    #     plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)         
    #     plt.ylabel('Predicted max. Magnitude', fontproperties='Arial', fontsize=18)  
    #     plt.xticks(size=14)
    #     plt.yticks(size=14)
    #     plt.ylim(2.5, 8.5)
    #     plt.legend(loc='upper left', prop=times, fontsize=15) 
    #     ax = plt.gca()
    #     ax.spines['bottom'].set_linewidth(1.5)
    #     ax.spines['left'].set_linewidth(1.5)
    #     ax.spines['top'].set_linewidth(1.5)
    #     ax.spines['right'].set_linewidth(1.5)
    #     plt.tight_layout()
    #     plt.savefig(file_location+r'.\figure\{}-fold{}.png'.format(file_name, fold_no+1))
        # plt.show()
    if fold_no+1 == config.n_splits:
        y_test = y_scaler.inverse_transform(y_test)
        y_test_pre = model.predict(x_test)
        y_test_pre = y_test_pre.reshape(-1, n_out)
        y_test_pre = y_scaler.inverse_transform(y_test_pre)
        y_test = np.power(y_test, 1/index).reshape(-1, )
        y_test_pre = np.power(y_test_pre, 1/index).reshape(-1, )
        if config.energy:
            y_test = (2*np.log10(y_test)-11.8) / 1.5
            y_test_pre = (2*np.log10(y_test_pre)-11.8) / 1.5    

        plt.figure(figsize=(8, 5))
        plt.grid(True, linestyle='--', linewidth=1.0) 
        plt.scatter(np.arange(len_train+len_val, len_train+len_val+len_test), y_test, 
            marker='o', c='none', edgecolors='grey', label='Observed', s=10)
        plt.scatter(np.arange(len_train+len_val, len_train+len_val+len_test), y_test_pre, 
            marker='*', c='indianred', label='Predicted (Testing Set)', s=20)
        plt.title('Testing Set', fontproperties='Arial', fontsize=20, color='red')
        plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)         
        plt.ylabel('Predicted max. Magnitude', fontproperties='Arial', fontsize=18)  
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.legend(loc='best', prop=times, fontsize=15) 
        plt.tight_layout()
        plt.savefig(figure_location+f'\{file_name}-fold{fold_no+1}.png')
        # plt.show()

        plt.figure(figsize=(8, 5))
        plt.grid(True, linestyle='--', linewidth=1.0) 
        plt.scatter(np.arange(len_train), y_train, 
            marker='o', c='none', edgecolors='grey', label='Observed', s=10)
        plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val, 
            marker='o', c='none', edgecolors='grey', s=10)
        plt.scatter(np.arange(len_train), y_train_pre, 
            marker='+', c='dodgerblue', label='Predicted (Training Set)', s=30)    
        plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val_pre, 
            marker='x', c='darkviolet', label='Predicted (Validation Set)', s=25)
        plt.title(f'Training for Fold {fold_no+1}', fontproperties='Arial', fontsize=20, color='red')
        plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)         
        plt.ylabel('Predicted max. Magnitude', fontproperties='Arial', fontsize=18)  
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.legend(loc='best', prop=times, fontsize=15) 
        plt.tight_layout()
        plt.savefig(figure_location+f'\{file_name}-fold{fold_no+1}.png')
        # plt.show()

        plt.figure(figsize=(8, 5))
        plt.grid(True, linestyle='--', linewidth=1.0) 
        plt.scatter(np.arange(len_train), y_train, 
            marker='o', c='none', edgecolors='grey', label='Observed', s=10)
        plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val, 
            marker='o', c='none', edgecolors='grey', s=10)
        plt.scatter(np.arange(len_train+len_val, len_train+len_val+len_test, 1), y_test, 
            marker='o', c='none', edgecolors='grey', s=10)
        plt.scatter(np.arange(len_train), y_train_pre, 
            marker='+', c='dodgerblue', label='Predicted (Training Set)', s=30)    
        plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val_pre, 
            marker='x', c='darkviolet', label='Predicted (Validation Set)', s=25)
        plt.scatter(np.arange(len_train+len_val, len_train+len_val+len_test, 1), y_test_pre, 
            marker='*', c='indianred', label='Predicted (Testing Set)', s=20)
        plt.title(f'All Set', fontproperties='Arial', fontsize=20, color='red')
        plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)         
        plt.ylabel('Predicted max. Magnitude', fontproperties='Arial', fontsize=18)  
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.legend(loc='upper left', prop=times, fontsize=15) 
        plt.tight_layout()
        plt.savefig(figure_location+f'\{file_name}-fold{fold_no+1}.png')
        # plt.show()

    if not os.path.exists(file_location+r'\loss'):
        os.mkdir(file_location+r'\loss')
    utils.save_data(file_location=file_location+r'\loss', 
        name='loss-{}-{}'.format(file_name, fold_no+1), value=history.history['loss'])    
    utils.save_data(file_location=file_location+r'\loss', 
        name='rmse-{}-{}'.format(file_name, fold_no+1), value=history.history['rmse'])  
    utils.save_data(file_location=file_location+r'\loss', 
        name='val_loss-{}-{}'.format(file_name, fold_no+1), value=history.history['val_loss'])    
    utils.save_data(file_location=file_location+r'\loss', 
        name='val_rmse-{}-{}'.format(file_name, fold_no+1), value=history.history['val_rmse'])  

model.save(file_location+f'\\{model_name}\{file_name}.h5')

fig = plt.figure(figsize=(3, 1.5))  
plt.text(0.01, 0.7, 'Average scores for all folds:', va='bottom', fontsize=14)
plt.text(0.01, 0.5, 
    f' MSE:{np.mean(loss_per_fold):.4f}', 
    va='bottom', fontsize=14)
plt.text(0.01, 0.1, 
    f' RMSE:{np.mean(rmse_per_fold):.4f}(+/-{np.std(rmse_per_fold):.4f})', 
    va='bottom', fontsize=14)
plt.ylim(0, 0.85)
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.tight_layout()
plt.savefig(file_location+r'\figure\error-{}.png'.format(file_name))
plt.show()

print((time.time()-start_time)/60, ' minutes')

# fig = plt.figure(figsize=(8, 6))  
# plt.text(0.01, 3.3, 'Score per Fold (Validation Set):', va='bottom', fontsize=14)
# for i in range(0, len(loss_per_fold)):
#     plt.text(0.01, 3.1-i/10, 
#         f'  Fold {i+1:.0f} - MSE: {100*loss_per_fold[i]:.2f}%' +
#         f'- MAE: {100*mae_per_fold[i]:.2f}%' +
#         f'- RMSE: {100*rmse_per_fold[i]:.2f}%', va='bottom', fontsize=14)    
# ax = plt.gca()
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# plt.show()

