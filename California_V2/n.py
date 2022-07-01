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
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import time
import os
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from matplotlib.font_manager import FontProperties
import datetime
import julian
from dateutil.relativedelta import relativedelta
import matplotlib

import ANN
import utils 
import config
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
n_out = config.blocks
simsun,times = config.times,config.simsun
loc_block = config.loc_block

matplotlib.rcParams.update(config.config_font)
plt.rcParams['font.sans-serif']=['simsun'] 
plt.rcParams['axes.unicode_minus']=False 
initial_julian = julian.to_jd(datetime.datetime(1930,1,1,0,0,0), fmt='jd')

date = np.genfromtxt(data_location+\
    f'\\date-{file_name}.txt', delimiter=' ')
# date = date[time_window:-next_month,:]
print(len(date))
year = np.array(date[:, 0], dtype=np.int)
month = np.array(date[:, 1], dtype=np.int)
day = np.array(date[:, 2], dtype=np.int)
date = pd.DataFrame({'year':year, 'month':month, 'day':day})
date = pd.to_datetime(date, format='%Y%m%d', errors='ignore')

jul_date0 = []
for i in np.arange(len(date)): 
    t = datetime.datetime(year[i],month[i],1,0,0,0)
    t_jd = julian.to_jd(t,fmt='jd')-initial_julian
    jul_date0.append(t_jd)

jul_date = []
for i in np.arange(time_window,len(date)-next_month+1,1): 
    t = datetime.datetime(year[i],month[i],1,0,0,0)
    t_jd = julian.to_jd(t,fmt='jd')-initial_julian
    jul_date.append(t_jd)

jul_date_pre = np.zeros(shape=(next_month,next_month))
for i in np.arange(next_month): 
    if len(year)==next_month+len(jul_date):
        t = datetime.datetime(year[-1], month[-1],1,0,0,0)+\
            relativedelta(months=i+time_window)
    else:
        for j in np.arange(next_month):
            t = datetime.datetime(year[len(jul_date)],
                month[len(jul_date)],1,0,0,0)+relativedelta(months=time_window+i+j)
            t_jd = julian.to_jd(t,fmt='jd')-initial_julian
            jul_date_pre[i,j] = t_jd

x_label1, x_label_tick1 = [], []
for tick in np.arange(1930, 2030+1, config.step): 
    t = datetime.datetime(tick,1,1,0,0,0)
    t_jd = julian.to_jd(t, fmt='jd')-initial_julian
    x_label1.append(t_jd)
    x_label_tick1.append(tick)

start_time = time.time()
seed = config.seed
np.random.seed(seed)
tf.random.set_seed(seed)

factor = np.genfromtxt(data_location+\
    f'\\factor-{file_name}1.txt', delimiter=' ')
print('\n initial factor:', factor.shape, blocks*(features*n+m))

input_data0 = np.concatenate((
    # factor[:, 1].reshape(-1, 1),  # frequency
    # factor[:, 2].reshape(-1, 1),  # max_magnitude
    # factor[:, 3].reshape(-1, 1),  # mean_magnitude
    # factor[:, 4].reshape(-1, 1), # b_lstsq
    # factor[:, 5].reshape(-1, 1),  # b_mle
    # factor[:, 6].reshape(-1, 1), # a_lstsq
    # factor[:, 7].reshape(-1, 1), # max_mag_absence
    # factor[:, 8].reshape(-1, 1), # rmse_lstsq
    # factor[:, 9].reshape(-1, 1), # total_energy_square
    # factor[:, 10].reshape(-1, 1), # mean_lon
    # factor[:, 11].reshape(-1, 1), # rmse_lon
    # factor[:, 12].reshape(-1, 1), # mean_lat
    # factor[:, 13].reshape(-1, 1), # rmse_lat
    # factor[:, 14].reshape(-1, 1), # k
    # factor[:, 15].reshape(-1, 1), # epicenter_longitude
    # factor[:, 16].reshape(-1, 1), # epicenter_latitude
    factor[:, 17].reshape(-1, 1), # N
    ), axis=1).reshape(-1, 1)
print('input=', input_data0.shape)
input_data = input_data0[:-time_window,:]
# input_data = input_data[:,:features]
output_data = input_data0[time_window:,:] 
print('input=', input_data.shape, 'output=', output_data.shape)

if loc_block <= 10: n_out = 1
else: n_out = loc_block//10
print('\tn_out={}'.format(n_out))
if loc_block<=10:
    output_data = output_data[:,loc_block-1].reshape(-1,n_out)
elif loc_block==55:
    output_data = output_data[:,1:].reshape(-1,n_out)
elif loc_block==66: output_data = output_data

num = int(len(input_data) * split_ratio)

output_data = np.power(output_data,index)
if energy:
    output_data = np.around(np.sqrt(np.power(10, 1.5*output_data+11.8)), 0)

x_scaler = MinMaxScaler().fit(input_data) 
input_data = x_scaler.transform(input_data)
y_scaler = MinMaxScaler().fit(output_data)
output_data = y_scaler.transform(output_data)

num = int(len(input_data) * split_ratio)
x_train, y_train = input_data[:num, :], output_data[:num, :]
x_test, y_test = input_data[num:, :], output_data[num:, :]
print('\n\tshape: ', x_train.shape, y_train.shape,
    x_test.shape, y_test.shape)

x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)
print(f'x_train_val.shape={x_train.shape}, y_train.shape={y_train.shape}\n'+\
    f'x_test.shape={x_test.shape}, y_test.shape={y_test.shape}')

if model_name=='lstm':
    model = ANN.LSTM(x=x_train, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='cnn':
    model = ANN.CNN(x=x_train, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='cnn_lstm':
    model = ANN.cnn_lstm(x=x_train, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='lstm_cnn':
    model = ANN.lstm_cnn(x=x_train, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='tcn':
    model = ANN.tcn(x=x_train, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='tcn_lstm':
    model = ANN.tcn_lstm(x=x_train, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='tcn_cnn':
    model = ANN.tcn_cnn(x=x_train, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='tcn_lstm_cnn':
    model = ANN.tcn_lstm_cnn(x=x_train, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
elif model_name=='tcn_cnn_lstm':
    model = ANN.tcn_cnn_lstm(x=x_train, n_out=n_out, layer=layer, rate=rate,
        layer_size=layer_size, weight_decay=weight_decay)
print('\t model.summary(): \n{} '.format(model.summary()))
print('\t layer nums:', len(model.layers))

# RMSprop Adam sgd adagrad adadelta adamax nadam
optimizer = optimizers.Adam(learning_rate=learning_rate) 
model.compile(optimizer=optimizer, loss='mean_squared_error',
    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
history = model.fit(x_train, y_train, 
	validation_data=(x_test, y_test),
	# validation_data=(x_train, y_train),
	epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False, 
	callbacks=utils.checkpoints(model_name,file_name))

model_dir = os.path.join(file_location, f'{model_name}')
if not os.path.exists(model_dir): os.mkdir(model_dir)
model_dir = os.path.join(model_dir, f'{file_name}')
if not os.path.exists(model_dir): os.mkdir(model_dir)
model.save(model_dir+f'\{file_name}-layer{layer:.0f}-layer_size{layer_size:.0f}-loc_block{loc_block:.0f}.h5')

model = tf.keras.models.load_model(
    model_dir+f'\{file_name}-layer{layer:.0f}-layer_size{layer_size:.0f}-loc_block{loc_block:.0f}.h5')
train_loss, train_rmse = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_rmse = model.evaluate(x_test, y_test, verbose=2)

y_train = y_scaler.inverse_transform(y_train)
y_train_pre = model.predict(x_train)
y_train_pre = y_train_pre.reshape(-1, n_out)
y_train_pre = y_scaler.inverse_transform(y_train_pre)
y_test = y_scaler.inverse_transform(y_test)
y_test_pre = model.predict(x_test)
y_test_pre = y_test_pre.reshape(-1, n_out)
y_test_pre = y_scaler.inverse_transform(y_test_pre)

len_train = len(y_train.reshape(-1,))
len_test = len(y_test.reshape(-1,))

y_train = np.power(y_train, 1/index).reshape(-1,)
y_train_pre = np.power(y_train_pre, 1/index).reshape(-1,)
y_test = np.power(y_test, 1/index).reshape(-1,)
y_test_pre = np.power(y_test_pre, 1/index).reshape(-1,)

if config.energy:
    y_train = (2*np.log10(y_train)-11.8) / 1.5
    y_train_pre = (2*np.log10(y_train_pre)-11.8) / 1.5
    y_test = (2*np.log10(y_test)-11.8) / 1.5
    y_test_pre = (2*np.log10(y_test_pre)-11.8) / 1.5

print(f'{train_loss:.4f} & {train_rmse:.4f} & {test_loss:.4f} & {test_rmse:.4f}')

fig = plt.figure(figsize=(6, 2))  
plt.text(0.02, 0.05, 
    'Evaluate on train data:\nMSE={0:.4f}, RMSE={1:.4f}'.format(
    train_loss, train_rmse), fontproperties=times, fontsize=14)
plt.text(0.02, 0.02, 
    'Evaluate on test data:\nMSE={0:.4f}, RMSE={1:.4f}'.format(
    test_loss, test_rmse), fontproperties=times, fontsize=14)
plt.ylim(0, 0.08)
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.tight_layout()
plt.savefig(file_location+r'.\figure\{}-EvaluateTest.png'.format(file_name))
plt.show() 

plt.figure(figsize=(8, 4))
plt.grid(True, linestyle='--')      
plt.plot(jul_date[:len(y_train)], y_train, label='observed', 
    marker='o', markersize=4, 
    color="grey", markerfacecolor='white')
plt.plot(jul_date[len(y_train):len(y_train)+len(y_test)], y_test, 
    marker='o', markersize=4, 
    color="grey", markerfacecolor='white')
plt.plot(jul_date[:len(y_train)], y_train_pre, label='predicted-train', 
    marker='s', markersize=4, 
    color="dodgerblue", markerfacecolor='white') 
plt.plot(jul_date[len(y_train):len(y_train)+len(y_test)], y_test_pre, 
    # marker='*', c='indianred', label='predicted-test', s=20)
    marker='*', markersize=4, label='predicted-test',
    color="indianred", markerfacecolor='white') 
error1 = tf.keras.metrics.mean_absolute_error(
    y_train, y_train_pre).numpy()
error2 = tf.keras.metrics.mean_absolute_error(
    y_test, y_test_pre).numpy()
plt.fill_between(jul_date[:len(y_train)], 
    y_train_pre-error1, y_train_pre+error1, 
    alpha=0.5, color='orange',label='confidence interval') 
plt.fill_between(jul_date[len(y_train):len(y_train)+len(y_test)],
    y_test_pre-error2, y_test_pre+error2, 
    alpha=0.5, color='orange') 
# plt.scatter(jul_date[:len(y_train)], y_train, 
#     marker='o', c='grey', label='observed', s=10)
# plt.scatter(jul_date[len(y_train):len(y_train)+len(y_test)], y_test, 
#     marker='o', c='grey', s=10)
# plt.scatter(jul_date[:len(y_train)], y_train_pre, 
#     marker='+', c='dodgerblue', label='predicted (train)', s=30)    
# plt.scatter(jul_date[len(y_train):len(y_train)+len(y_test)], y_test_pre, 
#     marker='*', c='indianred', label='predicted (test)', s=20)
plt.xlabel('Date', fontproperties=times, fontsize=18)  
plt.ylabel('Magnitude', fontproperties=times, fontsize=18)  
plt.tick_params(labelsize=16)
# plt.ylim(2.1, 8.1)
plt.xticks(x_label1, x_label_tick1)
# plt.xlim(min(x_label1)-1000, max(x_label1)+1000)
plt.legend(loc='upper left', prop=FontProperties(
    fname=r'C:\WINDOWS\Fonts\times.ttf',size=14)) 
ax = plt.gca()
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout()
plt.savefig(figure_location+f'\\seism-{file_name}-layer{layer:.0f}-layer_size{layer_size:.0f}-loc_block{loc_block:.0f}-PredictM.png')
plt.show()

if not os.path.exists(file_location+r'\loss'):
    os.mkdir(file_location+r'\loss')
utils.save_data(file_location=file_location+r'\loss', 
    name='loss-{}'.format(file_name), value=history.history['loss'])    
utils.save_data(file_location=file_location+r'\loss', 
    name='rmse-{}'.format(file_name), value=history.history['rmse'])  
utils.save_data(file_location=file_location+r'\loss', 
    name='val_loss-{}'.format(file_name), value=history.history['val_loss'])    
utils.save_data(file_location=file_location+r'\loss', 
    name='val_rmse-{}'.format(file_name), value=history.history['val_rmse'])  

# weights = model.get_layer(name='hidden_layer1').get_weights()
# print(len(weights), weights[0].shape, weights[1].shape, len(weights[2]), )
# np.savetxt('./weight/w.txt', weights[0], delimiter=' ')

print((time.time()-start_time)/60, 'minutes')

