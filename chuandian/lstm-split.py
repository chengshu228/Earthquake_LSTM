import config
from ANN import stateless_lstm, stateful_lstm
from utils import checkpoints, save_data

os = config.os
time = config.time
np = config.np
pd = config.pd
optimizers = config.optimizers
MinMaxScaler = config.MinMaxScaler
tf = config.tf
tf.config.experimental.set_virtual_device_configuration(config.gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
plt = config.plt
# Patch = config.Patch

span_lat = config.span_lat
span_lon = config.span_lon
time_window = config.time_window
next_month = config.next_month
blocks = config.blocks 
features = config.features
index = config.index
m = config.m
n = config.n
n_splits = config.n_splits
split_ratio = config.split_ratio
epochs = config.epochs
learning_rate = config.learning_rate
filename = config.filename
catolog_name = config.catolog_name
epochs = config.epochs
reset_number = epochs
output_node = config.blocks
learning_rate = config.learning_rate
layer = config.layer
layer_size = config.layer_size
rate = config.rate
weight = config.weight
batch_size = config.batch_size
fontcn = config.fontcn
fonten = config.fonten
file_location = config.file_location

start_time = time.time()

seed = config.seed
np.random.seed(seed)
tf.random.set_seed(seed)

factor = pd.read_csv(file_location+r'\factor\factor-'+catolog_name+r'.txt', 
    delimiter=' ', header=None, dtype=np.float32).values
print('\n  initial factor shape: ', factor.shape, blocks*(features*n+m))
output_data = factor[:, 0].reshape(-1, blocks) 
factor = np.concatenate((
    factor[:, 1].reshape(-1, 1),  # frequency
    factor[:, 2].reshape(-1, 1),  # max_magnitude
    # factor[:, 3].reshape(-1, 1),  # mean_magnitude
    factor[:, 4].reshape(-1, 1), # b_lstsq
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
), axis=1)
features = factor.shape[1]
print(factor.shape)
input_data = factor.reshape(-1, blocks*(features*n+m))
print('\n  input_data=', input_data.shape, 'output_data=', output_data.shape)

output_data = np.power(output_data, index)
if config.energy:
    output_data = np.around(np.sqrt(np.power(10, 1.5*output_data+11.8)), 0)

x_scaler = MinMaxScaler(feature_range=(0, 1))
input_data = x_scaler.fit_transform(input_data).reshape(-1, blocks*(features*n+m))
y_scaler = MinMaxScaler(feature_range=(0, 1)) 
output_data = y_scaler.fit_transform(output_data).reshape(-1, blocks)  

split_ratio2 = (1-split_ratio)/2
num1 = int(len(input_data) * split_ratio)
num2 = int(len(input_data) * split_ratio2)

x_train, y_train = input_data[:num1, :], output_data[:num1, :]
x_val, y_val = input_data[num1:(num1+num2), :], output_data[num1:(num1+num2), :]
x_test, y_test = input_data[(num1+num2):, :], output_data[(num1+num2):, :]
print('\n\tshape: ', x_train.shape, y_train.shape, x_val.shape, y_val.shape, 
    x_test.shape, y_test.shape)

time_series_length = 1
x_train = x_train.reshape((x_train.shape[0], time_series_length, x_train.shape[1]))
x_val = x_val.reshape((x_val.shape[0], time_series_length, x_val.shape[1]))
x_test = x_test.reshape((x_test.shape[0], time_series_length, x_test.shape[1]))
y_train = y_train.reshape(y_train.shape[0], blocks)
y_val = y_val.reshape(y_val.shape[0], blocks)
y_test = y_test.reshape(y_test.shape[0], blocks)
print('\n\tshape: ', x_train.shape, y_train.shape, x_val.shape, y_val.shape, 
    x_test.shape, y_test.shape)

x = np.concatenate((x_train, x_val), axis=0)
y = np.concatenate((y_train, y_val), axis=0)

# model = stateless_lstm(x, output_node, layer=layer, 
#     layer_size=layer_size, rate=rate, weight=weight)
batch_size = 32
import ANN
# model = stateful_lstm(x, output_node=blocks, 
#     batch_size=batch_size,
#     layer=config.layer, layer_size=config.layer_size, 
#     rate=config.rate, weight=config.weight)
model = ANN.cnn_lstm(x, output_node=blocks, 
    layer=config.layer, layer_size=config.layer_size, 
    rate=config.rate, weight=config.weight)
print('\t model.summary(): \n{} '.format(model.summary()))
print('\t layer nums:', len(model.layers))

# RMSprop Adam sgd adagrad adadelta adamax nadam
optimizer = optimizers.Adam(learning_rate=learning_rate) 

model.compile(optimizer=optimizer, loss='mean_squared_error',
    metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

# history = model.fit(x_train, y_train, epochs=epochs, shuffle=False, batch_size=batch_size, 
#     verbose=2, validation_data=(x_val, y_val), callbacks=checkpoints())
train_size = (x_train.shape[0] // batch_size) * batch_size
val_size = (x_val.shape[0] // batch_size) * batch_size
test_size = (x_test.shape[0] // batch_size) * batch_size
x_train, y_train = x_train[0:train_size,:,:], y_train[0:train_size,:]
x_val, y_val = x_val[0:val_size,:,:], y_val[0:val_size,:]
x_test, y_test = x_test[0:test_size,:,:], y_test[0:test_size,:]
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)
for i in np.arange(epochs):
    print("Epoch {:d}/{:d}".format(i+1, epochs))
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
        verbose=2, epochs=1, shuffle=False,
        batch_size=batch_size, 
        callbacks=checkpoints()
        )
    model.reset_states()
model.save(file_location+r'\model_lstm\{}.h5'.format(filename))

test_loss, test_mae, test_rmse = model.evaluate(x_test, y_test, verbose=2)

y_train = y_scaler.inverse_transform(y_train)
y_train_pre = model.predict(x_train)
y_train_pre = y_scaler.inverse_transform(y_train_pre)
y_val = y_scaler.inverse_transform(y_val)
y_val_pre = model.predict(x_val)
y_val_pre = y_scaler.inverse_transform(y_val_pre)
y_test = y_scaler.inverse_transform(y_test)
y_test_pre = model.predict(x_test)
y_test_pre = y_scaler.inverse_transform(y_test_pre)

len_train = len(y_train.reshape(-1, ))
len_val = len(y_val.reshape(-1, ))
len_test = len(y_test.reshape(-1, ))

y_train = np.power(y_train, 1/index).reshape(-1, )
y_train_pre = np.power(y_train_pre, 1/index).reshape(-1, )
y_val = np.power(y_val, 1/index).reshape(-1, )
y_val_pre = np.power(y_val_pre, 1/index).reshape(-1, )
y_test = np.power(y_test, 1/index).reshape(-1, )
y_test_pre = np.power(y_test_pre, 1/index).reshape(-1, )

if config.energy:
    y_train = (2*np.log10(y_train)-11.8) / 1.5
    y_train_pre = (2*np.log10(y_train_pre)-11.8) / 1.5
    y_val = (2*np.log10(y_val)-11.8) / 1.5
    y_val_pre = (2*np.log10(y_val_pre)-12) / 1.5
    y_test = (2*np.log10(y_test)-11.8) / 1.5
    y_test_pre = (2*np.log10(y_test_pre)-11.8) / 1.5

fig = plt.figure(figsize=(6, 1))  
plt.text(0.02, 0.02, 
    'Evaluate on test data:\nMSE={0:.4f}, MAE={1:.4f}, RMSE={2:.4f}'.format(
    test_loss, test_mae, test_rmse), fontdict=fonten, fontsize=14)
plt.ylim(0, 0.08)
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.savefig(file_location+r'.\figure\{}-EvaluateTest.png'.format(filename))
plt.show() 

from pylab import *
plt.figure(figsize=(10, 6))
plt.grid(True, linestyle='--', linewidth=1.0)      
plt.scatter(np.arange(len_train), y_train, 
    marker='o', c='grey', label='observed', s=10)
plt.scatter(np.arange(len_train, len_train+len_val), y_val, 
    marker='o', c='grey', s=10)
plt.scatter(np.arange(len_train+len_val, len_train+len_val+len_test, 1), y_test, 
    marker='o', c='grey', s=10)
plt.scatter(np.arange(len_train), y_train_pre, 
    marker='+', c='dodgerblue', label='predicted (training data)', s=30)    
plt.scatter(np.arange(len_train, len_train+len_val), y_val_pre, 
    marker='x', c='darkviolet', label='predicted (validation data)', s=25)   
plt.scatter(np.arange(len_train+len_val, len_train+len_val+len_test, 1), y_test_pre, 
    marker='*', c='indianred', label='predicted (testing data)', s=20)
plt.xlabel('sample index', fontproperties='Arial', fontsize=18)  
plt.ylabel('the predicted max. magnitude', fontproperties='Arial', fontsize=18)  
plt.xticks(size=14)
plt.yticks(size=14)
plt.ylim(2.1, 9.1)
plt.legend(loc='lower left', prop=fonten, fontsize=15) 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5) 
plt.savefig(file_location+r'.\figure\{}-PredictM.png'.format(filename))
plt.show()

plt.figure(figsize=(10, 6))
plt.grid(True, linestyle='--', linewidth=1.0)      
plt.scatter(np.arange(len_train), y_train, 
    marker='o', c='grey', label='observed', s=10)
plt.scatter(np.arange(len_train, len_train+len_val), y_val, 
    marker='o', c='grey', s=10)
plt.scatter(np.arange(len_train), y_train_pre, 
    marker='+', c='dodgerblue', label='predicted (training data)', s=30)    
plt.scatter(np.arange(len_train, len_train+len_val), y_val_pre, 
    marker='x', c='darkviolet', label='predicted (validation data)', s=25)   
plt.xlabel('sample index', fontproperties='Arial', fontsize=18)  
plt.ylabel('the predicted max. magnitude', fontproperties='Arial', fontsize=18)  
plt.xticks(size=14)
plt.yticks(size=14)
plt.ylim(2.1, 9.1)
plt.legend(loc='lower left', prop=fonten, fontsize=15) 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5) 
plt.savefig(file_location+r'.\figure\{}-PredictM1.png'.format(filename))
plt.show()

plt.figure(figsize=(10, 6))
plt.grid(True, linestyle='--', linewidth=1.0)      
plt.scatter(np.arange(len_test), y_test, label='observed', 
    marker='o', c='grey', s=10)
plt.scatter(np.arange(len_test), y_test_pre, 
    marker='*', c='indianred', label='predicted (testing data)', s=20)
plt.xlabel('sample index', fontproperties='Arial', fontsize=18)  
plt.ylabel('the predicted max. magnitude', fontproperties='Arial', fontsize=18)  
plt.xticks(size=14)
plt.yticks(size=14)
plt.ylim(2.1, 9.1)
plt.legend(loc='lower left', prop=fonten, fontsize=15) 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5) 
plt.savefig(file_location+r'.\figure\{}-PredictM2.png'.format(filename))
plt.show()

if not os.path.exists(file_location+r'\loss'):
    os.mkdir(file_location+r'\loss')
save_data(file_location=file_location+r'\loss', 
    name='loss-{}'.format(filename), value=history.history['loss'])    
save_data(file_location=file_location+r'\loss', 
    name='mae-{}'.format(filename), value=history.history['mae'])    
save_data(file_location=file_location+r'\loss', 
    name='rmse-{}'.format(filename), value=history.history['rmse'])  
save_data(file_location=file_location+r'\loss', 
    name='val_loss-{}'.format(filename), value=history.history['val_loss'])    
save_data(file_location=file_location+r'\loss', 
    name='val_mae-{}'.format(filename), value=history.history['val_mae'])    
save_data(file_location=file_location+r'\loss', 
    name='val_rmse-{}'.format(filename), value=history.history['val_rmse'])  

# weights = model.get_layer(name='hidden_layer1').get_weights()
# print(len(weights), weights[0].shape, weights[1].shape, len(weights[2]), )
# np.savetxt('./weight/w.txt', weights[0], delimiter=' ')

print((time.time()-start_time)/60, 'minutes')

