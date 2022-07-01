import config
from ANN import stateless_lstm, stateful_lstm
from utils import checkpoints, save_data

time = config.time
os = config.os
np = config.np
pd = config.pd
optimizers = config.optimizers
MinMaxScaler = config.MinMaxScaler
TimeSeriesSplit = config.TimeSeriesSplit
tf = config.tf
tf.config.experimental.set_virtual_device_configuration(config.gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
plt = config.plt

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
fontcn = config.fontcn
fonten = config.fonten
file_location = config.file_location
batch_size = config.batch_size

start_time = time.time()
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

factor = pd.read_csv(file_location+r'\factor\factor-'+catolog_name+r'.txt', 
    delimiter=' ', header=None, dtype=np.float32)
factor = np.array(factor)
print('\n  initial factor shape: ', factor[:, 1:].shape, blocks*(features*n+m))
output_data = factor[:, 0].reshape(-1, blocks)
factor = np.concatenate((
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
features = factor.shape[1]
input_data = factor.reshape(-1, blocks*(features*n+m))
print('features=', features)
print('\n  input_data=', input_data.shape, 'output_data=', output_data.shape)

M_inital = output_data

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

# x_train, y_train = input_data[:num1, :], output_data[:num1, :]
# x_val, y_val = input_data[num1:(num1+num2), :], output_data[num1:(num1+num2), :]
# x = np.concatenate((x_train, x_val), axis=0)
# y = np.concatenate((y_train, y_val), axis=0)
# x_test, y_test = input_data[(num1+num2):, :], output_data[(num1+num2):, :]
x, y = input_data[:num1, :], output_data[:num1, :]
x_test, y_test = input_data[num1:, :], output_data[num1:, :]

series_length = 1
x = x.reshape((-1, series_length, blocks*(features*n+m)))
x_test = x_test.reshape((-1, series_length, blocks*(features*n+m)))
input_data = input_data.reshape((-1, series_length, blocks*(features*n+m)))
y = y.reshape(-1, blocks)
y_test = y_test.reshape(-1, blocks)
output_data = output_data.reshape(-1, blocks)

print('\n\tshape: ', x.shape, y.shape, 
    x_test.shape, y_test.shape)


fold = TimeSeriesSplit(n_splits=n_splits, max_train_size=None)

# fig = plt.figure(figsize=(10, 10))
# for i, (train_index, val_index) in enumerate(fold.split(x, y)):
#     l1 = plt.scatter(train_index, [i+1]*len(train_index), 
#         c='dodgerblue', marker='_', lw=14)
#     l2 = plt.scatter(val_index, [i+1]*len(val_index), 
#         c='darkviolet', marker='_', lw=14)
#     # plt.legend([Patch(color='dodgerblue'), Patch(color='indianred')],
#     #     ['Training Set', 'Validation Set'],
#         # prop=fonten, loc=(0.55, 0.8), fontsize=16)
#     plt.legend([l1, l2], ['Training set', 'Validation set'], \
#         prop=fonten, loc='upper right', fontsize=16)
#     plt.xlabel('Sample Index (Month)', fontproperties='Arial', fontsize=18)  
#     plt.ylabel('CV Iteration', fontproperties='Arial', fontsize=18)  
#     plt.title('Time Series Split', fontproperties='Arial', fontsize=20)    # Blocking
#     plt.xticks(size=14)
#     plt.yticks(size=14)
#     # plt.axvline(x=444, ls=':', c='green')
#     plt.text(1.2, i+1.15, '{} | {}'.format(len(train_index), len(val_index)), 
#         fontproperties='Arial', fontsize=16)
#     ax = plt.gca()
#     ax.spines['bottom'].set_linewidth(1.5)
#     ax.spines['left'].set_linewidth(1.5)
#     ax.spines['top'].set_linewidth(1.5)
#     ax.spines['right'].set_linewidth(1.5)
#     ax.set(ylim=[n_splits+0.9, 0.1])
#     ax.yaxis.set_major_locator(plt.MultipleLocator(1))
#     plt.tight_layout()
# plt.savefig(file_location+r'\figure\{}-CV.png'.format(filename))
# plt.show()

loss_per_fold = []
mae_per_fold = []
rmse_per_fold = []

# model = stateless_lstm(x, output_node=blocks, 
#         layer=config.layer, layer_size=config.layer_size, 
#         rate=config.rate, weight=config.weight)
# print('\tmodel.summary(): \n{} '.format(model.summary()))
# print('\tlayer nums:', len(model.layers))       

for fold_no, (train_index, val_index) in enumerate(fold.split(x, y)):
    model = stateful_lstm(x, output_node=config.blocks, 
        batch_size=config.batch_size,
        layer=config.layer, layer_size=config.layer_size, 
        rate=config.rate, weight=config.weight)
    print('\tmodel.summary(): \n{} '.format(model.summary()))
    print('\tlayer nums:', len(model.layers))    
    optimizer = optimizers.Adam(learning_rate=learning_rate) 
    model.compile(optimizer=optimizer, loss='mean_squared_error',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    print(f'\n\n\n\nTraining for Fold {fold_no+1} ...')
    x_train, y_train = x[train_index], y[train_index]
    x_val, y_val = x[val_index], y[val_index]
    # history = model.fit(x_train, y_train, 
    #     verbose=2, epochs=epochs, 
    #     batch_size=batch_size, 
    #     validation_data=(x_val, y_val), 
    #     callbacks=checkpoints(), 
    #     shuffle=False)  
    train_size = (x_train.shape[0] // batch_size) * batch_size
    val_size = (x_val.shape[0] // batch_size) * batch_size
    test_size = (x_test.shape[0] // batch_size) * batch_size
    x_train, y_train = x_train[0:train_size,:,:], y_train[0:train_size,:]
    x_val, y_val = x_val[0:val_size,:,:], y_val[0:val_size,:]
    x_test, y_test = x_test[0:test_size,:,:], y_test[0:test_size,:]
    for i in np.arange(100):
        print("\n\n Epoch {:d}/{:d}".format(i+1, 100))
        history = model.fit(x_train, y_train, 
            validation_data=(x_val, y_val),
            batch_size=batch_size, verbose=2, epochs=1, shuffle=False,
            callbacks=checkpoints()
            )
        model.reset_states()

    val_loss, val_mae, val_rmse = model.evaluate(x_val, y_val,
        batch_size=batch_size, verbose=2)
    print('Evaluate on validation data: ' + \
        'val_mse={0:.2f}%, val_mae={1:.2f}%, val_rmse={2:.2f}%'.\
        format(val_loss*100, val_mae*100, val_rmse*100))    
    loss_per_fold.append(val_loss)
    mae_per_fold.append(val_mae)
    rmse_per_fold.append(val_rmse)

    y_train = y_scaler.inverse_transform(y_train)
    y_train_pre = model.predict(x_train)
    y_train_pre = y_scaler.inverse_transform(y_train_pre)
    y_val = y_scaler.inverse_transform(y_val)
    y_val_pre = model.predict(x_val)
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
    #     plt.figure(figsize=(10, 6))
    #     plt.grid(True, linestyle='--', linewidth=1.0) 
    #     plt.scatter(np.arange(len_train), y_train, 
    #         marker='o', c='', edgecolors='grey', label='Observed', s=10)
    #     plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val, 
    #         marker='o', c='', edgecolors='grey', s=10)
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
    #     plt.legend(loc='best', prop=fonten, fontsize=15) 
    #     ax = plt.gca()
    #     ax.spines['bottom'].set_linewidth(1.5)
    #     ax.spines['left'].set_linewidth(1.5)
    #     ax.spines['top'].set_linewidth(1.5)
    #     ax.spines['right'].set_linewidth(1.5)
    #     plt.savefig(file_location+r'.\figure\{}-fold{}.png'.format(filename, fold_no+1))
        # plt.show()
        # elif fold_no+1 == config.n_splits:
    if fold_no+1 == config.n_splits:
        y_test = y_scaler.inverse_transform(y_test)
        y_test_pre = model.predict(x_test)
        y_test_pre = y_scaler.inverse_transform(y_test_pre)
        y_test = np.power(y_test, 1/index).reshape(-1, )
        y_test_pre = np.power(y_test_pre, 1/index).reshape(-1, )
        if config.energy:
            y_test = (2*np.log10(y_test)-11.8) / 1.5
            y_test_pre = (2*np.log10(y_test_pre)-11.8) / 1.5       
        plt.figure(figsize=(10, 6))
        plt.grid(True, linestyle='--', linewidth=1.0) 
        plt.scatter(np.arange(len_train), y_train, 
            marker='o', c='', edgecolors='grey', label='Observed', s=10)
        plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val, 
            marker='o', c='', edgecolors='grey', s=10)
        plt.scatter(np.arange(len_train), y_train_pre, 
            marker='+', c='dodgerblue', label='Predicted (Training Set)', s=30)    
        plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val_pre, 
            marker='x', c='darkviolet', label='Predicted (Validation Set)', s=25)
        plt.title(f'Training for Fold {fold_no+1}', fontproperties='Arial', fontsize=20, color='red')
        plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)         
        plt.ylabel('Predicted max. Magnitude', fontproperties='Arial', fontsize=18)  
        plt.xticks(size=14)
        plt.yticks(size=14)
        # plt.ylim(2.5, 8.5)
        plt.legend(loc='best', prop=fonten, fontsize=15) 
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        plt.savefig(file_location+r'.\figure\{}-fold{}.png'.format(filename, fold_no+1))
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.grid(True, linestyle='--', linewidth=1.0) 
        plt.scatter(np.arange(len_train+len_val, len_train+len_val+len_test), y_test, 
            marker='o', c='', edgecolors='grey', label='Observed', s=10)
        plt.scatter(np.arange(len_train+len_val, len_train+len_val+len_test), y_test_pre, 
            marker='*', c='indianred', label='Predicted (Testing Set)', s=20)
        plt.title('Testing Set', fontproperties='Arial', fontsize=20, color='red')
        plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)         
        plt.ylabel('Predicted max. Magnitude', fontproperties='Arial', fontsize=18)  
        plt.xticks(size=14)
        plt.yticks(size=14)
        # plt.ylim(2.5, 8.5)
        plt.legend(loc='best', prop=fonten, fontsize=15) 
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        plt.savefig(file_location+r'.\figure\{}-fold{}.png'.format(filename, fold_no+1))
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.grid(True, linestyle='--', linewidth=1.0) 
        plt.scatter(np.arange(len_train), y_train, 
            marker='o', c='', edgecolors='grey', label='Observed', s=10)
        plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val, 
            marker='o', c='', edgecolors='grey', s=10)
        plt.scatter(np.arange(len_train+len_val, len_train+len_val+len_test, 1), y_test, 
            marker='o', c='', edgecolors='grey', s=10)
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
        # plt.ylim(2.5, 8.5)
        plt.legend(loc='best', prop=fonten, fontsize=15) 
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        plt.savefig(file_location+r'\figure\{}-fold{}.png'.format(filename, fold_no+1))
        plt.show()

    # if not os.path.exists(file_location+r'\loss'):
    #     os.mkdir(file_location+r'\loss')
    # save_data(file_location=file_location+r'\loss', 
    #     name='loss-{}-{}'.format(filename, fold_no+1), value=history.history['loss'])    
    # save_data(file_location=file_location+r'\loss', 
    #     name='mae-{}-{}'.format(filename, fold_no+1), value=history.history['mae'])    
    # save_data(file_location=file_location+r'\loss', 
    #     name='rmse-{}-{}'.format(filename, fold_no+1), value=history.history['rmse'])  
    # save_data(file_location=file_location+r'\loss', 
    #     name='val_loss-{}-{}'.format(filename, fold_no+1), value=history.history['val_loss'])    
    # save_data(file_location=file_location+r'\loss', 
    #     name='val_mae-{}-{}'.format(filename, fold_no+1), value=history.history['val_mae'])    
    # save_data(file_location=file_location+r'\loss', 
    #     name='val_rmse-{}-{}'.format(filename, fold_no+1), value=history.history['val_rmse'])  

model.save(file_location+r'\model_lstm\{}.h5'.format(filename))

fig = plt.figure(figsize=(4, 3))  
plt.text(0.01, 0.7, 'Average scores for all folds:', va='bottom', fontsize=14)
plt.text(0.01, 0.5, f' MSE:{np.mean(loss_per_fold):.4f}', va='bottom', fontsize=14)
plt.text(0.01, 0.3, f' MAE:{np.mean(mae_per_fold):.4f}(+/-{np.std(mae_per_fold):.4f})', 
    va='bottom', fontsize=14)
plt.text(0.01, 0.1, f' RMSE:{np.mean(rmse_per_fold):.4f}(+/-{np.std(rmse_per_fold):.4f})', 
    va='bottom', fontsize=14)
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.savefig(file_location+r'\figure\error-{}.png'.format(filename))
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

