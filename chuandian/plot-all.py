import config
time = config.time
np = config.np
pd = config.pd
math = config.math
optimizers = config.optimizers
MinMaxScaler = config.MinMaxScaler
TimeSeriesSplit = config.TimeSeriesSplit
matplotlib = config.matplotlib
plt = config.plt
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
matplotlib.rcParams['axes.unicode_minus'] = False
fontcn = config.fontcn
fonten = config.fonten
file_location = config.file_location

start_time = time.time()

model = tf.keras.models.load_model(file_location+r'\model_lstm\{}.h5'.format(filename))
factor = pd.read_csv(file_location+r'\factor\factor-'+catolog_name+r'.txt', 
    delimiter=' ', header=None, dtype=np.float32).values
print('\n  initial factor shape: ', factor.shape)

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
print(factor.shape)
input_data = factor.reshape(-1, blocks*(features*n+m))
print('\n  input_data', input_data.shape, 'output_data', output_data.shape)
M_inital = output_data.reshape(-1, 1)
output_data = np.power(output_data, index)
if config.energy:
    output_data = np.around(np.sqrt(np.power(10, 1.5*output_data+11.8)), 0)

x_scaler = MinMaxScaler(feature_range=(0, 1))
input_data = x_scaler.fit_transform(input_data).reshape(-1, blocks*(features*config.n+config.m))
y_scaler = MinMaxScaler(feature_range=(0, 1)) 
output_data = y_scaler.fit_transform(output_data).reshape(-1, blocks)  

split_ratio2 = (1-split_ratio)/2
num1 = int(len(input_data) * split_ratio)
num2 = int(len(input_data) * split_ratio2)
# len_train_val = (n_splits*len(input_data))//(n_splits+1) + len(input_data)%(n_splits+1)
if n_splits != 1:
    len_train_val = len(input_data)//(n_splits+1)
    x_train, y_train = input_data[:(num1+num2)-len_train_val, :], \
        output_data[:(num1+num2)-len_train_val, :]
    x_val, y_val = input_data[(num1+num2)-len_train_val:(num1+num2), :], \
        output_data[(num1+num2)-len_train_val:(num1+num2), :]
    x_test, y_test = input_data[(num1+num2):, :], output_data[(num1+num2):, :]

    # x_train, y_train = input_data[:num1, :], output_data[:num1, :]
    # x_val, y_val = input_data[num1:(num1+num2), :], output_data[num1:(num1+num2), :]
    # x = np.concatenate((x_train, x_val), axis=0)
    # y = np.concatenate((y_train, y_val), axis=0)
    # x_test, y_test = input_data[(num1+num2):, :], output_data[(num1+num2):, :]
    x_train, y_train = input_data[:num1-len_train_val, :], output_data[:num1-len_train_val, :]
    x_val, y_val = input_data[num1-len_train_val:num1, :], output_data[num1-len_train_val:num1, :]
    x_test, y_test = input_data[num1:, :], output_data[num1:, :]

else:
    x_train, y_train = input_data[:num1, :], output_data[:num1, :]
    x_val, y_val = input_data[num1:(num1+num2), :], output_data[num1:(num1+num2), :]
    x_test, y_test = input_data[(num1+num2):, :], output_data[(num1+num2):, :]
print('\n\tshape: ', x_train.shape, y_train.shape, x_val.shape, y_val.shape, 
    x_test.shape, y_test.shape)

series_length = 1
x_train = x_train.reshape((-1, series_length, x_train.shape[1]))
x_val = x_val.reshape((-1, series_length, x_val.shape[1]))
x_test = x_test.reshape((-1, series_length, x_test.shape[1]))
y_train = y_train.reshape(-1, blocks)
y_val = y_val.reshape(-1, blocks)
y_test = y_test.reshape(-1, blocks)

batch_size = config.batch_size
train_size = (x_train.shape[0] // batch_size) * batch_size
val_size = (x_val.shape[0] // batch_size) * batch_size
test_size = (x_test.shape[0] // batch_size) * batch_size
x_train, y_train = x_train[0:train_size,:,:], y_train[0:train_size,:]
x_val, y_val = x_val[0:val_size,:,:], y_val[0:val_size,:]
x_test, y_test = x_test[0:test_size,:,:], y_test[0:test_size,:]
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

y_train = y_scaler.inverse_transform(y_train)
y_train_pre = model.predict(x_train)
y_train_pre = y_scaler.inverse_transform(y_train_pre)
y_val = y_scaler.inverse_transform(y_val)
y_val_pre = model.predict(x_val)
y_val_pre = y_scaler.inverse_transform(y_val_pre)
y_test = y_scaler.inverse_transform(y_test)
y_test_pre = model.predict(x_test)
y_test_pre = y_scaler.inverse_transform(y_test_pre)

y_train = np.power(y_train, 1/index)
y_train_pre = np.power(y_train_pre, 1/index)
y_val = np.power(y_val, 1/index)
y_val_pre = np.power(y_val_pre, 1/index)
y_test = np.power(y_test, 1/index)
y_test_pre = np.power(y_test_pre, 1/index)

if config.energy: 
    y_train = (2*np.log10(y_train)-11.8) / 1.5
    y_train_pre = (2*np.log10(y_train_pre)-11.8) / 1.5
    y_val = (2*np.log10(y_val)-11.8) / 1.5
    y_val_pre = (2*np.log10(y_val_pre)-12) / 1.5
    y_test = (2*np.log10(y_test)-11.8) / 1.5
    y_test_pre = (2*np.log10(y_test_pre)-11.8) / 1.5

x = np.concatenate((x_train, x_val), axis=0)
y = np.concatenate((y_train, y_val), axis=0)
y_pre = np.concatenate((y_train_pre, y_val_pre), axis=0)

diff_train = (y_train_pre-y_train).reshape(-1, blocks)
diff_val = (y_val_pre-y_val).reshape(-1, blocks)
diff_test = (y_test_pre-y_test).reshape(-1, blocks)
diff_all = np.concatenate((diff_train, diff_val, diff_test), axis=0)
diff_train_val = np.concatenate((diff_train, diff_val), axis=0)

y_all = np.concatenate((y_train, y_val, y_test), axis=0)
y_all_pre = np.concatenate((y_train_pre, y_val_pre, y_test_pre), axis=0)

print('\n\tshape: ', x.shape, y.shape, x_test.shape, y_test.shape)

linestyles = [':', '-', '--']
labels = []
for i in np.arange(blocks):
    labels.append('block{}'.format(i+1))
colors = ['blueviolet', 'green', 'blue', 'goldenrod', 'cyan']
markers = ['p', 'd', 'v', '^', 'x', 'o', '+', '<', '>', 's', '*', 'P']

diff_all_blocks = diff_all[np.where(y_all>=np.min(y_all))].reshape(-1, 1)
diff6 = diff_all[np.where(y_all>=6)].reshape(-1, 1)
diff7 = diff_all[np.where(y_all>=7)].reshape(-1, 1)

x_label = np.arange(-2.0, 2.0+0.01, 0.1)
num_list1 = []
num_list2 = []
num_list3 = []
for i, value in enumerate(x_label):
    if i == 0:
        num_list1.append(np.sum(diff_all_blocks<=value))
        num_list2.append(np.sum(diff6<=value))
        num_list3.append(np.sum(diff7<=value))
    elif i == len(x_label)-1:
        num_list1.append(np.sum(diff_all_blocks>value))
        num_list2.append(np.sum(diff6>value))
        num_list3.append(np.sum(diff7>value))
    else:
        num_list1.append(np.sum(np.logical_and(diff_all_blocks<=x_label[i], diff_all_blocks>x_label[i-1])))
        num_list2.append(np.sum(np.logical_and(diff6<=x_label[i], diff6>x_label[i-1])))
        num_list3.append(np.sum(np.logical_and(diff7<=x_label[i], diff7>x_label[i-1])))

fake = np.argwhere(diff_all.reshape(-1, )>0.5)
miss = np.argwhere(diff_all.reshape(-1, )<-0.5)
acc = np.argwhere(np.absolute(diff_all.reshape(-1, ))<=0.5)
fake_test = np.argwhere(diff_test.reshape(-1, )>0.5)
miss_test = np.argwhere(diff_test.reshape(-1, )<-0.5)
acc_test = np.argwhere(np.absolute(diff_test.reshape(-1, ))<=0.5)
fake_val = np.argwhere(diff_val.reshape(-1, )>0.5)
miss_val = np.argwhere(diff_val.reshape(-1, )<-0.5)
acc_val = np.argwhere(np.absolute(diff_val.reshape(-1, ))<=0.5)

fig = plt.figure(figsize=(14, 6))   
plt.title(r'Testing Set', fontdict=fonten, fontsize=20, color='red')
fig.add_subplot(1,1,1)
plt.grid(True, linestyle='--', linewidth=1)
for key, value in enumerate([0, 2, 18, 20]):
    plt.plot(np.arange(len(y_test)),  y_test[:, value], linewidth=2, 
        marker=markers[key], label='Observed Block{}'.format(value+1), color=colors[key])   
    plt.plot(np.arange(len(y_test_pre)), y_test_pre[:, value], linewidth=2,
        linestyle=':', marker=markers[key+6],
        label='Predicted Block{}'.format(value+1), color=colors[key])   
plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)  
plt.ylabel('max. Magnitude', fontdict=fonten, fontsize=18)  
plt.legend(loc='upper left', fontsize=12) 
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)   
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
plt.tight_layout()
plt.savefig(r'.\figure\{}-Histgram-test.png'.format(filename))
plt.show()

fig = plt.figure(figsize=(14, 7))   
plt.title(r'All Set', fontdict=fonten, fontsize=20, color='red')
plt.grid(True, linestyle='--', linewidth=1)
for key, value in enumerate([0, 2, 18, 20]):
    # plt.plot(np.arange(len(y_all)),  y_all[:, value], linewidth=2, 
    #     label='Observed Block{}'.format(value+1), color=colors[key])   
    plt.plot(np.arange(len(y_all_pre)), y_all_pre[:, value], linewidth=2,
        linestyle=':', label='Predicted Block{}'.format(value+1), color=colors[key])      
plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)  
plt.ylabel('max. Magnitude', fontdict=fonten, fontsize=18)  
plt.legend(loc='upper left', fontsize=10) 
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)   
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
ax.xaxis.set_major_locator(plt.MultipleLocator(24))
plt.tight_layout()
plt.savefig(r'.\figure\{}-Histgram.png'.format(filename))
plt.show()

fig = plt.figure(figsize=(14, 6)) 
plt.suptitle(r'Testing Set', fontdict=fonten, fontsize=20, color='red')  
fig.add_subplot(1,2,1)
plt.grid(True, linestyle='--', linewidth=1.0)
plt.hist(diff_test, bins=11, range=(-2.25, 2.25), stacked=True, label=labels)
plt.xlabel('Predicted - Observed', fontdict=fonten, fontsize=18)  
plt.ylabel('Frequency', fontdict=fonten, fontsize=18)  
plt.title(r'Absolute Error', fontdict=fonten, fontsize=20)
plt.legend(loc='upper left', fontsize=9) 
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
fig.add_subplot(1,2,2)
plt.grid(True, linestyle='--', linewidth=1.0)
plt.hist(diff_test/y_test, bins=11, range=(-0.425, 0.425), stacked=True, label=labels)
plt.xlabel('(Predicted - Observed) / Observed', fontdict=fonten, fontsize=18)  
plt.ylabel('Frequency', fontdict=fonten, fontsize=18)  
plt.title(r'Relative Error', fontdict=fonten, fontsize=20)
plt.legend(loc='upper left', fontsize=9) 
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.subplots_adjust(wspace=0.2)       
plt.savefig(r'.\figure\{}-Frequency-test.png'.format(filename))
plt.show()

fig = plt.figure(figsize=(14, 6))   
plt.suptitle(r'All Set', fontdict=fonten, fontsize=20, color='red')
fig.add_subplot(1,2,1)
plt.grid(True, linestyle='--', linewidth=1.0)
plt.hist(diff_all, bins=11, range=(-2.25, 2.25), stacked=True, label=labels)
plt.xlabel('Predicted - Observed', fontdict=fonten, fontsize=18)  
plt.ylabel('Frequency', fontdict=fonten, fontsize=18)  
plt.title(r'Absolute Error', fontdict=fonten, fontsize=20)
plt.legend(loc='upper right', fontsize=9) 
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
fig.add_subplot(1,2,2)
plt.grid(True, linestyle='--', linewidth=1.0)
plt.hist(diff_all/y_all, bins=11, range=(-0.425, 0.425), stacked=True, label=labels)
plt.xlabel('(Predicted - Observed) / observed', fontdict=fonten, fontsize=18)  
plt.ylabel('frequency', fontdict=fonten, fontsize=18)  
plt.title(r'Relative Error', fontdict=fonten, fontsize=20)
plt.legend(loc='upper right', fontsize=9) 
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.subplots_adjust(wspace=0.2)   
plt.savefig(r'.\figure\{}-Frequency.png'.format(filename))
plt.show()


fig = plt.figure(figsize=(9, 6))   
plt.grid(True, linestyle='--', linewidth=1.0)
plt.bar(x=x_label, height=np.array(num_list1), width=0.1, 
    color='indianred', label='M>=3')
plt.bar(x=x_label, height=np.array(num_list2), width=0.1, 
    color='dodgerblue', label='M>=6')
plt.bar(x=x_label, height=np.array(num_list3), width=0.1, 
    color='darkviolet', label='M>=7')
plt.xlabel('Predicted - Observed', fontdict=fonten, fontsize=18)  
plt.ylabel('Frequency', fontdict=fonten, fontsize=18)  
plt.legend(loc='upper left', prop=fonten, fontsize=15) 
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)  
plt.title(r'All Set', fontdict=fonten, fontsize=20, color='red')
plt.tight_layout()
plt.savefig(r'.\figure\{}-Frequency367.png'.format(filename))
plt.show()   

fig = plt.figure(figsize=(5.2, 2.8))  
plt.text(0.02, 0.4, '$M>=3: \mu={:.4f}$, $\sigma={:.4f}$'.format(
    np.mean(np.array(diff_all_blocks)[:, 0], axis=0), 
    np.std(np.array(diff_all_blocks)[:, 0], axis=0)), 
    fontdict=fonten, fontsize=14)
plt.text(0.02, 0.3, '$M>=6: \mu={:.4f}$, $\sigma={:.4f}$'.format(
    np.mean(np.array(diff6)[:, 0], axis=0), np.std(np.array(diff6)[:, 0], axis=0)), 
    fontdict=fonten, fontsize=14)
plt.text(0.02, 0.2, '$M>=7: \mu={:.4f}$, $\sigma={:.4f}$'.format(
    np.mean(np.array(diff7)[:, 0], axis=0), np.std(np.array(diff7)[:, 0], axis=0)), 
    fontdict=fonten, fontsize=14)    
plt.ylim(0.18, 0.45)
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.title(r'All Set', fontdict=fonten, fontsize=20, color='red')
plt.tight_layout()
plt.savefig(r'.\figure\{}-MuSigma.png'.format(filename))
plt.show() 

fig = plt.figure(figsize=(8, 6))  
for i, threshold in enumerate([6, 7]):
    plt.subplot(1,2,i+1)  
    TP = np.sum(np.logical_and(y_all_pre>=threshold, y_all>=threshold)) # 报震有震
    FN = np.sum(np.logical_and(y_all_pre<threshold, y_all>=threshold)) # 报无有震
    TP_FN = TP + FN
    FP = np.sum(np.logical_and(y_all_pre>=threshold, y_all<threshold)) # 报震无震
    TN = np.sum(np.logical_and(y_all_pre<threshold, y_all<threshold)) # 报无无震
    FP_TN = FP + TN
    TP_FP = TP + FP
    FN_TN = FN + TN
    N = TP_FN + FP_TN
    ACC = (TP+TN) / (TP+TN+FP+FN)
    P0 = TN / (TN+FN)
    P1 = TP / (TP+FP)
    Sn = TP / (TP+FN)
    Sp = TN / (TN+FP)
    Avg = 0.25 * (P0+P1+Sn+Sp)
    MCC = (TP*TN - FP*FN) / (((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**(1/2))
    R_Score = TP/TP_FN - FP/FP_TN
    F1_Score = 2 / (1/P1 + 1/Sn)
    plt.text(0.015, 1.95, 'M >= {}'.format(threshold), va='bottom', fontsize=15)
    plt.text(0.015, 1.8, '精确度 ACC = {:.2f}%'.format(100*ACC), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 1.7, '无震报准率 P0 = {:.2f}%'.format(100*P0), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 1.6, '有震报准率 P1 = {:.2f}%'.format(100*P1), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 1.5, '敏感度 Sn = {:.2f}%'.format(100*Sn), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 1.4, '特异度 Sp = {:.2f}%'.format(100*Sp), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 1.3, '平均值 Avg = {:.2f}%'.format(100*Avg), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 1.2, 'MCC = {:.2f}%'.format(100*MCC), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 1.1, '调和平均数 F1_Score = {:.2f}%'.format(100*F1_Score), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 1.0, '综合评分 R_Score = {:.2f}'.format(R_Score), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 0.9, '{}'.format('-'*28), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 0.8, 'TP = {}'.format(TP), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 0.7, 'FN = {}'.format(FN), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 0.6, 'TP+FN = {}'.format(TP_FN), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 0.5, 'FP = {}'.format(FP), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 0.4, 'TN = {}'.format(TN), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 0.3, 'FP+TN = {}'.format(FP_TN), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 0.2, 'TP+FP = {}'.format(TP_FP), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 0.1, 'FN+TN = {}'.format(FN_TN), va='bottom', fontsize=14, fontdict=fontcn)
    plt.text(0.015, 0.0, 'N = {}'.format(N), va='bottom', fontsize=14, fontdict=fontcn)
    plt.ylim(-0.1, 2.1)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
plt.suptitle(r'All Set', fontdict=fonten, fontsize=20, color='red')
plt.tight_layout()
plt.savefig(r'.\figure\{}-ConfuseMatrix.png'.format(filename))
plt.show()

fig = plt.figure(figsize=(8.2, 2.5))
total_M = diff_all_blocks.shape[0]*diff_all_blocks.shape[1]
accuracy_M = np.sum(np.absolute(diff_all_blocks)<=0.5)
ratio_M = np.sum(np.absolute(diff_all_blocks)<=0.5) / total_M
plt.text(0.02, 0.4, 'M>=3: 地震数量={}, 准确预测数量={}, 占比={:.2f}%'.format(
    total_M, accuracy_M, 100*ratio_M), va='bottom', fontsize=16, fontdict=fontcn)
for i in np.arange(6, 8+1, 1):
    total_M = np.sum(M_inital>=i)
    accuracy_M = np.sum(np.logical_and(np.absolute(diff_all_blocks)<=0.5, M_inital>=i))
    ratio_M = np.sum(np.logical_and(np.absolute(
        diff_all_blocks)<=0.5, M_inital>=i)) / total_M
    plt.text(0.02, 0.3-0.1*(i-6), 'M>={}: 地震数量={}, 准确预测数量={}, 占比={:.2f}%'.\
        format(i, total_M, accuracy_M, 100*ratio_M), 
        va='bottom', fontsize=16, fontdict=fontcn)
plt.ylim(0.07, 0.5)
plt.xlim(0, 1)
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.title(r'All Set', fontdict=fonten, fontsize=20, color='red')
plt.tight_layout()
plt.savefig(r'.\figure\{}-Ratio367.png'.format(filename))
plt.show()

fig = plt.figure(figsize=(7, 7))   
plt.grid(True, linestyle='--', linewidth=1.0)
l1 = plt.plot(y_all, y_all, c='indianred', linewidth=1.5)
l2 = plt.plot(y_all-0.5, y_all, c='grey', linewidth=1.5)
l3 = plt.plot(y_all+0.5, y_all, c='grey', linewidth=1.5)
plt.text(7.8, 8.1, 'y=x', fontdict=fonten, fontsize=16)
if n_splits == 1:
    plt.scatter(y_train, y_train_pre, marker='o', c='', edgecolors='dodgerblue', label='Training Set', s=10)    
    plt.scatter(y_val, y_val_pre, marker='x', c='darkviolet', label='Validation data', s=25)
else:
    plt.scatter(y, y_pre, marker='o', c='', edgecolors='dodgerblue', label='Training and Validation Set', s=10)    
plt.scatter(y_test, y_test_pre, marker='*', c='indianred', label='Testing Set', s=20) 
plt.xlabel('Observed', fontproperties='Arial', fontsize=18)  
plt.ylabel('Predicted', fontproperties='Arial', fontsize=18)  
plt.xticks(size=14)
plt.yticks(size=14)
plt.xlim(2.6, 9.1)
plt.ylim(2.6, 9.1)
plt.legend(loc='lower left', prop=fonten, fontsize=15) 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.set_aspect(aspect='equal')
plt.title(r'All Set', fontdict=fonten, fontsize=20, color='red')
plt.tight_layout()
plt.savefig(r'.\figure\{}-PredictedObserved.png'.format(filename))
plt.show()

fig = plt.figure(figsize=(9, 6))   
plt.subplot(1,1,1) 
plt.grid(True, linestyle='--', linewidth=1)
for i, value in enumerate(diff_test.reshape(-1, 1)):
    if i == fake[0]:
        plt.scatter(i, value, c='darkviolet', label='Fake Predict', marker='x', s=25)
    elif i == miss[0]:
        plt.scatter(i, value, c='indianred', label='Missing Predict', marker='+', s=30)
    elif i == acc[0]:
        plt.scatter(i, value, c='', edgecolors='dodgerblue', 
            label='Accuracy Predict', marker='o', s=10)
    else:
        if value > 0.5: # label='虚报'
            plt.scatter(i, value, c='darkviolet', marker='x', s=25)
        elif value < -0.5: # label='漏报'
            plt.scatter(i, value, c='indianred', marker='+', s=30)
        else: # label='准确预报'
            plt.scatter(i, value, c='', edgecolors='dodgerblue', marker='o', s=10)
plt.legend(loc='best', prop=fonten, fontsize=15) 
plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)  
plt.ylabel('Predicted - Observed', fontproperties='Arial', fontsize=18)  
plt.xticks(size=14)
plt.yticks(size=14)
plt.ylim(-3.2, 3.2)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.title(r'Testing Set', fontdict=fonten, fontsize=20, color='red')
plt.tight_layout()
plt.savefig(r'.\figure\{}-DiffPredictedObserved.png'.format(filename))
plt.show()

fig = plt.figure(figsize=(4, 2.5))  
plt.text(0.02, 0.8, '虚报{}次，虚报率={:.2f}%'.format(len(fake_test), \
    100*len(fake_test)/len(diff_test.reshape(-1,))), fontsize=16, fontdict=fontcn)
plt.text(0.02, 0.5, '漏报{}次，漏报率={:.2f}%'.format(len(miss_test), \
    100*len(miss_test)/len(diff_test.reshape(-1,))), fontsize=16, fontdict=fontcn)
plt.text(0.02, 0.2, '虚报比漏报多{}次'.format(len(fake_test)-len(miss_test)), fontsize=16, fontdict=fontcn)
plt.suptitle(r'Testing Set', fontdict=fonten, fontsize=20, color='red')
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.tight_layout()
plt.savefig(r'.\figure\{}-FakeMiss.png'.format(filename))
plt.show()

fig = plt.figure(figsize=(9, 6))   
plt.subplot(1,1,1) 
plt.grid(True, linestyle='--', linewidth=1)
for i, value in enumerate(diff_all.reshape(-1, 1)):
    if i == fake_test[0]:
        plt.scatter(i, value, c='darkviolet', label='Fake Predict', marker='x', s=25)
    elif i == miss_test[0]:
        plt.scatter(i, value, c='indianred', label='Missing Predict', marker='+', s=30)
    elif i == acc_test[0]:
        plt.scatter(i, value, c='', edgecolors='dodgerblue', 
            label='Accuracy Predict', marker='o', s=10)
    else:
        if value > 0.5: # label='虚报'
            plt.scatter(i, value, c='darkviolet', marker='x', s=25)
        elif value < -0.5: # label='漏报'
            plt.scatter(i, value, c='indianred', marker='+', s=30)
        else: # label='准确预报'
            plt.scatter(i, value, c='', edgecolors='dodgerblue', marker='o', s=10)
plt.legend(loc='best', prop=fonten, fontsize=15) 
plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)  
plt.ylabel('Predicted - Observed', fontproperties='Arial', fontsize=18)  
plt.title(r'All Set', fontdict=fonten, fontsize=20, color='red')
plt.xticks(size=14)
plt.yticks(size=14)
plt.ylim(-3.2, 3.2)
plt.axvline(x=(len(x_train)+len(x_test))*blocks, ls=':', c='green')
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig(r'.\figure\{}-DiffPredictedObserved.png'.format(filename))
plt.show()

fig = plt.figure(figsize=(4, 2.5))  
plt.text(0.02, 0.75, '虚报{}次，虚报率={:.2f}%'.format(len(fake), \
    100*len(fake)/len(diff_all_blocks)), fontsize=16, fontdict=fontcn)
plt.text(0.02, 0.45, '漏报{}次，漏报率={:.2f}%'.format(len(miss), \
    100*len(miss)/len(diff_all_blocks)), fontsize=16, fontdict=fontcn)
plt.text(0.02, 0.15, '虚报比漏报多{}次'.format(len(fake)-len(miss)), fontsize=16, fontdict=fontcn)
plt.title(r'All Set', fontdict=fonten, fontsize=20, color='red')
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.tight_layout()
plt.savefig(r'.\figure\{}-FakeMiss.png'.format(filename))
plt.show()

# fig = plt.figure(figsize=(10, blocks))  
# plt.suptitle(r'Testing Set', fontdict=fonten, fontsize=20, color='red')
# fig.add_subplot(1,2,1)
# plt.title(r'绝对误差', fontdict=fontcn, fontsize=16)
# for i in np.arange(blocks):
#     plt.text(0.02, 1-(i+0.7)/blocks, '区块{}：$\mu={:.4f}$, $\sigma={:.4f}$'.format(i+1,
#         np.mean(diff_test, axis=0)[i], np.std(diff_test, axis=0)[i]), 
#         fontdict=fontcn, fontsize=14)
# ax = plt.gca()
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.add_subplot(1,2,2)
# plt.title(r'相对误差', fontdict=fontcn, fontsize=16)
# for i in np.arange(blocks):
#     plt.text(0.02, 1-(i+0.7)/blocks, '区块{}: $\mu={:.4f}$, $\sigma={:.4f}$'.format(i+1,
#         np.mean(diff_test/y_test, axis=0)[i], np.std(diff_test/y_test, axis=0)[i]), 
#         fontdict=fontcn, fontsize=14)
# ax = plt.gca()
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# plt.tight_layout()
# plt.subplots_adjust(wspace=0.2, right=.7) 
# plt.tight_layout()
# plt.savefig(r'.\figure\{}-AbsoluteError-test.png'.format(filename))
# plt.show()

# fig = plt.figure(figsize=(10, blocks))  
# plt.suptitle(r'All Set', fontdict=fonten, fontsize=20, color='red')
# fig.add_subplot(1,2,1)
# plt.title(r'绝对误差', fontdict=fontcn, fontsize=16)
# for i in np.arange(blocks):
#     plt.text(0.02, 1-(i+0.7)/blocks, '区块{}：$\mu={:.4f}$, $\sigma={:.4f}$'.format(i+1,
#         np.mean(diff_all, axis=0)[i], np.std(diff_all, axis=0)[i]), 
#         fontdict=fontcn, fontsize=14)
# ax = plt.gca()
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.add_subplot(1,2,2)
# plt.title(r'相对误差 all', fontdict=fontcn, fontsize=16)
# for i in np.arange(blocks):
#     plt.text(0.02, 1-(i+0.7)/blocks, '区块{}: $\mu={:.4f}$, $\sigma={:.4f}$'.format(i+1,
#         np.mean(diff_all/y_all, axis=0)[i], np.std(diff_all/y_all, axis=0)[i]), 
#         fontdict=fontcn, fontsize=14)
# ax = plt.gca()
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# plt.tight_layout()
# plt.subplots_adjust(wspace=0.2, right=.7) 
# plt.tight_layout()
# plt.savefig(r'.\figure\{}-AbsoluteError.png'.format(filename))
# plt.show()


# fig = plt.figure(figsize=(16, 8))  
# plt.subplot(1,2,1) 
# plt.grid(True, linestyle='--', linewidth=1)
# plt.axvspan(xmin=6, xmax=9.1, alpha=0.2, facecolor='green')
# plt.axhspan(ymin=6, ymax=9.1, alpha=0.2, facecolor='yellow')
# plt.plot(y_all, y_all, c='indianred', linewidth=1.5)
# plt.plot(y_all, y_all+0.5, c='grey', linewidth=1.5)
# plt.plot(y_all, y_all-0.5, c='grey', linewidth=1.5)
# plt.text(7.8, 8.1, 'y=x', fontdict=fonten, fontsize=16)
# if n_splits == 1:
#     plt.scatter(y_train, y_train_pre, marker='o', c='', edgecolors='dodgerblue', label='training data', s=10)    
#     plt.scatter(y_val, y_val_pre, marker='x', c='darkviolet', label='validation data', s=25)
# else:
#     plt.scatter(y, y_pre, marker='o', c='', edgecolors='dodgerblue', label='training and validation data', s=10)    
# plt.scatter(y_test, y_test_pre, marker='*', c='indianred', label='testing data', s=20) 
# plt.xlabel('observed', fontdict=fonten, fontsize=18)  
# plt.ylabel('predicted', fontdict=fonten, fontsize=18)  
# plt.xticks(size=14)
# plt.yticks(size=14)
# plt.xlim(2.6, 9.1)
# plt.ylim(2.6, 9.1)
# plt.legend(loc='lower left', prop=fonten, fontsize=15) 
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# ax.set_aspect(aspect='equal')
# plt.subplot(1,2,2) 
# plt.grid(True, linestyle='--', linewidth=1)
# plt.axvspan(xmin=7, xmax=9.1, alpha=0.2, facecolor='green')
# plt.axhspan(ymin=7, ymax=9.1, alpha=0.2, facecolor='yellow')
# plt.plot(y_all, y_all, c='indianred', linewidth=1.5)
# plt.plot(y_all, y_all+0.5, c='grey', linewidth=1.5)
# plt.plot(y_all, y_all-0.5, c='grey', linewidth=1.5)
# plt.text(7.8, 8.1, 'y=x', fontdict=fonten, fontsize=16)
# if n_splits == 1:
#     plt.scatter(y_train, y_train_pre, marker='o', c='', edgecolors='dodgerblue', label='training data', s=10)    
#     plt.scatter(y_val, y_val_pre, marker='x', c='darkviolet', label='validation data', s=25)
# else:
#     plt.scatter(y, y_pre, marker='o', c='', edgecolors='dodgerblue', label='training and validation data', s=10)    
# plt.scatter(y_test, y_test_pre, marker='*', c='indianred', label='testing data', s=20) 
# plt.xlabel('observed', fontdict=fonten, fontsize=18)  
# plt.ylabel('predicted', fontdict=fonten, fontsize=18)  
# plt.xticks(size=14)
# plt.yticks(size=14)
# plt.xlim(2.6, 9.1)
# plt.ylim(2.6, 9.1)
# plt.legend(loc='lower left', prop=fonten, fontsize=15) 
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# ax.set_aspect(aspect='equal')
# plt.tight_layout()
# plt.savefig(r'.\figure\{}-PredictedObserved0.png'.format(filename))
# plt.show()