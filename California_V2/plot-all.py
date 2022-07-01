import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
import matplotlib.pyplot as plt
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
simsun,times = config.times,config.simsun
n_splits = config.n_splits
loc_block = config.loc_block
# n_out = config.blocks

start_time = time.time()
seed = config.seed
np.random.seed(seed)
tf.random.set_seed(seed)

matplotlib.rcParams.update(config.config_font)
plt.rcParams['font.sans-serif']=['simsun'] 
plt.rcParams['axes.unicode_minus']=False 
initial_julian = julian.to_jd(datetime.datetime(1930,1,1,0,0,0), fmt='jd')

# date = np.genfromtxt(data_location+f'\\date-{file_name}.txt', delimiter=' ')
date = np.genfromtxt(data_location+f'\\date-{file_name}-blocks1.txt', delimiter=' ')
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
for i in np.arange(time_window+next_month-1,len(date),1): 
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


# factor = np.genfromtxt(data_location+'\\factor-{file_name}.txt',delimiter=' ')
blocks =1
factor = np.genfromtxt(data_location+f'\\factor-{file_name}-blocks1.txt', delimiter=' ')

print('\n initial factor:', factor.shape, blocks*(features*n+m))

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
# input_data = input_data[:,:features]
output_data = factor[:,0].reshape(-1,blocks) 
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

model_dir = os.path.join(file_location, f'{model_name}')
if not os.path.exists(model_dir): os.mkdir(model_dir)
model_dir = os.path.join(model_dir, f'{loc_block}')
if not os.path.exists(model_dir): os.mkdir(model_dir)

model = tf.keras.models.load_model(model_dir+f'\layer{layer:.0f}-layer_size{layer_size:.0f}-loc_block{loc_block:.0f}.tf')

train_loss, train_rmse = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_rmse = model.evaluate(x_test, y_test, verbose=2)
print(f'{train_loss:.4f} & {train_rmse:.4f} & {test_loss:.4f} & {test_rmse:.4f}')

y_train = y_scaler.inverse_transform(y_train)
y_train_pre = model.predict(x_train)
y_train_pre = y_train_pre.reshape(-1, n_out)
y_train_pre = y_scaler.inverse_transform(y_train_pre)

y_test = y_scaler.inverse_transform(y_test)
y_test_pre = model.predict(x_test)
y_test_pre = y_test_pre.reshape(-1, n_out)
y_test_pre = y_scaler.inverse_transform(y_test_pre)

y_train = np.power(y_train, 1/index)
y_train_pre = np.power(y_train_pre, 1/index)
y_test = np.power(y_test, 1/index)
y_test_pre = np.power(y_test_pre, 1/index)

if config.energy: 
    y_train = (2*np.log10(y_train)-11.8) / 1.5
    y_train_pre = (2*np.log10(y_train_pre)-11.8) / 1.5
    y_test = (2*np.log10(y_test)-11.8) / 1.5
    y_test_pre = (2*np.log10(y_test_pre)-11.8) / 1.5

# x = np.concatenate((x_train, x_val), axis=0)
# y = np.concatenate((y_train, y_val), axis=0)
# y_pre = np.concatenate((y_train_pre, y_val_pre), axis=0)

diff_train = (y_train_pre-y_train).reshape(-1, n_out)
diff_test = (y_test_pre-y_test).reshape(-1, n_out)

diff_all = np.concatenate((diff_train, diff_test), axis=0)

y_all = np.concatenate((y_train, y_test), axis=0)
y_all_pre = np.concatenate((y_train_pre, y_test_pre), axis=0)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

linestyles = [':', '-', '--']
labels = []
for i in np.arange(blocks):
    labels.append('block{}'.format(i+1))
colors = ['blueviolet', 'green', 'blue', 'goldenrod', 'cyan', 'grey']
markers = ['p', 'd', 'v', '^', 'x', 'o', '+', '<', '>', 's', '*', 'P']
# location_block = [0, 2, blocks-3, blocks-1]
location_block = [0] 

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

loc_block = config.loc_block
if loc_block<=10: location_block=label=[loc_block]
elif loc_block==55: location_block=label=np.arange(1,5+1,1)
elif loc_block==66: location_block=label=np.arange(0,5+1,1)

if loc_block==55 or loc_block==66: label += 1

len_train = len(y_train.reshape(-1,))
len_test = len(y_test.reshape(-1,))
# test_loss, test_rmse = model.evaluate(x_test, y_test, verbose=2)

plt.figure(figsize=(8, 4))
plt.grid(True, linestyle='--')      
plt.plot(jul_date[:len(y_train)], y_train, label='observed', 
    marker='o', markersize=4, 
    color="grey", markerfacecolor='white')
# plt.plot(jul_date[len(y_train):len(y_train)+len(y_test)], y_test, 
#     marker='o', markersize=4, 
#     color="grey", markerfacecolor='white')
# plt.plot(jul_date[:len(y_train)], y_train_pre, label='predicted-train', 
#     marker='s', markersize=4, 
#     color="dodgerblue", markerfacecolor='white') 
# plt.plot(jul_date[len(y_train):len(y_train)+len(y_test)], y_test_pre, 
#     # marker='*', c='indianred', label='predicted-test', s=20)
#     marker='*', markersize=4, label='predicted-test',
#     color="indianred", markerfacecolor='white') 
# error1 = tf.keras.metrics.mean_absolute_error(
#     y_train, y_train_pre).numpy()
# error2 = tf.keras.metrics.mean_absolute_error(
#     y_test, y_test_pre).numpy()
# plt.fill_between(jul_date[:len(y_train)], 
#     y_train_pre.reshape(-1,)-error1, y_train_pre.reshape(-1,)+error1, 
#     alpha=0.5, color='orange',label='confidence interval') 
# plt.fill_between(jul_date[len(y_train):len(y_train)+len(y_test)],
#     y_test_pre.reshape(-1,)-error2, y_test_pre.reshape(-1,)+error2, 
#     alpha=0.5, color='orange') 
# plt.scatter(jul_date[:len(y_train)], y_train, 
#     marker='o', c='grey', label='observed', s=10)
# plt.scatter(jul_date[len(y_train):len(y_train)+len(y_test)], y_test, 
#     marker='o', c='grey', s=10)
# plt.scatter(jul_date[:len(y_train)], y_train_pre, 
#     marker='+', c='dodgerblue', label='predicted (train)', s=30)    
# plt.scatter(jul_date[len(y_train):len(y_train)+len(y_test)], y_test_pre, 
#     marker='*', c='indianred', label='predicted (test)', s=20)
plt.xlabel('Date', fontproperties=times, fontsize=20)  
plt.ylabel('Magnitude', fontproperties=times, fontsize=20)  
plt.tick_params(labelsize=18)
# plt.ylim(2.1, )
plt.ylim(4.9, 7.9)
plt.xticks(x_label1, x_label_tick1)
plt.xlim(min(x_label1)+5000, max(x_label1)-2000)
plt.legend(loc='lower center', prop=FontProperties(
# plt.legend(loc='lower center', prop=FontProperties(
    fname=r'C:\WINDOWS\Fonts\times.ttf',size=16), ncol=2) 
ax = plt.gca()
ax.yaxis.set_major_locator(plt.MultipleLocator(1))
plt.tight_layout()
# plt.savefig(figure_location+f'\\seism_{model_name}_minyear_{min_year}_maxyear_{max_year}_spanlat_{span_lat}_spanlon_{span_lon}_timewindow_{time_window}_nextmonth_{next_month}_minmag_{min_mag}_block_{config.loc_block}.pdf')
# plt.savefig(figure_location+f'\\seism_{model_name}_minyear_{min_year}_maxyear_{max_year}_spanlat_{span_lat}_spanlon_{span_lon}_timewindow_{time_window}_nextmonth_{next_month}_minmag_{min_mag}_split_ratio_{split_ratio}_blocks1.pdf')

plt.show()


fig = plt.figure(figsize=(6, 1))  
plt.text(0.02, 0.02, 
    'Evaluate on test data:\nMSE={0:.4f}, RMSE={1:.4f}'.format(
    test_loss, test_rmse), fontproperties=times, fontsize=14)
plt.ylim(0, 0.08)
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.tight_layout()
plt.savefig(figure_location+f'\\{file_name}-EvaluateTest.png')
plt.show() 


plt.figure(figsize=(10, 4))
plt.grid(True, linestyle='--')      
plt.scatter(jul_date[:len(y_train)], y_train,
    marker='o', c='grey', label='observed', s=10)
plt.scatter(jul_date[len(y_train):len(y_train)+len(y_test)], y_test,
    marker='o', c='grey', s=10)
plt.scatter(jul_date[:len(y_train)], y_train_pre,
    marker='+', c='dodgerblue', label='predicted-train', s=30)    
plt.scatter(jul_date[len(y_train):len(y_train)+len(y_test)], y_test_pre,
    marker='*', c='indianred', label='predicted-test', s=20)
plt.xlabel('Date', fontproperties=times, fontsize=20)  
plt.ylabel('Magnitude', fontproperties=times, fontsize=20)  
plt.tick_params(labelsize=18)
plt.ylim(2.1, 8.1)
plt.xticks(x_label1, x_label_tick1)
plt.xlim(min(x_label1)-1000, max(x_label1)+1000)
plt.legend(loc='upper center', prop=FontProperties(
    fname=r'C:\WINDOWS\Fonts\times.ttf',size=14),ncol=4) 
plt.tight_layout()
# plt.savefig(figure_location+f'\\seism_minyear_{min_year}_maxyear_{max_year}_spanlat_{span_lat}_spanlon_{span_lon}_timewindow_{time_window}_nextmonth_{next_month}_minmag_{min_mag}_block_{config.loc_block}.pdf')
plt.show()


fig = plt.figure(figsize=(8, 5)) 
plt.suptitle(r'Testing Set', fontproperties=times, fontsize=20, color='red')  
fig.add_subplot(1,2,1)
plt.grid(True, linestyle='--', linewidth=1.0)
plt.hist(diff_test.reshape(-1,), bins=11, range=(-2.25, 2.25), color='dodgerblue', stacked=True)
    # label=labels)
plt.xlabel('Predicted - Observed', fontproperties=times, fontsize=18)  
plt.ylabel('Frequency', fontproperties=times, fontsize=18)  
plt.title(r'Absolute Error', fontproperties=times, fontsize=20)
plt.legend(loc='upper left', fontsize=9) 
plt.tick_params(labelsize=16)
ax = plt.gca()
fig.add_subplot(1,2,2)
plt.grid(True, linestyle='--', linewidth=1.0)
plt.hist(diff_test.reshape(-1,)/y_test.reshape(-1,), bins=11, range=(-0.425, 0.425), color='dodgerblue', stacked=True)
    # label=labels)
plt.xlabel('(Predicted-Observed)/Observed', fontproperties=times, fontsize=18)  
plt.ylabel('Frequency', fontproperties=times, fontsize=18)  
plt.title(r'Relative Error', fontproperties=times, fontsize=20)
plt.legend(loc='upper left', fontsize=12) 
plt.tick_params(labelsize=16)
ax = plt.gca()
plt.tight_layout()
plt.subplots_adjust(wspace=0.2)       
plt.savefig(figure_location+f'\\{file_name}-Frequency-test.png')
plt.show()

# fig = plt.figure(figsize=(8, 5))   
# plt.suptitle(r'All Set', fontproperties=times, fontsize=20, color='red')
# fig.add_subplot(1,2,1)
# plt.grid(True, linestyle='--', linewidth=1.0)
# plt.hist(diff_all, bins=11, range=(-2.25, 2.25), color='dodgerblue', stacked=True)
#     # , label=labels)
# plt.xlabel('Predicted - Observed', fontproperties=times, fontsize=18)  
# plt.ylabel('Frequency', fontproperties=times, fontsize=18)  
# plt.title(r'Absolute Error', fontproperties=times, fontsize=20)
# plt.legend(loc='upper right', fontsize=12) 
# plt.xticks(size=14)
# plt.yticks(size=14)
# ax = plt.gca()
# fig.add_subplot(1,2,2)
# plt.grid(True, linestyle='--', linewidth=1.0)
# plt.hist(diff_all/y_all, bins=11, range=(-0.425, 0.425), color='dodgerblue', stacked=True)
#     #  label=labels)
# plt.xlabel('(Predicted - Observed) / observed', fontproperties=times, fontsize=18)  
# plt.ylabel('Frequency', fontproperties=times, fontsize=18)  
# plt.title(r'Relative Error', fontproperties=times, fontsize=20)
# plt.legend(loc='upper right', fontsize=9) 
# plt.xticks(size=14)
# plt.yticks(size=14)
# ax = plt.gca()
# plt.tight_layout()
# plt.subplots_adjust(wspace=0.2)   
# plt.savefig(r'.\figure\{}-Frequency.png'.format(file_name))
# plt.show()


fig = plt.figure(figsize=(6, 5))   
plt.grid(True, linestyle='--', linewidth=1.0)
plt.bar(x=x_label, height=np.array(num_list1), width=0.1, 
    color='indianred', label='M>=3')
plt.bar(x=x_label, height=np.array(num_list2), width=0.1, 
    color='dodgerblue', label='M>=6')
plt.bar(x=x_label, height=np.array(num_list3), width=0.1, 
    color='darkviolet', label='M>=7')
plt.xlabel('Predicted - Observed', fontproperties=times, fontsize=18)  
plt.ylabel('Frequency', fontproperties=times, fontsize=18)  
plt.legend(loc='upper left', prop=times, fontsize=15) 
plt.tick_params(labelsize=16)
plt.title(r'All Set', fontproperties=times, fontsize=20, color='red')
plt.tight_layout()
plt.savefig(r'.\figure\{}-Frequency367.png'.format(file_name))
plt.show()   

fig = plt.figure(figsize=(4, 2))  
plt.text(0.02, 0.06, '$M>=3: \mu={:.4f}$, $\sigma={:.4f}$'.format(
    np.mean(np.array(diff_all_blocks)[:, 0], axis=0), 
    np.std(np.array(diff_all_blocks)[:, 0], axis=0)), 
    fontproperties=times, fontsize=14)
plt.text(0.02, 0.04, '$M>=6: \mu={:.4f}$, $\sigma={:.4f}$'.format(
    np.mean(np.array(diff6)[:, 0], axis=0), np.std(np.array(diff6)[:, 0], axis=0)), 
    fontproperties=times, fontsize=14)
plt.text(0.02, 0.02, '$M>=7: \mu={:.4f}$, $\sigma={:.4f}$'.format(
    np.mean(np.array(diff7)[:, 0], axis=0), np.std(np.array(diff7)[:, 0], axis=0)), 
    fontproperties=times, fontsize=14)    
plt.ylim(0.01, 0.08)
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.title(r'All Set', fontproperties=times, fontsize=20, color='red')
plt.tight_layout()
plt.savefig(r'.\figure\{}-MuSigma.png'.format(file_name))
plt.show() 

fig = plt.figure(figsize=(6, 6))  
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
    plt.text(0.015, 1.8, '精确度 ACC = {:.2f}%'.format(100*ACC), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.7, '无震报准率 P0 = {:.2f}%'.format(100*P0), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.6, '有震报准率 P1 = {:.2f}%'.format(100*P1), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.5, '敏感度 Sn = {:.2f}%'.format(100*Sn), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.4, '特异度 Sp = {:.2f}%'.format(100*Sp), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.3, '平均值 Avg = {:.2f}%'.format(100*Avg), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.2, 'MCC = {:.2f}%'.format(100*MCC), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.1, '调和平均数 F1_Score = {:.2f}'.format(F1_Score), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.0, '综合评分 R_Score = {:.2f}'.format(R_Score), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.9, '{}'.format('-'*26), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.8, 'TP = {}'.format(TP), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.7, 'FN = {}'.format(FN), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.6, 'TP+FN = {}'.format(TP_FN), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.5, 'FP = {}'.format(FP), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.4, 'TN = {}'.format(TN), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.3, 'FP+TN = {}'.format(FP_TN), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.2, 'TP+FP = {}'.format(TP_FP), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.1, 'FN+TN = {}'.format(FN_TN), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.0, 'N = {}'.format(N), va='bottom', fontsize=14, fontproperties=times)
    plt.ylim(-0.05, 2.1)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
plt.suptitle(r'All Set', fontproperties=times, fontsize=20, color='red')
plt.tight_layout()
plt.savefig(r'.\figure\{}-ConfuseMatrix.png'.format(file_name))
plt.show()

fig = plt.figure(figsize=(6, 6))  
for i, threshold in enumerate([6, 7]):
    plt.subplot(1,2,i+1)  
    TP = np.sum(np.logical_and(y_test_pre>=threshold, y_test>=threshold)) # 报震有震
    FN = np.sum(np.logical_and(y_test_pre<threshold, y_test>=threshold)) # 报无有震
    TP_FN = TP + FN
    FP = np.sum(np.logical_and(y_test_pre>=threshold, y_test<threshold)) # 报震无震
    TN = np.sum(np.logical_and(y_test_pre<threshold, y_test<threshold)) # 报无无震
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
    plt.text(0.015, 1.8, '精确度 ACC = {:.2f}%'.format(100*ACC), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.7, '无震报准率 P0 = {:.2f}%'.format(100*P0), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.6, '有震报准率 P1 = {:.2f}%'.format(100*P1), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.5, '敏感度 Sn = {:.2f}%'.format(100*Sn), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.4, '特异度 Sp = {:.2f}%'.format(100*Sp), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.3, '平均值 Avg = {:.2f}%'.format(100*Avg), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.2, 'MCC = {:.2f}%'.format(100*MCC), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.1, '调和平均数 F1_Score = {:.2f}'.format(F1_Score), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 1.0, '综合评分 R_Score = {:.2f}'.format(R_Score), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.9, '{}'.format('-'*26), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.8, 'TP = {}'.format(TP), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.7, 'FN = {}'.format(FN), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.6, 'TP+FN = {}'.format(TP_FN), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.5, 'FP = {}'.format(FP), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.4, 'TN = {}'.format(TN), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.3, 'FP+TN = {}'.format(FP_TN), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.2, 'TP+FP = {}'.format(TP_FP), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.1, 'FN+TN = {}'.format(FN_TN), va='bottom', fontsize=14, fontproperties=times)
    plt.text(0.015, 0.0, 'N = {}'.format(N), va='bottom', fontsize=14, fontproperties=times)
    plt.ylim(-0.05, 2.1)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
plt.suptitle(r'Test Set', fontproperties=times, fontsize=20, color='red')
plt.tight_layout()
plt.savefig(r'.\figure\{}-ConfuseMatrix.png'.format(file_name))
plt.show()


fig = plt.figure(figsize=(6, 2))
total_M = diff_all_blocks.shape[0]*diff_all_blocks.shape[1]
accuracy_M = np.sum(np.absolute(diff_all_blocks)<=0.5)
ratio_M = np.sum(np.absolute(diff_all_blocks)<=0.5) / total_M
plt.text(0.02, 0.4, 'M>=3: 地震数量={}, 准确预测数量={}, 占比={:.2f}%'.format(
    total_M, accuracy_M, 100*ratio_M), va='bottom', fontsize=16, fontproperties=times)
for i in np.arange(6, 7+1, 1):
    total_M = np.sum(M_inital>=i)
    accuracy_M = np.sum(np.logical_and(
        np.absolute(diff_all_blocks.reshape(-1,))<=0.5, M_inital.reshape(-1,)>=i))
    ratio_M = np.sum(np.logical_and(np.absolute(
        diff_all_blocks.reshape(-1,))<=0.5, M_inital.reshape(-1,)>=i)) / total_M
    plt.text(0.02, 0.3-0.1*(i-6), 'M>={}: 地震数量={}, 准确预测数量={}, 占比={:.2f}%'.\
        format(i, total_M, accuracy_M, 100*ratio_M), 
        va='bottom', fontsize=16, fontproperties=times)
plt.ylim(0.15, 0.5)
plt.xlim(0, 1)
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.title(r'All Set', fontproperties=times, fontsize=20, color='red')
plt.tight_layout()
plt.savefig(r'.\figure\{}-Ratio367.png'.format(file_name))
plt.show()

fig = plt.figure(figsize=(5, 5))   
plt.grid(True, linestyle='--', linewidth=1.0)
l1 = plt.plot(y_all, y_all, c='indianred', linewidth=1.5)
l2 = plt.plot(y_all-0.5, y_all, c='grey', linewidth=1.5)
l3 = plt.plot(y_all+0.5, y_all, c='grey', linewidth=1.5)
plt.text(7.2, 7.2, 'y=x', fontproperties=times, fontsize=16)
if n_splits == 1:
    plt.scatter(y_train, y_train_pre, marker='o', c='none', edgecolors='dodgerblue', label='Training Set', s=10)    
# else:
#     plt.scatter(y, y_pre, marker='o', c='none', edgecolors='dodgerblue', label='Training and Validation Set', s=10)    
plt.scatter(y_test, y_test_pre, marker='*', c='indianred', label='Testing Set', s=20) 
plt.xlabel('Observed', fontproperties='Arial', fontsize=18)  
plt.ylabel('Predicted', fontproperties='Arial', fontsize=18)  
plt.tick_params(labelsize=16)
plt.xlim(2.9, 7.6)
plt.ylim(2.9, 7.6)
plt.legend(loc='best', prop=times, fontsize=15)
plt.title(r'All Set', fontproperties=times, fontsize=20, color='red')
plt.tight_layout() 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.set_aspect(aspect='equal')
plt.savefig(r'.\figure\{}-PredictedObserved.png'.format(file_name))
plt.show()

fig = plt.figure(figsize=(8, 5))   
plt.subplot(1,1,1) 
plt.grid(True, linestyle='--', linewidth=1)
for i, value in enumerate(diff_test.reshape(-1, 1)):
    # if i == fake[0]:
    #     plt.scatter(i, value, c='darkviolet', label='Fake Predict', marker='x', s=25)
    # elif i == miss[0]:
    #     plt.scatter(i, value, c='indianred', label='Missing Predict', marker='+', s=30)
    # elif i == acc[0]:
    #     plt.scatter(i, value, c='none', edgecolors='dodgerblue', 
    #         label='Accuracy Predict', marker='o', s=10)
    # else:
    #     if value > 0.5: # label='虚报'
    #         plt.scatter(i, value, c='darkviolet', marker='x', s=25)
    #     elif value < -0.5: # label='漏报'
    #         plt.scatter(i, value, c='indianred', marker='+', s=30)
    #     else: # label='准确预报'
    #         plt.scatter(i, value, c='none', edgecolors='dodgerblue', marker='o', s=10)
    if value > 0.5: # label='虚报'
        plt.scatter(i, value, c='darkviolet', marker='x', s=25)
    elif value < -0.5: # label='漏报'
        plt.scatter(i, value, c='indianred', marker='+', s=30)
    else: # label='准确预报'
        plt.scatter(i, value, c='none', edgecolors='dodgerblue', marker='o', s=10)
plt.legend(loc='best', prop=times, fontsize=15) 
plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)  
plt.ylabel('Predicted - Observed', fontproperties='Arial', fontsize=18)  
plt.tick_params(labelsize=16)
plt.ylim(-3.2, 3.2)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.title(r'Testing Set', fontproperties=times, fontsize=20, color='red')
plt.tight_layout()
plt.savefig(r'.\figure\{}-DiffPredictedObserved.png'.format(file_name))
plt.show()

fig = plt.figure(figsize=(3, 1.8))  
plt.text(0.02, 0.06, '虚报{}次，虚报率={:.2f}%'.format(len(fake_test), \
    100*len(fake_test)/len(diff_test.reshape(-1,))), fontsize=16, fontproperties=times)
plt.text(0.02, 0.04, '漏报{}次，漏报率={:.2f}%'.format(len(miss_test), \
    100*len(miss_test)/len(diff_test.reshape(-1,))), fontsize=16, fontproperties=times)
if len(fake_test)-len(miss_test) >= 0:
    plt.text(0.02, 0.02, '虚报比漏报多{}次'.format(len(fake_test)-len(miss_test)), fontsize=16, fontproperties=times)
else:
    plt.text(0.02, 0.02, '虚报比漏报少{}次'.format(len(miss_test)-len(fake_test)), fontsize=16, fontproperties=times)
plt.ylim(0.015, 0.07)
plt.title(r'Testing Set', fontproperties=times, fontsize=20, color='red')
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.tight_layout()
plt.savefig(r'.\figure\{}-FakeMiss.png'.format(file_name))
plt.show()

fig = plt.figure(figsize=(8, 5))   
plt.subplot(1,1,1) 
plt.grid(True, linestyle='--', linewidth=1)
for i, value in enumerate(diff_all.reshape(-1, 1)):
    # if i == fake_test[0]:
    #     plt.scatter(i, value, c='darkviolet', label='Fake Predict', marker='x', s=25)
    # elif i == miss_test[0]:
    #     plt.scatter(i, value, c='indianred', label='Missing Predict', marker='+', s=30)
    # elif i == acc_test[0]:
    #     plt.scatter(i, value, c='none', edgecolors='dodgerblue', 
    #         label='Accuracy Predict', marker='o', s=10)
    # else:
    #     if value > 0.5: # label='虚报'
    #         plt.scatter(i, value, c='darkviolet', marker='x', s=25)
    #     elif value < -0.5: # label='漏报'
    #         plt.scatter(i, value, c='indianred', marker='+', s=30)
    #     else: # label='准确预报'
    #         plt.scatter(i, value, c='none', edgecolors='dodgerblue', marker='o', s=10)
    if value > 0.5: # label='虚报'
        plt.scatter(i, value, c='darkviolet', marker='x', s=25)
    elif value < -0.5: # label='漏报'
        plt.scatter(i, value, c='indianred', marker='+', s=30)
    else: # label='准确预报'
        plt.scatter(i, value, c='none', edgecolors='dodgerblue', marker='o', s=10)
plt.legend(loc='best', prop=times, fontsize=15) 
plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)  
plt.ylabel('Predicted - Observed', fontproperties='Arial', fontsize=18)  
plt.title(r'All Set', fontproperties=times, fontsize=20, color='red')
plt.xticks(size=14)
plt.yticks(size=14)
plt.ylim(-3.2, 3.2)
# plt.axvline(x=(len(x_train)+len(x_test))*blocks, ls=':', c='green')
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig(r'.\figure\{}-DiffPredictedObserved.png'.format(file_name))
plt.show()

fig = plt.figure(figsize=(3, 1.8))  
plt.text(0.02, 0.06, '虚报{}次，虚报率={:.2f}%'.format(len(fake), \
    100*len(fake)/len(diff_all_blocks)), fontsize=16, fontproperties=times)
plt.text(0.02, 0.04, '漏报{}次，漏报率={:.2f}%'.format(len(miss), \
    100*len(miss)/len(diff_all_blocks)), fontsize=16, fontproperties=times)
if len(fake)-len(miss) >= 0:
    plt.text(0.02, 0.02, '虚报比漏报多{}次'.format(len(fake)-len(miss)), fontsize=16, fontproperties=times)
else:
    plt.text(0.02, 0.02, '虚报比漏报少{}次'.format(len(miss)-len(fake)), fontsize=16, fontproperties=times)
plt.ylim(0.015, 0.07)
plt.title(r'All Set', fontproperties=times, fontsize=20, color='red')
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.tight_layout()
plt.savefig(r'.\figure\{}-FakeMiss.png'.format(file_name))
plt.show()

fig = plt.figure(figsize=(9, 5))   
plt.title(r'Testing Set', fontproperties=times, fontsize=20, color='red')
plt.grid(True, linestyle='--', linewidth=1)
for key, value in enumerate(location_block):
    print(key, value)
    plt.plot(np.arange(len(y_test)),  y_test[:, key], linewidth=2,
        marker=markers[key], color=colors[key], 
        label='Observed Block{}'.format(label[key]))   
    plt.plot(np.arange(len(y_test_pre)), y_test_pre[:, key], linewidth=2,
        linestyle=':', marker=markers[key+6], color=colors[key], 
        label='Predicted Block{}'.format(label[key]))   
plt.xlabel('Date', fontproperties=times, fontsize=18)  
plt.ylabel('Magnitude', fontproperties=times, fontsize=18)  
plt.legend(loc='best', fontsize=12) 
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca() 
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
plt.tight_layout()
plt.savefig(r'.\figure\{}-Histgram-test.png'.format(file_name))
plt.show()

fig = plt.figure(figsize=(9, 5))   
plt.title(r'All Set', fontproperties=times, fontsize=20, color='red')
plt.grid(True, linestyle='--', linewidth=1)
for key, value in enumerate(location_block):
    plt.plot(np.arange(len(y_all)),  y_all[:, key], linewidth=2,
        marker=markers[key], color=colors[key], 
        label='Observed Block{}'.format(label[key]))   
    plt.plot(np.arange(len(y_all_pre)), y_all_pre[:, key], linewidth=2,
        linestyle=':', marker=markers[key+6], color=colors[key], 
        label='Predicted Block{}'.format(label[key]))   
plt.xlabel('Date', fontproperties=times, fontsize=18)  
plt.ylabel('Magnitude', fontproperties=times, fontsize=18)  
plt.legend(loc='upper left', fontsize=12) 
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca() 
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
ax.xaxis.set_major_locator(plt.MultipleLocator(50))
plt.tight_layout()
plt.savefig(r'.\figure\{}-Histgram.png'.format(file_name))
plt.show()


# fig = plt.figure(figsize=(8, blocks))  
# plt.suptitle(r'Testing Set', fontproperties=times, fontsize=20, color='red')
# fig.add_subplot(1,2,1)
# plt.title(r'绝对误差', fontproperties=times, fontsize=16)
# for i in np.arange(blocks):
#     plt.text(0.02, 1-(i+0.7)/blocks, '区块{}：$\mu={:.4f}$, $\sigma={:.4f}$'.format(i+1,
#         np.mean(diff_test, axis=0)[i], np.std(diff_test, axis=0)[i]), 
#         fontproperties=times, fontsize=14)
# ax = plt.gca()
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.add_subplot(1,2,2)
# plt.title(r'相对误差', fontproperties=times, fontsize=16)
# for i in np.arange(blocks):
#     plt.text(0.02, 1-(i+0.7)/blocks, '区块{}: $\mu={:.4f}$, $\sigma={:.4f}$'.format(i+1,
#         np.mean(diff_test/y_test, axis=0)[i], np.std(diff_test/y_test, axis=0)[i]), 
#         fontproperties=times, fontsize=14)
# ax = plt.gca()
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# plt.tight_layout()
# plt.subplots_adjust(wspace=0.2, right=.7) 
# plt.tight_layout()
# plt.savefig(r'.\figure\{}-AbsoluteError-test.png'.format(file_name))
# plt.show()

# fig = plt.figure(figsize=(8, blocks))  
# plt.suptitle(r'All Set', fontproperties=times, fontsize=20, color='red')
# fig.add_subplot(1,2,1)
# plt.title(r'绝对误差', fontproperties=times, fontsize=16)
# for i in np.arange(blocks):
#     plt.text(0.02, 1-(i+0.7)/blocks, '区块{}：$\mu={:.4f}$, $\sigma={:.4f}$'.format(i+1,
#         np.mean(diff_all, axis=0)[i], np.std(diff_all, axis=0)[i]), 
#         fontproperties=times, fontsize=14)
# ax = plt.gca()
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.add_subplot(1,2,2)
# plt.title(r'相对误差 all', fontproperties=times, fontsize=16)
# for i in np.arange(blocks):
#     plt.text(0.02, 1-(i+0.7)/blocks, '区块{}: $\mu={:.4f}$, $\sigma={:.4f}$'.format(i+1,
#         np.mean(diff_all/y_all, axis=0)[i], np.std(diff_all/y_all, axis=0)[i]), 
#         fontproperties=times, fontsize=14)
# ax = plt.gca()
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# plt.tight_layout()
# plt.subplots_adjust(wspace=0.2, right=.7) 
# plt.tight_layout()
# plt.savefig(r'.\figure\{}-AbsoluteError.png'.format(file_name))
# plt.show()


# fig = plt.figure(figsize=(10, 8))  
# plt.subplot(1,2,1) 
# plt.grid(True, linestyle='--', linewidth=1)
# plt.axvspan(xmin=6, xmax=9.1, alpha=0.2, facecolor='green')
# plt.axhspan(ymin=6, ymax=9.1, alpha=0.2, facecolor='yellow')
# plt.plot(y_all, y_all, c='indianred', linewidth=1.5)
# plt.plot(y_all, y_all+0.5, c='grey', linewidth=1.5)
# plt.plot(y_all, y_all-0.5, c='grey', linewidth=1.5)
# plt.text(7.8, 8.1, 'y=x', fontproperties=times, fontsize=16)
# if n_splits == 1:
#     plt.scatter(y_train, y_train_pre, marker='o', c='none', edgecolors='dodgerblue', label='training data', s=10)    
# else:
#     plt.scatter(y, y_pre, marker='o', c='none', edgecolors='dodgerblue', label='training and validation data', s=10)    
# plt.scatter(y_test, y_test_pre, marker='*', c='indianred', label='testing data', s=20) 
# plt.xlabel('observed', fontproperties=times, fontsize=18)  
# plt.ylabel('predicted', fontproperties=times, fontsize=18)  
# plt.xticks(size=14)
# plt.yticks(size=14)
# plt.xlim(2.6, 9.1)
# plt.ylim(2.6, 9.1)
# plt.legend(loc='lower left', prop=times, fontsize=15) 
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
# plt.text(7.8, 8.1, 'y=x', fontproperties=times, fontsize=16)
# if n_splits == 1:
#     plt.scatter(y_train, y_train_pre, marker='o', c='none', edgecolors='dodgerblue', label='training data', s=10)    
# else:
#     plt.scatter(y, y_pre, marker='o', c='none', edgecolors='dodgerblue', label='training and validation data', s=10)    
# plt.scatter(y_test, y_test_pre, marker='*', c='indianred', label='testing data', s=20) 
# plt.xlabel('observed', fontproperties=times, fontsize=18)  
# plt.ylabel('predicted', fontproperties=times, fontsize=18)  
# plt.xticks(size=14)
# plt.yticks(size=14)
# plt.xlim(2.6, 9.1)
# plt.ylim(2.6, 9.1)
# plt.legend(loc='lower left', prop=times, fontsize=15) 
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# ax.set_aspect(aspect='equal')
# plt.tight_layout()
# plt.savefig(r'.\figure\{}-PredictedObserved0.png'.format(file_name))
# plt.show()