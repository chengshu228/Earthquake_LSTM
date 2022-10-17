import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
import tensorflow as tf
tf.keras.backend.clear_session()  
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

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

plt.rcParams['axes.unicode_minus']=False 

import ANN
import utils 
import config
file_name = config.file_name
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

date = np.genfromtxt(data_location+f'\\date-{file_name}-blocks1.txt', delimiter=' ')
# date = np.genfromtxt(data_location+f'\\date-{file_name}.txt', delimiter=' ')
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

start_time = time.time()
seed = config.seed
np.random.seed(seed)
tf.random.set_seed(seed)



seed = config.seed
tf.random.set_seed(seed)
np.random.seed(seed)
n_out = config.n_out
epochs = config.epochs
timesteps = config.timesteps


MSE, RMSE = [], []
# for model_name in ['svr', 'lr']:
for model_name in ['svr', 'lr', 'rf', 'gbr','dt', 'kn', 'etr']:
	print(f'{model_name}')
    blocks = 1 
    factor = np.genfromtxt(data_location+f'\\factor-{file_name}-blocks1.txt', delimiter=' ')
    # factor = np.genfromtxt(data_location+f'\\factor-{file_name}.txt', delimiter=' ')
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
    output_data0 = output_data
    # print('input=', input_data.shape, 'output=', output_data.shape)

    if loc_block <= 10: n_out = 1
    else: n_out = loc_block//10
    # print('\tn_out={}'.format(n_out))
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
    # print('\tshape: ', x_train.shape, y_train.shape,
    #     x_test.shape, y_test.shape)

    if model_name=='svr': 
        model = MultiOutputRegressor(LinearSVR())
        model = LinearSVR(C=1,random_state=0)
    elif model_name=='lr': 
        model = LinearRegression(copy_X=True,fit_intercept=True,normalize=True)
    elif model_name=='rf': 
        model = RandomForestRegressor(n_estimators=2,random_state=0)
    elif model_name=='dt': 
        model = DecisionTreeRegressor(max_depth=1000)  
    elif model_name=='kn': 
        model = KNeighborsRegressor(n_neighbors=1)  
    elif model_name=='gbr': 
        model = GradientBoostingRegressor()
    elif model_name=='etr': 
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(n_estimators=1)

    model.fit(x_train, y_train)

    y_train_pre = model.predict(x_train).reshape(-1, n_out)
    y_test_pre = model.predict(x_test).reshape(-1, n_out)

    rss_train=((y_train-y_train_pre)**2).sum()
    mse_train = np.mean((y_train-y_train_pre)**2)
    rmse_train = np.sqrt(mse_train)    
    rss_test=((y_test-y_test_pre)**2).sum()
    mse_test = np.mean((y_test-y_test_pre)**2)
    rmse_test = np.sqrt(mse_test)
    print(f'n_in={config.n_in} & n_out={n_out}, {mse_train:.4f} & {rmse_train:.4f} & {mse_test:.4f} & {rmse_test:.4f}')

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

    error1 = tf.keras.metrics.mean_absolute_error(
        y_train, y_train_pre).numpy()
    error2 = tf.keras.metrics.mean_absolute_error(
        y_test, y_test_pre).numpy()

    plt.figure(figsize=(8, 4))
    plt.grid(True, linestyle='--')      
    # plt.plot(jul_date[:len(y_train)+time_window], y_train, label='observed', 
    # plt.plot(jul_date[:len(y_train)], y_train, 
    #     marker='o', markersize=4, 
    #     color="grey", markerfacecolor='white')
    # plt.plot(jul_date[len(y_train):len(y_train)+len(y_test)], y_test, 
    #     marker='o', markersize=4, 
    #     color="grey", markerfacecolor='white')
    plt.plot(jul_date[-len(output_data0):], output_data0, label='observed', 
        marker='o', markersize=4, 
        color="grey", markerfacecolor='white')
    plt.plot(jul_date[:len(y_train)], y_train_pre, label='predicted-train', 
        marker='s', markersize=4, 
        color="dodgerblue", markerfacecolor='white') 
    plt.plot(jul_date[len(y_train):len(y_train)+len(y_test)], y_test_pre, 
        # marker='*', c='indianred', label='predicted-test', s=20)
        marker='*', markersize=4, label='predicted-test',
        color="indianred", markerfacecolor='white') 
    plt.fill_between(jul_date[:len(y_train)], 
        y_train_pre-error1, y_train_pre+error1, 
        alpha=0.5, color='orange',label='confidence interval') 
    plt.fill_between(jul_date[len(y_train):len(y_train)+len(y_test)],
        y_test_pre-error2, y_test_pre+error2, 
        alpha=0.5, color='orange') 
    # plt.errorbar(jul_date[:len(y_train)], 
    #     y_train_pre, yerr=error1,fmt='k')
    # plt.errorbar(jul_date[len(y_train):len(y_train)+len(y_test)], 
    #     y_test_pre, yerr=error2,fmt='k')
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
    plt.ylim(2.1, 6.5)
    plt.xticks(x_label1, x_label_tick1)
    plt.xlim(min(x_label1)+5000, max(x_label1)-2000)
    plt.ylim(2.1, )
    plt.ylim(4.9, 7.9)
    # plt.legend(loc='upper center', prop=FontProperties(fname=r'C:\WINDOWS\Fonts\times.ttf',size=18),ncol=2) 
    plt.legend(loc='lower center', prop=FontProperties(fname=r'C:\WINDOWS\Fonts\times.ttf',size=16),ncol=2) 
    ax = plt.gca()
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(figure_location+f'\\seism_{model_name}_minyear_{min_year}_maxyear_{max_year}_spanlat_{span_lat}_spanlon_{span_lon}_timewindow_{time_window}_nextmonth_{next_month}_minmag_{min_mag}_split_ratio_{split_ratio}_blocks{blocks}.pdf')
    # plt.savefig(figure_location+f'\\seism_{model_name}_minyear_{min_year}_maxyear_{max_year}_spanlat_{span_lat}_spanlon_{span_lon}_timewindow_{time_window}_nextmonth_{next_month}_minmag_{min_mag}_block_{config.loc_block}.pdf')

    # plt.savefig(file_location+r'.\figure\{}-PredictM.png'.format(file_name))
    # plt.show()

    # picture_dir = os.path.join(file_location, 'pictures')
    # if not os.path.exists(picture_dir): os.mkdir(picture_dir)

    # plt.figure(figsize=(4, 4))
    # plt.scatter(y_train.reshape(-1, 1), y_train_pre.reshape(-1, 1),
    #     label='train data', marker='o', c='none', edgecolors='dodgerblue', s=20)
    # plt.scatter(y_val.reshape(-1, 1), y_val_pre.reshape(-1, 1),
    #     label='test data', marker='s', c='none', edgecolors='indianred', s=20)
    # plt.legend(loc='upper left', shadow=False, fontsize=16)
    # plt.xlim(2.4, 7.1)
    # plt.ylim(2.4, 7.1)
    # plt.xlabel('Actual (m$^3$/s)', fontproperties=times, fontsize=20)
    # plt.ylabel('Predicted (m$^3$/s)', fontproperties=times, fontsize=20)
    # plt.tick_params(labelsize=16)
    # plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    # plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    # plt.tight_layout()
    # plt.savefig(picture_dir+r'\spring_{}_fit_in_{}_out_{}.pdf'.format(model_name, n_in, n_out))
    # plt.savefig(picture_dir+r'\spring_{}_fit_in_{}_out_{}.eps'.format(model_name, n_in, n_out))
    # plt.show()
print('\n')
print('It takes {:.2f} minutes.'.format((time.time()-t)/60))


# diff = (y_pre - y_all)
# y_future = model.predict(x_all[-1,:].reshape(-1, 10*n_in))
# y_future = y_scaler.inverse_transform(y_future)

# print('R_squared={:.4f}/{:.4f}'.format(model.score(x_val, y_val), r2_score(y_val, y_val_pre)))

# print('train')
# print('\tMSE={:.4f}'.format(mean_squared_error(
# 	y_scaler.inverse_transform(y_train.reshape(-1, 1)),
#     y_scaler.inverse_transform(y_train_pre.reshape(-1, 1)))))
# print('\tMAE={:.4f}'.format(mean_absolute_error(
# 	y_scaler.inverse_transform(y_train.reshape(-1, 1)),
#     y_scaler.inverse_transform(y_train_pre.reshape(-1, 1)))))
# print('\tRMSE={:.4f}'.format(np.sqrt(mean_squared_error(
# 	y_scaler.inverse_transform(y_train.reshape(-1, 1)),
#     y_scaler.inverse_transform(y_train_pre.reshape(-1, 1))))))
# print('val')
# print('\tMSE={:.4f}'.format(mean_squared_error(
# 	y_scaler.inverse_transform(y_val.reshape(-1, 1)),
#     y_scaler.inverse_transform(y_val_pre.reshape(-1, 1)))))
# print('\tMAE={:.4f}'.format(mean_absolute_error(
# 	y_scaler.inverse_transform(y_val.reshape(-1, 1)),
#     y_scaler.inverse_transform(y_val_pre.reshape(-1, 1)))))
# print('\tRMSE={:.4f}'.format(np.sqrt(mean_squared_error(
# 	y_scaler.inverse_transform(y_val.reshape(-1, 1)),
#     y_scaler.inverse_transform(y_val_pre.reshape(-1, 1))))))


# reg = LinearRegression().fit(y_pre.reshape(-1, 1), y_all.reshape(-1, 1))
# plt.plot(y_pre.reshape(-1, 1), reg.predict(y_pre.reshape(-1, 1)), 
# 	color='darkviolet', label='predict')
# plt.text(2.5, 6.7, 'y={1:.4f}+{0:.4f}x'.format(
# 	reg.coef_[0, 0], reg.intercept_[0]), fontsize=16)
# plt.text(2.5, 6.3, 'R$^2$ = {:.4f}'.format(
# 	r2_score(y_pre.reshape(-1, 1), y_all.reshape(-1, 1))), fontsize=16)

# plt.figure(figsize=(12, 4))
# plt.plot(np.arange(1, total_month+1, 1), dataset[:,0], label='1987.1-2018.12观测值',
# 	marker='o', markersize=4, color='k', markerfacecolor='none', alpha=0.75)
# for i in np.arange(1, n_out+1, 1):
# 	plt.plot(np.arange(n_out+i, total_month-n_in+i+1, 1), y_pre[:,i-1], label='1987.1-2018.12',
# 		marker='o', markersize=4, color='y', markerfacecolor='none', alpha=0.75)
# plt.plot(np.arange(total_month+1, total_month+n_out+1, 1), y_future.reshape(-1,), label='2019.1-2019.6预测值',
# 	marker='o', markersize=4, color='indianred', markerfacecolor='none', alpha=0.75)
# plt.axvline(26*12, color='grey')
# # plt.text(1.0, 6.6, '(a)', fontsize=18)
# plt.xlim(-2, 32*12+n_out+2)
# plt.ylim(1.9, 7.1)
# plt.xlabel('日期(月)', fontproperties=font, fontsize=20)
# plt.ylabel('泉流量(m$^3$/s)', fontproperties=font, fontsize=20)
# plt.tick_params(labelsize=16)
# plt.xticks(np.arange(0, 32*12+n_out+1, 12), 
# 	('1986.12','1987.12','1988.12','1989.12','1990.12',
# 	'1991.12','1992.12','1993.12','1994.12','1995.12',
# 	'1996.12','1997.12','1998.12','1999.12', '2000.12',
# 	'2001.12','2002.12','2003.12','2004.12','2005.12',
# 	'2006.12','2007.12','2008.12','2009.12','2010.12',
# 	'2011.12','2012.12','2013.12','2014.12','2015.12',
# 	'2016.12','2017.12','2018.12'), rotation=45)
# plt.gca().xaxis.set_major_locator(MultipleLocator(12))
# plt.gca().yaxis.set_major_locator(MultipleLocator(1))
# plt.legend(loc='upper right', prop={'size':16, 'family':'STXINWEI'}, shadow=False, ncol=2)
# plt.tight_layout()
# plt.savefig(picture_dir+r'\spring.pdf')
# plt.savefig(picture_dir+r'\spring.eps')
# plt.show()

# cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=seed)
# n_scores = cross_val_score(model, x_train, y_train, cv=cv, n_jobs=-1,
#     scoring='neg_mean_squared_error', error_score='raise')
# mse = np.absolute(n_scores)
# rmse = np.sqrt(mse)
# print('MSE = %.4f (%.4f)' % (np.mean(mse), np.std(mse)))
# print('RMSE = %.4f (%.4f)' % (np.mean(rmse), np.std(rmse)))
