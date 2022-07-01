import config
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import config
import matplotlib
import datetime
import julian
from dateutil.relativedelta import relativedelta
from matplotlib.font_manager import FontProperties

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
data_location = config.data_location
file_location = config.file_location
file_name = config.file_name 
times = config.times
min_year = config.min_year
max_year = config.max_year 

matplotlib.rcParams.update(config.config_font)
plt.rcParams['font.sans-serif']=['simsun'] 
plt.rcParams['axes.unicode_minus']=False 

start_time = time.time()

initial_julian = julian.to_jd(datetime.datetime(1930,1,1,0,0,0), fmt='jd')
date = np.genfromtxt(data_location+\
    f'\\date-{file_name}.txt', delimiter=' ')
print(len(date))
year = np.array(date[:, 0], dtype=np.int)
month = np.array(date[:, 1], dtype=np.int)
day = np.array(date[:, 2], dtype=np.int)
date = pd.DataFrame({'year':year, 'month':month, 'day':day})
date = pd.to_datetime(date, format='%Y%m%d', errors='ignore')

jul_date = []
for i in np.arange(time_window,len(date)-next_month+1,1): 
    t = datetime.datetime(year[i],month[i],1,0,0,0)
    t_jd = julian.to_jd(t,fmt='jd')-initial_julian
    jul_date.append(t_jd)

x_label1, x_label_tick1 = [], []
for tick in np.arange(1930, 2030+1, config.step): 
    t = datetime.datetime(tick,1,1,0,0,0)
    t_jd = julian.to_jd(t, fmt='jd')-initial_julian
    x_label1.append(t_jd)
    x_label_tick1.append(tick)



factor = np.genfromtxt(data_location+\
    f'\\factor-{file_name}.txt', delimiter=' ')
print('\n  initial factor shape: ', factor.shape)
output_data = factor[:, 0]
print('b_mle: ', max(factor[:, 5].reshape(-1, 1)))
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
factor = factor.reshape(-1, blocks*(features*n+m))
# factor = factor[:, 1:].reshape(-1, blocks*features)
print('\n  factor', factor.shape)


location_block = [0, 2, blocks-3, blocks-1]
# location_block = [0]
linestyles = [':', '-', '--']
labels = []
for i in location_block:
    labels.append('block{}'.format(i+1))
colors = ['blueviolet', 'green', 'blue', 'goldenrod', 'cyan']
markers = ['p', 'd', 'v', '^', 'x', 'o', '+', '<', '>', 's', '*', 'P']
title =['frequency', 'max_magnitude', 'mean_magnitude', \
        'b_lstsq', 'b_mle', 'a_lstsq', \
        'max_mag_absence', 'rmse_lstsq', 'total_energy_square', \
        'mean_lon', 'rmse_lon', 'mean_lat', 'rmse_lat', \
        'k', 'epicenter_longitude', 'epicenter_latitude']

# fig = plt.figure(figsize=(5, 5))
# plt.grid(True, linestyle='--')
# plt.scatter(factor[:, 3], factor[:, 4], c='none', edgecolor='dodgerblue')
# plt.xlabel('$b_{lstsq}$', fontproperties=times, fontsize=18)
# plt.ylabel('$b_{mle}$', fontproperties=times, fontsize=18)
# # plt.title('b Value', fontsize=20, color='red')
# plt.tick_params(labelsize=16)
# # plt.xlim(-0.1, 2.6)
# ax = plt.gca()
# ax.set_aspect(aspect='equal')
# ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
# ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
# plt.tight_layout()
# plt.savefig(file_location+f'\\figure\seism_b_lstsq_mle_{min_year}_{max_year}.pdf')
# plt.show()
print(len(jul_date[:]), output_data.reshape(-1,6).shape, len(output_data.reshape(-1,6)[:,0]))
fig = plt.figure(figsize=(10, 5))
plt.grid(True, linestyle='--')
for i, color in enumerate(\
    ['blueviolet', 'green', 'blue', 'goldenrod', 'cyan','dodgerblue',]):
    plt.scatter(jul_date[:], output_data.reshape(-1,6)[:,i], c='none', edgecolor=color, label=f'block {i+1}')
# markerline, stemlines, baseline = plt.stem(np.arange(len(output_data)), output_data, linefmt='-',markerfmt='o',basefmt='none')
# 可单独设置棉棒末端，棉棒连线以及基线的属性
# plt.setp(markerline, color='k')#将棉棒末端设置为黑色
plt.xlabel('Date', fontproperties=times, fontsize=18)
plt.ylabel('Maximum Magnitude in Next Year', fontproperties=times, fontsize=18)
# plt.title('M-t', fontsize=20, color='red')
plt.tick_params(labelsize=16)
plt.xticks(x_label1, x_label_tick1)
# plt.xlim(min(x_label1)-1000, max(x_label1)+1000)
plt.ylim(2.8, 8.5)
plt.legend(loc='upper center', prop=FontProperties(
    fname=r'C:\WINDOWS\Fonts\times.ttf',size=14),ncol=3) 
plt.tight_layout()
plt.savefig(file_location+f'\\figure\seism_magnitude_index_{min_year}_{max_year}.pdf')
plt.show()

for earthquake_index in np.arange(0, features, 1):
    fig = plt.figure(figsize=(14, 6))
    fig.add_subplot(1,1,1)
    plt.title(f'{title[earthquake_index]}', fontdict=times, fontsize=20, color='red')
    plt.grid(True, linestyle='--', linewidth=1)
    for key, value in enumerate(location_block):
        plt.plot(np.arange(len(factor)), 
            factor[:, earthquake_index+(features*value)], 
            linewidth=2, marker=markers[key], label=labels[key], color=colors[key])   
        plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)  
        plt.ylabel(f'{title[earthquake_index]}', fontdict=times, fontsize=18)  
        plt.legend(loc='upper left', fontsize=12) 
        plt.xticks(size=14)
        plt.yticks(size=14)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.)
        ax.spines['left'].set_linewidth(1.)
        ax.spines['top'].set_linewidth(1.)
        ax.spines['right'].set_linewidth(1.)   
        ax.xaxis.set_major_locator(plt.MultipleLocator(blocks*6))
        plt.tight_layout()
    # plt.savefig(r".\figure\{}-Histgram-test.png".format(filename))
    plt.show()

