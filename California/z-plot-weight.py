
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
# assert tf.__version__.startswith("2.") 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, TimeSeriesSplit

import config
from ANN import stateless_lstm, stateful_lstm, stateless_lstm_more
from utils import series_to_supervised, \
    checkpoints, save_data, dataset, read_data

start_time = time.time()
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

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

weights = np.loadtxt("./weight/w.txt", delimiter=" ")

l = 0.91
b = 0.2
w = 0.02
h = 1 - 2*b 

fig = plt.figure(figsize=(10, 8))
h1 = plt.imshow(np.abs(weights), cmap='bwr', vmin=0, vmax=0.16) 
plt.ylim(0-1, weights.shape[0]+1)
# plt.xlim(0, weights.shape[1])
plt.xticks(size=16)
plt.yticks(size=16)
ax = plt.gca()
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_major_locator(plt.MultipleLocator(features+m))
ax.set_aspect(aspect='equal')
cbar = fig.add_axes([l,b,w,h]) 
cbar = plt.colorbar(h1, cax=cbar)
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

if m==1:
    feature_name = ['the max. Magnitude in the next year',
        'frequency', 'max magnitude', 'mean magnitude', \
        'b_lstsq', 'b_mle', 'a_lstsq', \
        'max_mag_absence', 'rmse_lstsq', 'total_energy_square', \
        'mean_lon', 'rmse_lon', 'mean_lat', 'rmse_lat', \
        'k', 'epicenter_longitude', 'epicenter_latitude']
else:
    feature_name = ['frequency', 'max magnitude', 'mean magnitude', \
        'b_lstsq', 'b_mle', 'a_lstsq', \
        'max_mag_absence', 'rmse_lstsq', 'total_energy_square', \
        'mean_lon', 'rmse_lon', 'mean_lat', 'rmse_lat', \
        'k', 'epicenter_longitude', 'epicenter_latitude']
for j in np.arange(features+m): 
# for j in np.arange(2): 
    fig = plt.figure(figsize=(15, 5))
    for i in np.arange(len(weights)): 
        if i%(features+m) == j:
            plt.subplot(3, math.ceil(blocks/3), int(i/(features+m))+1) 
            h1 = plt.imshow(np.abs(weights[i, :]).reshape(2**4, -1), cmap='bwr', vmin=0, vmax=0.25) 
            ax = plt.gca()
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            print('{0} 第{1}个属性 权重的最大值={2:.4f}'.format(feature_name[j], i+1, np.max(np.abs(weights[i, :]))))
            plt.title('feature '+str(i+1), fontsize=14)
            l = 0.91            
            b = 0.2
            w = 0.015
            h = 1 - 2*b 
            cbar = fig.add_axes([l,b,w,h]) 
            cbar = plt.colorbar(h1, cax=cbar)
        plt.suptitle(str(feature_name[j]), fontsize=20, color='r')
        
    plt.show()
