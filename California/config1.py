import math
import numpy as np
import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
assert tf.__version__.startswith("2.") 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
from tensorflow.keras import layers, initializers, regularizers, optimizers
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import time
import datetime 
from dateutil.relativedelta import relativedelta
import h5py
import random

type_location = r'\California'
file_location = r'C:\Users\cshu\Desktop\shi'+type_location
seed = 12345

min_year, max_year = 1932, 2021 # 1932, 2021
min_latitude, max_latitude = 35, 37    # 5
min_longitude, max_longitude = -122, -191   # 8
span_lat, span_lon = 2, 3
each_move = 1
blocks = int((max_latitude-min_latitude-span_lat+1/each_move)*\
    (max_longitude-min_longitude-span_lon+1/each_move))-1
print(blocks)

min_magnitude = 3.  # 2 2.1 2.2 3 3.1 3.2 3.3
min_number = 30
time_window, next_month = 12, 12  # 12 18    # 8 9 12

features = 16 

split_ratio = 0.90
n_splits = 30 # 20

layer = 4  # 3 4 
layer_size = 256*2  # 128 256 
batch_size = 10*10
epochs = 1*10**2 # 100
learning_rate = 1e-4
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    decay_steps=epochs, initial_learning_rate=learning_rate, 
    decay_rate=0.99, staircase=False)
rate = 0.5  
weight = 0 

catolog_name = f'span_lat{span_lat:.0f}-span_lon{span_lon:.0f}-' +\
    f'time_window{time_window:.0f}-next_month{next_month:.0f}'

m, n = 0, 1
index = 1 
energy = False  # True False

filename = f'span_lat{span_lat:.0f}-span_lon{span_lon:.0f}-' +\
    f'time_window{time_window:.0f}-next_month{next_month:.0f}-' +\
    f'index{index:.0f}-energy{energy:.0f}-' +\
    f'split_ratio{split_ratio:.2f}-n_splits{n_splits:.0f}-' +\
    f'layer{layer:.0f}-layer_size{layer_size:.0f}'





fontcn = {'family':'YouYuan', 'size':15} 
fonten = {'family':'Arial', 'size':15}
