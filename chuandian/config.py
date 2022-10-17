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

file_location = os.getcwd()

seed = 12345

min_year, max_year = 1970, 2021
min_latitude, max_latitude = 21, 33  # 24 21   # 32 33
min_longitude, max_longitude = 98, 106
span_lat, span_lon = 6, 6  # 4 5 6
min_magnitude = 3 # 1.7 3
min_number = 30  # 30 50
time_window, next_month = 12, 12  # 12 18 24 30 36    # 3 6 9 12
each_move = 1

blocks = int((max_latitude-min_latitude-span_lat+1)*\
    (max_longitude-min_longitude-span_lon+1))

features = 16 

m, n = 0, 1

index = 1 
energy = False  # True False

split_ratio = 0.6
n_splits = 10 # 20

layer = 5  # 3 4 
layer_size = 256  # 128 256
rate = 0.5  
weight = 0  
batch_size = 16
epochs = 1*10**3 # 100
learning_rate = 1e-3

catolog_name = f'span_lat{span_lat:.0f}-span_lon{span_lon:.0f}-' +\
    f'time_window{time_window:.0f}-next_month{next_month:.0f}'

filename = f'span_lat{span_lat:.0f}-span_lon{span_lon:.0f}-' +\
    f'time_window{time_window:.0f}-next_month{next_month:.0f}-' +\
    f'index{index:.0f}-energy{energy:.0f}-' +\
    f'split_ratio{split_ratio:.2f}-n_splits{n_splits:.0f}-' +\
    f'layer{layer:.0f}-layer_size{layer_size:.0f}'

learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    decay_steps=epochs, initial_learning_rate=learning_rate, 
    decay_rate=0.99, staircase=False)

fontcn = {'family':'YouYuan', 'size':15} 
fonten = {'family':'Arial', 'size':15}
