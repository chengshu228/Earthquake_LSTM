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
from matplotlib.font_manager import FontProperties

file_location = os.getcwd()
catalog_location = file_location+r'\catalog'
data_location = file_location+r'\data'
figure_location = file_location+r'\figure'

# model_name = input('model_name=')
model_name = 'lstm'

# min_year = int(input('min_year=1932/1968/1980 '))
min_year = 1932

max_year = 2021
min_mag = -2
# if min_year==1932: min_mag=3.0
# elif min_year==1968: min_mag=2.7
# elif min_year==1980: min_mag=2.2
min_lat, max_lat = 32, 37   # 5
min_lon, max_lon = 114, 122 # 8
span_lat, span_lon = 2, 4

each_move = 1
# blocks = int((max_lat-min_lat-span_lat+1/each_move)*\
#     (max_lon-min_lon-span_lon+1/each_move))-1
blocks = 6
# loc_block = 66 # 1, 2, 3, 4, 5, 6, 55, 66

loc_block = 1
# loc_block = int(input('loc_block=1/2/3/4/5/6/55/66 '))

min_number = 30
# time_window, next_month = 72, 12  # 12 18    # 8 9 12
# time_window, next_month = 72, 12  # 12 18    # 8 9 12
time_window, next_month = 3, 3  # 12 18    # 8 9 12
index,energy = 1,False  # True False

features = 16 
timesteps = 1
step = 10
n_out = 1
split_ratio = 0.90
n_splits = 1
rate = 0.5
weight_decay = 1e-4

layer = 2
# layer = int(input('layer=2/3 '))
lockback_samples = 4

layer_size = 10 

batch_size = 12
epochs = 1000
learning_rate = 1e-3
# learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
#     decay_steps=epochs/10, initial_learning_rate=learning_rate, 
#     decay_rate=1-1e-6, staircase=False)

simsun = FontProperties(fname=r'C:\WINDOWS\Fonts\simsun.ttc')
times = FontProperties(fname=r'C:\WINDOWS\Fonts\times.ttf')
config_font = {#'font.family':'serif', 'font.serif':['SimSun'], 
    'mathtext.fontset':'stix', 'font.size':16}

file_name = f'min_year{min_year:.0f}-max_year{max_year:.0f}-'+\
    f'span_lat{span_lat:.0f}-span_lon{span_lon:.0f}-'+\
    f'time_window{time_window:.0f}-next_month{next_month:.0f}-' +\
    f'min_mag{min_mag:.0f}'
    # f'index{index:.0f}-energy{energy:.0f}-' +\
    # f'split_ratio{split_ratio:.2f}-n_splits{n_splits:.0f}-' +\
    # f'layer{layer:.0f}-layer_size{layer_size:.0f}'

m = 0
n = 1
seed = 12345 
