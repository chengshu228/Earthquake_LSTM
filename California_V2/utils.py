
import config
import colorsys
import numpy as np
import pandas as pd
import h5py
import random
import datetime
import tensorflow as tf
import os

file_location = config.file_location
# catolog_name = config.catolog_name

# def dataset(blocks):	
# 	indicators = pd.read_csv(
# 		file_location+r'\code\data\dataset-'+catolog_name+r'.txt', 
# 		delimiter=' ', header=None, dtype=np.float32).values.astype('float64')
# 	features = indicators.shape[1]
# 	print('initial dataset.shape=', indicators.shape)
# 	reframed = series_to_supervised(indicators, n_in=blocks, n_out=blocks)
# 	value = reframed.values
# 	value = value[[i for i in np.arange(0, len(value), blocks)], :]
# 	print('value.shape=', value.shape)
# 	return value

def read_data(file_location, name):
    data = h5py.File(file_location + r'\{}.h5'.format(name),'r')
    data = data['elem'][:].astype('float32')
    # train_db = tf.data.Dataset.from_tensor_slices(data
    #             ).shuffle(self.batch_size*4).batch(self.batch_size)
    return data

def save_data(file_location, name, value):
    with h5py.File(file_location + r'\{}.h5'.format(name),'w') as hf:
        hf.create_dataset('elem', data=value, 
            compression='gzip', compression_opts=9)
        hf.close()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  '''
  函数用途：将时间序列转化为监督学习数据集。
  参数说明：
    data: 观察值序列，数据类型可以是 list 或者 NumPy array。
    n_in: 作为输入值(X)的滞后组的数量。
    n_out: 作为输出值(y)的观察组的数量。
    dropnan: Boolean 值，确定是否将包含 NaN 的行移除。
  返回值:
    经过转换的用于监督学习的 Pandas DataFrame 序列。
  '''
  n_vars = 1 if type(data) is list else data.shape[1]
  df = pd.DataFrame(data)
  cols, names = list(), list()
  # 输入序列 (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
  # 预测序列 (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # 将所有列拼合
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop 掉包含 NaN 的行
  if dropnan:
    agg.dropna(inplace=True)
  return agg

def checkpoints(model_name,name):
	model_dir = os.path.join(model_name)
	if not os.path.exists(model_dir): os.mkdir(model_dir)
	model_dir = os.path.join(model_dir, name)
	if not os.path.exists(model_dir): os.mkdir(model_dir)
	filepath = os.path.join(model_dir, 
		'{epoch:04d}-{rmse:.4f}-{val_rmse:.4f}.h5')
		# '{epoch:04d}-{loss:.4f}-{rmse:.4f}-{val_loss:.4f}-{val_rmse:.4f}.h5')
		# '{epoch:04d}-{loss:.6f}-{rmse:.6f}-{mape:.6f}-{val_loss:.6f}-{val_rmse:.6f}-{val_mape:.6f}.h5')
	checkpoint = [tf.keras.callbacks.ModelCheckpoint(filepath, verbose=2,  
		monitor='val_loss', save_best_only=True, save_weights_only=False),
		tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=True),
		tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=1)]				 
	return checkpoint

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    return hls_colors

def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors

def color(value):
    digit = list(map(str, range(10))) + list('ABCDEF')
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)

def getRandomIndex(n, x):
	# 索引范围为[0, n)，随机选x个不重复，注意replace=False才是不重复，replace=True则有可能重复
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index
