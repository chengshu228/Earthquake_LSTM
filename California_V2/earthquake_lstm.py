# %%
import pandas as pd
import config

# file = r'C:\Users\cshuu\Desktop\Earthquake_LSTM-main\California_V2\data\factor-min_year1932-max_year2021-span_lat2-span_lon4-'+\
#        f'time_window{config.time_window}-next_month{config.next_month}-min_mag-2-blocks1.txt'
data_location = config.data_location

file = data_location+f'\\6-San Jacinto Fault(1).txt'



data = pd.read_csv(file,sep=' ',header=None)

cols = ['max_mag_next',
         'frequency',
         'max_mag',
         'mean_mag',
         'b_lstsq',
         'b_mle',
         'a_lstsq',
         'max_mag_absence',
         'rmse_lstsq',
         'total_energy_square',
         'mean_lon',
         'rmse_lon',
         'mean_lat',
         'rmse_lat',
         'k',
         'epicenter_lon',
         'epicenter_lat']
data.columns = cols

# LSTM处理特征的方式
# 不光使用当月的数据来预测，还使用过去lookback个月的数据作为模型的输入，
# 构建一个三维的数据（datalength,lookback,features）输入LSTM模型，这样兼顾了多个时间点的特征，

# %%
# data manipulation
import numpy as np
import pandas as pd
import datetime, os, random
from pathlib import Path

# scikit-learn modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# tensorflow & keras modules
import tensorflow as tf                                                       # tf.keras.optimizers.Adam - use 'tf.' only when calling direct methods ~ tf 2.0+
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator       # convert dataframe to array to use in timeseries generator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM

from tensorflow.keras.optimizers import Adam, RMSprop 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_absolute_error
# plotting & outputs
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# supress warnings
import warnings
warnings.filterwarnings('ignore')

def set_seeds(seed=123): 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
results_path = Path('results', 'lstm_time_series')
if not results_path.exists():
    results_path.mkdir(parents=True)

    
def model_main(nodes_num,batch_size,epoches ,data,features,target,lookback=12,name='LSTM'):
    train_len = int(len(data)*0.9)
    train_data = data.iloc[:train_len].copy()
    test_data = data.iloc[train_len:].copy()

    print(train_data.head())
    print(train_data.shape, data.shape)


    # train_data=train_data.drop(labels=[
    #      'b_lstsq',
    #     'b_mle',
    #     'a_lstsq'],axis=1)
    # test_data=test_data.drop(labels=[
    #     'b_lstsq',
    #     'b_mle',
    #     'a_lstsq'],axis=1)

    # print(train_data['b_mle'])
    # Output the train and test data size
    print(f"Train and Test Size {len(train_data)}, {len(test_data)}")

    # Scale the features MinMax for training and test datasets
    X_scaler = MinMaxScaler()
    scaled_train_data_X = X_scaler.fit_transform(train_data[features])
    scaled_test_data_X = X_scaler.transform(test_data[features])

    y_scaler = MinMaxScaler()
    scaled_train_data_y = y_scaler.fit_transform(train_data[[target]])
    scaled_test_data_y = y_scaler.transform(test_data[[target]])

    # 第一列是target
    scaled_train_data = pd.concat([pd.DataFrame(scaled_train_data_y),pd.DataFrame(scaled_train_data_X)],axis=1).values
    scaled_test_data = pd.concat([pd.DataFrame(scaled_test_data_y),pd.DataFrame(scaled_test_data_X)],axis=1).values


    # time sequence
    def generate_sequence(data, sequence_length=6):
        # create X & y data array
        X = []
        y = []
        # for i in range(sequence_length, len(data)):
        #     # 当前之前的sequence_length个
        #     X.append(data[i - sequence_length:i, :])
        #     # 当前第一列
        #     y.append(data[i, 0])

        for i in range(sequence_length, len(data)):
            # 当前之前的sequence_length个
            X.append(data[i - sequence_length:i, 1:])
            # 当前第一列
            y.append(data[i - 1, 0])

        # Converting x_train and y_train to Numpy arrays
        return np.array(X), np.array(y)

    X_train, y_train = generate_sequence(data=scaled_train_data, sequence_length=lookback)
    print(f'X_train: {X_train.shape}, y_train {y_train.shape}')

    # generate sequence
    X_test, y_test = generate_sequence(data=scaled_test_data, sequence_length=lookback)
    print(f'X_test: {X_test.shape}, y_test {y_test.shape}')


    # reshaping array
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    y_train = y_train[:, np.newaxis] 

    # check the array size
    print(f'X_train Shape: {X_train.shape}, y_train {y_train.shape}')


    # reshaping array
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_train.shape[2]))
    y_test = y_test[:, np.newaxis] 

    # check the array size
    print(f'X_test Shape: {X_test.shape}, y_test {y_test.shape}')

    # Create a model
    def create_model(lookback,hu=512,):

        # instantiate the model
        model = Sequential()
        model.add(LSTM(units=hu, input_shape=(lookback, X_train.shape[2]), activation = 'relu', return_sequences=False, name='LSTM'))
        model.add(Dense(units=1, name='Output'))        # can also specify linear activation function 

        # specify optimizer separately (preferred method))
    #     opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        opt = Adam(lr=0.0001, epsilon=1e-08, decay=0.0)       # adam optimizer seems to perform better for a single lstm

        # model compilation
        model.compile(optimizer=opt, loss='mse', metrics=['mae'])

        return model

    # lstm network
    model = create_model(hu=nodes_num, lookback=lookback)

    # summary
    print(model.summary())
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    # Specify callback functions
    model_path = (results_path / ('model.h5')).as_posix()
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    my_callbacks = [
        EarlyStopping(patience=100, monitor='val_loss', mode='min', verbose=1, restore_best_weights=True),
        # save best params
    #     ModelCheckpoint(filepath=model_path, verbose=1, monitor='loss', save_best_only=True),
        # tensorboard
    #     TensorBoard(log_dir=logdir, histogram_freq=1)
    ]
    # Model fitting

    print(f'Final X_train Shape;{X_train.shape},{y_train.shape}')
    print(f'Final X_test Shape;{X_test.shape},{y_test.shape}')

    history = model.fit(X_train, 
                              y_train, 
                              batch_size=batch_size, 
                              epochs = epoches,  # 设置epoch
                              verbose=1, 
                              callbacks=my_callbacks, 
                              validation_data = (X_test,y_test),
                              shuffle=False)

    print("training finished... please wait...")
    # calculate rmse of loss function
    train_rmse_scaled = np.sqrt(model.evaluate(X_train, y_train, verbose=0))
    test_rmse_scaled = np.sqrt(model.evaluate(X_test, y_test, verbose=0))
    print(f'Train RMSE: {train_rmse_scaled[0]:.4f} | Test RMSE: {test_rmse_scaled[0]:.4f}')

    # predictions
    y_pred = model.predict(X_test)
    df = pd.DataFrame({
        'actual': y_scaler.inverse_transform(y_test).flatten(),
        'prediction': y_scaler.inverse_transform(y_pred).flatten()}, 
        index = test_data[lookback:].index)

    df['spread'] = df['prediction'] - df['actual']
    mae= mean_absolute_error(df.actual, df.prediction)
    rmse = mean_squared_error(df.actual, df.prediction)**0.5
    r2 = r2_score(df.actual, df.prediction)
    print(f'R-square: {r2:0.4}')
    print(f'MAE: {mae:0.4}')
    print(f'RMSE: {rmse:0.4}')

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.plot(pd.DataFrame(df.actual-df.prediction), '-o', color='red', label='actual-prediction')
    # ax.plot(df.prediction, '-o', color='blue', label='prediction')
    plt.axhline(y=0.5)
    plt.axhline(y=-0.5)
    # ax.legend()
    # plt.suptitle('LSTM Actual-Prediction')
    # plt.savefig(f'{name}_Prediction.jpg', dpi=300)
    # plt.show()

    plt.subplot(2,1,2)
    plt.plot(pd.DataFrame(df.actual),'-o', color='red', label='actual')
    plt.plot(pd.DataFrame(df.prediction),'-*', color='blue', label='prediction')
    plt.legend()
    plt.suptitle('LSTM Prediction')
    plt.savefig(f'{name}_Prediction.jpg',dpi=300)
    plt.show()
    ########## History
    # get training history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x_epoch = list(range(1,len(loss)+1))
    # plot the figure
    fig = plt.figure(figsize=(10,6))
    plt.plot(x_epoch,loss,label='train_loss')
    plt.plot(x_epoch,val_loss,label='test_loss')
    plt.title("Loss")
    plt.xlabel('Epoches')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{name}_Loss.jpg',dpi=300)
    plt.show()
    
    # hist
    error_abs = df.prediction - df.actual
    error_relative = (df.prediction - df.actual)/ df.actual
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(1,2,1)
    ax1.hist(error_abs)
    ax1.set_title("Absolute Error")
    ax1.set_ylabel("Frequency")
    ax1.set_xlabel("Predicted - Observed")
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.hist(error_relative)
    ax2.set_title("Relative Error")
    ax2.set_ylabel("Frequency")
    ax2.set_xlabel("(Predicted - Observed)/Observed")
    plt.show()
    return mae,rmse,r2

nodes_num = 256    # 隐藏层的节点数
batch_size = 12  # 一次性往里面丢多少样本
epoches = 1000
name = 'lstm'
lookback = 4
features = ['frequency',
         'max_mag',
         'mean_mag',
         'b_lstsq',
         'b_mle',
         'a_lstsq',
         'max_mag_absence',
         'rmse_lstsq',
         'total_energy_square',
         'mean_lon',
         'rmse_lon',
         'mean_lat',
         'rmse_lat',
         'k',
         'epicenter_lon',
         'epicenter_lat']
target = 'max_mag_next'

# rt = model_main(nodes_num,batch_size,epoches ,data,lookback,features,target)
rt = model_main(nodes_num,batch_size,epoches ,data,features,target,lookback)
# def model_main(nodes_num,batch_size,epoches ,data,lookback,features,target,name='LSTM'):
# %%



