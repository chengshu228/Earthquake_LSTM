import os
import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, optimizers, regularizers, initializers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import datetime

lag = 40

dataframe = pd.read_csv(r'C:\cshu\Person\Lilinfang\dataset.csv', header=None)
raw_data = dataframe.values.astype('float32')
dataframe.head()
print("raw_data.shape=", raw_data.shape)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg

reframed = series_to_supervised(raw_data, n_in=lag, n_out=lag)
print(reframed.head())
values = reframed.values
n_features = values.shape[1]
print("values.shape=", values.shape)
values = values[[i for i in np.arange(0, len(values), 9)], :]


data = values[:, :17*9]
area = 9 # 9,10...,17
output_data = values[:, 17*area]
# output_data = values[:, -17]
print("output_data.shape=", output_data.shape, np.max(output_data), np.min(output_data))
print(data.shape)

labels = np.where(output_data<6, 1, output_data)
labels = np.where(output_data>=6, 0, output_data)
print(labels.shape)
data = data[:, :]#.reshape(-1, 17*9)

train_data, test_data = data[:400,:], data[400:,:]
train_labels, test_labels = labels[:400], labels[400:]
print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

# Normalize the data to [0,1]
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)
train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)
data = (data - min_val) / (max_val - min_val)
train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)
data = tf.cast(data, tf.float32)
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)
labels = labels.astype(bool)

# You will train the autoencoder using only the normal rhythms, 
# which are labeled in this dataset as 1. 
# Separate the normal rhythms from the abnormal rhythms
normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]
normal_data = data[labels]
anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]
anomalous_data = data[~labels]
print(normal_train_data.shape, normal_test_data.shape, 
        anomalous_train_data.shape, anomalous_test_data.shape)

# normal_train_data1 = normal_train_data[:, tf.newaxis, :]
# normal_test_data1 = normal_test_data[:, tf.newaxis, :]
# normal_data1 = normal_data[:, tf.newaxis, :]
# anomalous_train_data1 = anomalous_train_data[:, tf.newaxis, :]
# anomalous_test_data1 = anomalous_test_data[:, tf.newaxis, :]
# anomalous_data1 = anomalous_data[:, tf.newaxis, :]
# train_data1 = train_data[:, tf.newaxis, :]
# test_data1 = test_data[:, tf.newaxis, :]
# data1 = data[:, tf.newaxis, :]

# plt.figure(figsize=(14, 5))
# plt.subplot(121)
# plt.grid()
# plt.plot(np.arange(len(normal_train_data[0])), normal_train_data[0])
# plt.xlabel("length_normal_train_data")
# plt.ylabel("normal_train_data")
# plt.title("A Normal Earthquake")
# plt.subplot(122)
# plt.grid()
# plt.plot(np.arange(len(anomalous_train_data[0])), anomalous_train_data[0])
# plt.xlabel("length_anomalous_train_data")
# plt.ylabel("anomalous_train_data")
# plt.title("An Anomalous Earthquake")
# plt.show()

class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            # layers.LSTM(16, activation='tanh', 
            #             # kernel_initializer=initializers.he_normal(), 
            #             return_sequences=True),
            # layers.LSTM(8, activation='relu', 
            #             kernel_initializer=initializers.he_normal(), 
            #             return_sequences=True),
            layers.Dense(16, activation="relu"),
            # layers.Dense(8, activation="relu"), 
        ])
        self.decoder = tf.keras.Sequential([
            # layers.LSTM(8, activation='relu', 
            #             kernel_initializer=initializers.he_normal(), 
            #             return_sequences=True),
            # layers.LSTM(16, activation='relu', 
            #             kernel_initializer=initializers.he_normal(), 
            #             return_sequences=False),
            # layers.Dense(units=17*9, 
            #             activation='relu', 
            #             kernel_initializer=initializers.glorot_normal(),
            #             bias_initializer=tf.zeros_initializer(), 
            #             #kernel_regularizer=regularizers.l2(0.01), 
            #             name="output_layer"),
            # layers.Dense(8, activation="relu"),
            # layers.Dense(16, activation="relu"),
            layers.Dense(17*9, activation="relu"),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def callback():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join("models_vae")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_dir = os.path.join(model_dir, current_time)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_dir1 = os.path.join(model_dir, "{epoch:05d}-{loss:.6f}.h5")
    callbacks = [tf.keras.callbacks.ModelCheckpoint(model_dir1, monitor="val_loss", 
                    verbose=1, save_best_only = True, save_weights_only=False),
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", 
                    patience=10**4, verbose=True),
                # tf.keras.callbacks.ModelCheckpoint(filepath="SofaSofa_model.h5", 
                #     verbose=0, save_best_only=True)
                ] 
    return callbacks

autoencoder = AnomalyDetector()
learning_rate = 1e-4
optimizer = optimizers.Adam(lr=learning_rate)
autoencoder.compile(optimizer=optimizer, loss='mse', 
                    metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
history = autoencoder.fit(normal_train_data, normal_train_data, 
                        epochs=1000, 
                        batch_size=8,
                        validation_data=(test_data, test_data),
                        shuffle=True, 
                        callbacks=callback(),
                        ).history

# 画出损失函数曲线
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot(history['loss'], c='dodgerblue', lw=2)
plt.plot(history['val_loss'], c='coral', lw=2)
plt.title('model loss')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.subplot(132)
plt.plot(history['mae'], c='dodgerblue', lw=2)
plt.plot(history['val_mae'], c='coral', lw=2)
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.subplot(133)
plt.plot(history['rmse'], c='dodgerblue', lw=2)
plt.plot(history['val_rmse'], c='coral', lw=2)
plt.title('model rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

# 读取模型
# autoencoder = load_model(r'C:\cshu\Person\Lilinfang\SofaSofa_model.h5')
encoded_imgs = autoencoder.encoder(normal_test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
plt.figure(figsize=(6, 5))
plt.plot(normal_test_data[0],'b')
plt.plot(decoded_imgs[0],'r')
plt.fill_between(np.arange(17*9), decoded_imgs[0], normal_test_data[0], color='lightcoral' )
plt.title('normal_test_data')
plt.legend(labels=["Input", "Reconstruction", "Error"], loc='upper right')

encoded_imgs = autoencoder.encoder(anomalous_test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
plt.figure(figsize=(6, 5))
plt.plot(anomalous_test_data[0],'b')
plt.plot(decoded_imgs[0],'r')
plt.fill_between(np.arange(17*9), decoded_imgs[0], anomalous_test_data[0], color='lightcoral' )
plt.title('anomalous_test_data')
plt.legend(labels=["Input", "Reconstruction", "Error"], loc='upper right')

reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
threshold_mae = np.mean(train_loss) + np.std(train_loss)
train_loss = tf.keras.losses.mse(reconstructions, normal_train_data)
threshold_mse = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold_mae, threshold_mse)
plt.figure(figsize=(6, 5))
plt.hist(train_loss, bins=50)
plt.xlabel("Train loss")
plt.ylabel("Number of training samples")

reconstructions = autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)
plt.figure(figsize=(6, 5))
plt.hist(test_loss, bins=50)
plt.xlabel("Test loss")
plt.ylabel("Number of testing samples")

def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, preds)))
    print("Precision = {}".format(precision_score(labels, preds)))
    print("Recall = {}".format(recall_score(labels, preds)))

preds = predict(autoencoder, test_data, threshold_mae)
print_stats(preds, test_labels) 
preds = predict(autoencoder, test_data, threshold_mse)
print_stats(preds, test_labels) 

# 利用训练好的autoencoder重建测试集
pred_test = autoencoder.predict(normal_test_data)
pred_fraud = autoencoder.predict(anomalous_data)
# 计算还原误差MSE和MAE
mse_test = np.mean(np.power(normal_test_data - pred_test, 2), axis=1)
mse_fraud = np.mean(np.power(anomalous_data - pred_fraud, 2), axis=1)
mae_test = np.mean(np.abs(normal_test_data - pred_test), axis=1)
mae_fraud = np.mean(np.abs(anomalous_data - pred_fraud), axis=1)
mse_df = pd.DataFrame()
mse_df['Class'] = [1] * len(mse_test) + [0] * len(mse_fraud)
mse_df['MSE'] = np.hstack([mse_test, mse_fraud])
mse_df['MAE'] = np.hstack([mae_test, mae_fraud])
# sample()参数frac是要返回的比例，比如df中有10行数据，我只想返回其中的30%,那么frac=0.3
# set_index()和reset_oindex()的区别 前者为现有的dataframe设置不同于之前的index;
# 而后者是还原和最初的index方式：0,1,2,3,4……
mse_df = mse_df.sample(frac=1).reset_index(drop=True)

label_reconstruction = np.hstack([test_labels[test_labels], labels[~labels]])

print(mse_test.shape, mse_fraud.shape, mae_test.shape, mae_fraud.shape)
print(label_reconstruction.shape, labels[~labels].shape, test_labels[test_labels].shape, mse_df[mse_df['Class'] == 0]['MAE'])

# 分别画出测试集中正样本和负样本的还原误差MAE和MSE
markers = ['o', '^']
markers = ['o', '^']
colors = ['coral', 'dodgerblue']
labels = ['M>=6', 'M<6']
plt.figure(figsize=(14, 5))
plt.subplot(121)
for flag in [1, 0]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp.index, 
                temp['MAE'],  
                alpha=0.7, 
                marker=markers[flag], 
                c='',
                edgecolors=colors[flag], 
                label=labels[flag])
plt.title('Reconstruction MAE')
plt.ylabel('Reconstruction MAE')
plt.xlabel('Number of normal_test_data and anomalous_data')
plt.legend(loc='upper right')
plt.axhline(y=threshold_mae, c="r", ls="--", lw=2)
plt.yscale("log") 
plt.ylim(1e-6, 1)
plt.subplot(122)
for flag in [1, 0]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp.index, 
                temp['MSE'],  
                alpha=0.7, 
                marker=markers[flag], 
                c='',
                edgecolors=colors[flag], 
                label=labels[flag])
plt.legend(fontsize=12, loc='upper right')
plt.title('Reconstruction MSE')
plt.ylabel('Reconstruction MSE')
plt.xlabel('Number of normal_test_data and anomalous_data')
plt.legend(loc='upper right')
plt.axhline(y=threshold_mse, c="r", ls="--", lw=2)
plt.yscale("log") 
plt.ylim(1e-8, 1)

# 画出Precision-Recall曲线
plt.figure(figsize=(14, 6))
for i, metric in enumerate(['MAE', 'MSE']):
    plt.subplot(1, 2, i+1)
    precision, recall, _ = precision_recall_curve(mse_df['Class'], mse_df[metric])
    pr_auc = auc(recall, precision)
    plt.title('Precision-Recall curve based on %s\nAUC = %0.2f'%(metric, pr_auc))
    plt.plot(recall[:-2], precision[:-2], c='coral', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')

# 画出ROC曲线
plt.figure(figsize=(14, 6))
for i, metric in enumerate(['MAE', 'MSE']):
    plt.subplot(1, 2, i+1)
    fpr, tpr, _ = roc_curve(mse_df['Class'], mse_df[metric])
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic based on %s\nAUC = %0.2f'%(metric, roc_auc))
    plt.plot(fpr, tpr, c='coral', lw=2)
    plt.plot([0,1],[0,1], c='dodgerblue', ls='--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')

# 画出MSE、MAE散点图
markers = ['o', '^']
colors = ['coral', 'dodgerblue']
labels = ['M>=6', 'M<6']
plt.figure(figsize=(10, 5))
for flag in [1, 0]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp['MAE'], 
                temp['MSE'],  
                alpha=0.7, 
                marker=markers[flag], 
                c=colors[flag], 
                label=labels[flag])
plt.legend(loc='upper right')
plt.ylabel('Reconstruction RMSE')
plt.xlabel('Reconstruction MAE')
plt.show()

plt.figure(figsize=(8, 5))
plt.rcParams['font.sans-serif'] = ['SimHei']
for flag in [1, 0]:
    temp = mse_df[mse_df['Class'] == flag]
    index_max = np.array(np.where(np.array(temp['MAE'])>threshold_mae))
    index_min = np.array(np.where(np.array(temp['MAE'])<=threshold_mae))
    for i in np.arange(mse_df.shape[0]):
        if i in index_max and flag==1 and np.array(temp['MAE'])[i]>threshold_mae:
            plt.scatter(i, 1, alpha=0.7, marker="*", c='', edgecolors="black") # 报大震，有小震
        elif i in index_min and flag==1 and np.array(temp['MAE'])[i]<threshold_mae:
            plt.scatter(i, 2, alpha=0.7, marker="o", c='', edgecolors="g") # 报小震，有小震
        elif i in index_max and flag==0 and np.array(temp['MAE'])[i]>threshold_mae:
            plt.scatter(i, 3, alpha=0.7, marker="v", c='', edgecolors="b") # 报大震，有大震
        elif i in index_min and flag==0 and np.array(temp['MAE'])[i]<threshold_mae:
            plt.scatter(i, 4, alpha=0.7, marker="^", c='', edgecolors="r") # 报小震，有大震
plt.title('Testing')
plt.xlabel('Index of reconstruction_test_data')
plt.ylabel('classification')
# plt.legend(loc='upper left')
plt.show()