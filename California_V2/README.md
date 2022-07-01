## 南加州地区地震目录
- [地震目录下载链接](https://service.scedc.caltech.edu/ftp/catalogs/SCEC_DC/)

## config.py
> 放置参数

## merge_catalog.py
> 合并1932年至今的地震目录数据

## filter_catalog_California.py
> 将地震目录中的数据限定在某个区域内，并限制最小震级

## generate-seismic_factor1.py
> 计算整个区块的17个地震因子
## generate-seismic_factor6.py
> 划分6个区块，计算17个地震因子


## ANN.py
放置不同的神经网络

## lstm-fold.py
## lstm-split.py
训练神经网络

## ml.py
调用不同的机器学习方法

## plot-*.py
绘制图像


## 文件夹说明：
.history 跟 _pycache_ 是缓存的文件，可以忽略
catalog: 原始的地震目录数据
data: 经过筛选后的数据，并计算了不同情况下17个地震因子
cnn/lstm/lstm_cnn：训练过程中产生的不同模型，最新生成的模型是最优的
figure: 图像数据集
loss: 训练过程中会保存loss值

## 
