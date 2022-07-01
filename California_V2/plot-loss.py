
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler

import utils 
import config

start_time = time.time()

file_location = config.file_location
file_name = config.file_name
n_splits = config.n_splits

times = config.times

if n_splits == 1:
    loss = utils.read_data(file_location=file_location+r'\loss', name='loss-{}'.format(file_name))
    rmse = utils.read_data(file_location=file_location+r'\loss', name='rmse-{}'.format(file_name))
    val_loss = utils.read_data(file_location=file_location+r'\loss', name='val_loss-{}'.format(file_name))  
    val_rmse = utils.read_data(file_location=file_location+r'\loss', name='val_rmse-{}'.format(file_name))
    
    x = np.arange(len(loss))

    fig = plt.figure(figsize=(16, 5))
    plt.subplot(1,3,1) 
    plt.grid(True, linestyle='--', linewidth=1.0)
    plt.plot(x, loss, color='dodgerblue', label='training set', linewidth=2)
    plt.plot(x, val_loss, color='indianred', label='validation set', linewidth=2)
    plt.xlabel('epoch', fontproperties='Arial', fontsize=16)
    plt.ylabel('MSE', fontproperties='Arial', fontsize=16)
    plt.xticks(size=12)
    plt.yticks(size=12)
    # plt.ylim(-0.01, 0.21)
    # plt.text(5, 0.215, '(a)', fontsize=14)
    plt.legend(loc='upper right', prop=times, fontsize=14)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    plt.subplot(1,2,2) 
    plt.grid(True, linestyle='--', linewidth=1.0)
    plt.plot(x, rmse, color='dodgerblue', label='training set', linewidth=2)
    plt.plot(x, val_rmse, color='indianred', label='validation set', linewidth=2)
    plt.xlabel('epoch', fontproperties='Arial', fontsize=16)
    plt.ylabel('RMSE', fontproperties='Arial', fontsize=16)
    plt.xticks(size=12)
    plt.yticks(size=12)
    # plt.ylim(-0.01, 0.41)
    # plt.text(5, 0.425, '(c)', fontsize=14)
    plt.legend(loc='upper right', prop=times, fontsize=14)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    plt.subplots_adjust(wspace=0.25) 
    plt.savefig(r'.\figure\{}-EpochLoss.png'.format(file_name))
    plt.show()
else:
    for fold_no in np.arange(config.n_splits):
        loss = utils.read_data(file_location=file_location+r'\loss', name='loss-{}-{}'.format(file_name, fold_no+1))
        rmse = utils.read_data(file_location=file_location+r'\loss', name='rmse-{}-{}'.format(file_name, fold_no+1))
        val_loss = utils.read_data(file_location=file_location+r'\loss', name='val_loss-{}-{}'.format(file_name, fold_no+1))  
        val_rmse = utils.read_data(file_location=file_location+r'\loss', name='val_rmse-{}-{}'.format(file_name, fold_no+1))
        
        x = np.arange(len(loss))

        fig = plt.figure(figsize=(18, 5.5))
        plt.subplot(1,2,1) 
        plt.grid(True, linestyle='--', linewidth=1.0)
        plt.plot(x, loss, color='dodgerblue', label='training set', linewidth=2)
        plt.plot(x, val_loss, color='indianred', label='validation set', linewidth=2)
        plt.xlabel('epoch', fontproperties='Arial', fontsize=18)
        plt.ylabel('MSE', fontproperties='Arial', fontsize=18)
        plt.xticks(size=14)
        plt.yticks(size=14)
        # plt.ylim(-0.01, 0.21)
        plt.legend(prop=times, fontsize=15)
        # plt.text(5, 0.215, '(a)', fontproperties='Arial', fontsize=16)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        plt.subplot(1,2,2) 
        plt.plot(x, rmse, color='dodgerblue', label='training set', linewidth=2)
        plt.plot(x, val_rmse, color='indianred', label='validation set', linewidth=2)
        plt.xlabel('epoch', fontproperties='Arial', fontsize=18)
        plt.ylabel('RMSE', fontproperties='Arial', fontsize=18)
        plt.xticks(size=14)
        plt.yticks(size=14)
        # plt.ylim(-0.01, 0.41)
        plt.legend(prop=times, fontsize=15)
        plt.grid(True, linestyle='--', linewidth=1.0)
        # plt.text(5, 0.425, '(c)', fontproperties='Arial', fontsize=16)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        plt.suptitle(f'Error for fold {fold_no+1}', fontproperties='Arial', fontsize=20)
        plt.subplots_adjust(wspace=0.35) 
        plt.show()

print((time.time()-start_time)/60, ' minutes')
