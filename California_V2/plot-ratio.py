
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from utils import dataset

features = 17
blocks = 21

value = dataset(blocks=blocks)
value[:, 0::features] = value[:, 0::features]
output_data = value[:, features*blocks::features]

fontcn = {'family':'YouYuan','size': 10} 

# 查看样本比例
num_nonfraud = np.sum(output_data<6)
num_fraud = np.sum(output_data>=6)
plt.bar(['M>=6', 'M<6'], [num_fraud, num_nonfraud], color='dodgerblue')
plt.ylabel("Number of samples")
x = [0, 1]
y = [num_fraud, num_nonfraud]
for a, b in zip(x, y):
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
plt.show()

for i, threshold in enumerate([6, 7, 8]):
    num_nonfraud = np.sum(output_data<threshold)
    num_fraud = np.sum(output_data>=threshold)
    plt.bar(['M>={}'.format(threshold), 'M<{}'.format(threshold)], \
        [num_fraud, num_nonfraud], color='dodgerblue')
    plt.ylabel('Number of samples', fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.grid(True, linestyle='--', linewidth=1.0)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    x = [0, 1]
    y = [num_fraud, num_nonfraud]
    plt.text(0.5, 2000, '困难度1={:.2f}%'.format(100*num_nonfraud/num_fraud), \
        ha='center', va='bottom', fontsize=12, fontdict=fontcn)
    plt.text(0.5, 1500, '困难度2={:.2f}%'.format(100*num_nonfraud/(len(output_data)*blocks)), \
        ha='center', va='bottom', fontsize=12, fontdict=fontcn)
    for a, b in zip(x, y):
        plt.text(a, b+0.05, '{:.0f}'.format(b), ha='center', va='bottom', fontsize=12)
        plt.text(a, b-500, '{:.2f}%'.format(100*b/(len(output_data)*blocks)), \
            ha='center', va='bottom', fontsize=12)
    plt.show()