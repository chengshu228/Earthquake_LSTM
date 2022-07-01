
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config

fonten = config.fonten
file_location = config.file_location
min_magnitude = config.min_magnitude

# min_magnitude = -0.1
catalog = pd.read_csv(file_location+r'\catalog\merge_catalog.txt', 
    dtype=str, delimiter=' ', header=None)
catalog = np.array(catalog)
catalog = np.around(np.array(catalog[:,-3], dtype=np.float32), 1)
fig = plt.figure(figsize=(14, 6))
plt.scatter(np.arange(len(catalog)), catalog,
    c='', edgecolor='dodgerblue')
plt.xlabel('Time (1932/1/1-2021/4/1)', fontsize=18)
plt.ylabel('Magnitude', fontsize=18)
plt.title(u'M-t', fontsize=20, color='red')
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(True, linestyle='--', linewidth=1.5)
# plt.ylim(1.6, 8.1)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.show()

catalog = pd.read_csv(file_location+r'\catalog\filter_catalog.txt', 
    dtype=str, delimiter=' ', header=None)
catalog = np.array(catalog)
catalog = np.around(np.array(catalog[:,-3], dtype=np.float32), 1)

count = []
for i in np.arange(min_magnitude, max(catalog)+0.05, 0.1):       
    names = globals()
    names['count_' + str(i)] = np.sum(catalog>=np.around(i, 1)-0.05)
    count.append(names['count_' + str(i)])

fig = plt.figure(figsize=(6, 5))
plt.grid(True, linestyle='--', linewidth=1.5)
plt.plot(np.arange(min_magnitude, max(catalog)+0.05, 0.1), np.log10(count), 
    color='dodgerblue', linewidth=2, marker='o')
plt.xlabel('M>=Mi-0.05', fontsize=18)
plt.ylabel('logN', fontsize=18)
plt.title(r'logN-M', fontdict=fonten, fontsize=20, color='red')
plt.tight_layout()
plt.xticks(size=14, rotation=60)
plt.yticks(size=14)
plt.xlim(-0.01, 6.01)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1.)
ax.spines['top'].set_linewidth(1.)
ax.spines['right'].set_linewidth(1.)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout()
plt.savefig(r'.\figure\cumulative.pdf')
plt.show()

fig = plt.figure(figsize=(14, 6))
plt.subplot(1,1,1) 
plt.scatter(np.arange(len(catalog)), catalog,
    c='', edgecolor='dodgerblue')
plt.xlabel('Sample Index (1932-2021)', fontsize=18)
plt.ylabel('ML/Ms/mb', fontsize=18)
plt.title(u'M-t', fontsize=20, color='red')
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(True, linestyle='--', linewidth=1.5)
# plt.ylim(1.6, 8.1)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.show()

factor = pd.read_csv(file_location+r"\factor\factor-"+config.catolog_name+r".txt", 
    delimiter=' ', header=None, dtype=np.float32)
factor = factor.values
print('\n  initial factor shape: ', factor.shape)
output_data = np.around(np.array(factor[:, 0], dtype=np.float32), 1)
count = []
for i in np.arange(min_magnitude, max(output_data)+0.05, 0.1):       
    names = globals()
    names['count_' + str(i)] = np.sum(i==output_data)
    count.append(names['count_' + str(i)])

fig = plt.figure(figsize=(8, 6))
plt.grid(True, linestyle='--', linewidth=1.5)
plt.plot(np.arange(min_magnitude, max(output_data)+0.05, 0.1), count, 
    color='dodgerblue', linewidth=2)
plt.xlabel('Earthquake Magnitude', fontsize=18)
plt.ylabel('Cumulative Frequency', fontsize=18)
plt.title(r'Frequency-M', fontdict=fonten, fontsize=20, color='red')
plt.tight_layout()
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.savefig(r'.\figure\cumulative.pdf')
plt.show()