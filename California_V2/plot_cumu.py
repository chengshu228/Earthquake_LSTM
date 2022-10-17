
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import config

matplotlib.rcParams.update(config.config_font)
plt.rcParams['font.sans-serif']=['simsun'] 
plt.rcParams['axes.unicode_minus']=False 

times = config.times
data_location = config.data_location
file_location = config.file_location
min_magnitude = config.min_mag
min_year = config.min_year
max_year = config.max_year 

min_magnitude = -0.1

catalog = np.genfromtxt(data_location+\
    f'\merge_catalog_{min_year}_{max_year}.txt',delimiter=' ')

# print(catalog.shape)
# catalog = np.around(np.array(catalog[:,-3], dtype=np.float32), 1)
# fig = plt.figure(figsize=(10, 5))
# plt.grid(True, linestyle='--')
# plt.scatter(np.arange(len(catalog)), catalog,
#     c='none', edgecolor='dodgerblue')
# plt.xlabel('Time (1932/1/1-2021/4/1)', fontproperties=times, fontsize=18)
# plt.ylabel('Magnitude', fontproperties=times, fontsize=18)
# # plt.title(u'M-t', fontsize=20, color='red')
# plt.tick_params(labelsize=16)
# plt.tight_layout()
# plt.show()



# fig = plt.figure(figsize=(10, 5))
# plt.grid(True, linestyle='--')
# plt.scatter(np.arange(len(catalog)), catalog,
#     c='none', edgecolor='dodgerblue', marker='o')
# plt.xlabel('Sample Index (1932-2021)', fontproperties=times, fontsize=18)
# plt.ylabel('ML/Ms/mb', fontproperties=times, fontsize=18)
# # plt.title(u'M-t', fontproperties=times, fontsize=20, color='red')
# plt.tick_params(labelsize=16)
# ax = plt.gca()
# plt.tight_layout()
# plt.show()


output_data = np.around(np.array(catalog[:, -3], dtype=np.float32), 1)
count = []
for i in np.arange(min_magnitude, max(output_data)+0.05, 0.1):       
    names = globals()
    names['count_' + str(i)] = np.sum(i==output_data)
    count.append(names['count_' + str(i)])

fig = plt.figure(figsize=(6, 4))
plt.grid(True, linestyle='--')
plt.plot(np.arange(min_magnitude, max(output_data)+0.05, 0.1), 
    count, color='dodgerblue', marker='o', 
    markersize=4, markerfacecolor='white')
plt.xlabel('Magnitude', fontproperties=times, fontsize=18)
plt.ylabel('Cumulative Frequency', fontproperties=times, fontsize=18)
# plt.title(r'Frequency-M', fontdict=times, fontsize=20, color='red')
plt.tick_params(labelsize=16)
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.MultipleLocator(10000))
plt.tight_layout()
plt.savefig(file_location+r'\figure\seism_m_f.pdf')
plt.show()