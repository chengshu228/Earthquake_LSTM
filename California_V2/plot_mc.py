import config
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time 
start_time = time.time()

data_location = config.data_location
file_location = config.file_location
times = config.times
min_year = config.min_year
max_year = config.max_year 
matplotlib.rcParams.update(config.config_font)
plt.rcParams['font.sans-serif']=['simsun'] 
plt.rcParams['axes.unicode_minus']=False 

catalog = np.genfromtxt(data_location+\
    f'\merge_catalog_{min_year}_{max_year}.txt',delimiter=' ')
print(catalog.shape)
magnitude = np.around(np.array(catalog[:,-3], dtype=np.float32), 1)

# Mc = np.arange(-0.01, 6.01, 0.1)
Mc = np.arange(0.09, 4.01, 0.1)
# Mc = np.arange(-0.01, 7.21, 0.1)
b = []
for Mtry in Mc:
    min_magnitude = Mtry        
    magnitude0 = magnitude[np.where(magnitude>=Mtry-0.05)]
    mean_magnitude = np.mean(magnitude0)  
    # b值最大似然估计法
    b_mle = (np.log10(math.exp(1)) / (mean_magnitude-min_magnitude))
    b.append(b_mle)
fig = plt.figure(figsize=(5, 4))
plt.grid(True, linestyle='--')
plt.plot(Mc, b, 'dodgerblue', marker='o',
    markersize=4, markerfacecolor='white')
plt.xlabel('Lower Limit Magnitude', fontproperties=times, fontsize=18)
plt.ylabel('b Value (MLE)', fontproperties=times, fontsize=18)
plt.tick_params(labelsize=16)
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
plt.tight_layout()
plt.savefig(file_location+f'\\figure\seism_b_{min_year}_{max_year}.pdf')
plt.show()

count = []
for i in np.arange(min(magnitude), max(magnitude)+0.05, 0.1):       
    names = globals()
    names['count_' + str(i)] = np.sum(magnitude>=np.around(i, 1)-0.05)
    count.append(names['count_' + str(i)])
fig = plt.figure(figsize=(5, 4))
plt.grid(True, linestyle='--')
plt.plot(np.arange(min(magnitude), max(magnitude)+0.05, 0.1), 
    np.log10(count), color='dodgerblue', marker='o', 
    markersize=4, markerfacecolor='white')
plt.xlabel('Lower Limit Magnitude', fontproperties=times, fontsize=18)
plt.ylabel('logN', fontproperties=times, fontsize=18)
# plt.title(r'logN-M', fontproperties=times, fontsize=20, color='red')
plt.tick_params(labelsize=16)
plt.xlim(-0.01, 8.01)
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.MultipleLocator(1))
plt.tight_layout()
plt.savefig(file_location+f'\\figure\seism_logN_{min_year}_{max_year}.pdf')
plt.show()

print((time.time()-start_time)/60, 'minutes')

