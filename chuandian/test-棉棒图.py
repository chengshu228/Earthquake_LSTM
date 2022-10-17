

import julian
import datetime
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
# 以上几个包需要安装，如何安装可以自己上网查

import config
file_location = config.file_location # 根据你的目录设置该变量

catalog = pd.read_csv(file_location+r'\catalog\filter_catalog.txt', 
    dtype=float, delimiter=' ', header=None)
catalog = np.array(catalog)

# 为了让图更清楚提携，我们去掉了小震
catalog = np.delete(catalog, 
    list(np.argwhere(catalog[:, -1]<4)), 
    axis=0).reshape(-1, catalog.shape[1])

date = pd.DataFrame({'year': catalog[:,0], 'month': catalog[:,1], 
    'day': catalog[:,2], 'hour': catalog[:,3], 
    'minute': catalog[:,4], 'second': catalog[:,5]})
date = pd.to_datetime(date)

# 横轴为儒略日
jul_date = []
for i in np.arange(len(date)):
    jd = julian.to_jd(date[i], fmt='jd')
    jul_date.append(jd)

# 设置横轴标签
year = []
for t in np.arange(1970, 2021+1, 2):
    time = '{}-01-01T00:00:00.00'.format(t)
    t = Time(time, format='isot', scale='utc')
    t_jd=t.jd  
    year.append(t_jd)

# 画图
fig = plt.figure(figsize=(12, 6))
markerline, stemlines, baseline = plt.stem(jul_date, catalog[:, -1],\
    linefmt='-', markerfmt=None, basefmt='--', label='TestStem')
plt.xlabel('Time', fontsize=18)
plt.ylabel('$M_L$', fontsize=18)
plt.title('M-t', fontsize=20, color='red')
plt.xticks(year, list(np.arange(1970, 2021+1, 2)), size=6)
plt.yticks(size=14)
plt.grid(True, linestyle='--', linewidth=1.5)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.show()