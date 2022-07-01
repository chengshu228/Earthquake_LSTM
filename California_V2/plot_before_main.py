
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timedelta
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import time
import matplotlib

import config
times = config.times
data_location = config.data_location
min_year = config.min_year
max_year = config.max_year 

matplotlib.rcParams.update(config.config_font)
plt.rcParams['font.sans-serif']=['simsun'] 
plt.rcParams['axes.unicode_minus']=False 

catalog = np.genfromtxt(data_location+\
    f'\merge_catalog_{min_year}_{max_year}.txt',delimiter=' ')
print(catalog.shape)

year_all = np.around(np.array(catalog[:,0], dtype=np.float32), 0)
month_all = np.around(np.array(catalog[:,1], dtype=np.float32), 0)
day_all = np.around(np.array(catalog[:,2], dtype=np.float32), 0)
hour_all = np.around(np.array(catalog[:,3], dtype=np.float32), 0)
minute_all = np.around(np.array(catalog[:,4], dtype=np.float32), 0)
second_all = np.around(np.array(catalog[:,5], dtype=np.float32), 0)

start_time = time.time()
data_all = []
for i in np.arange(len(catalog)):
    data_temp = datetime(year_all[i],month_all[i],day_all[i],\
        hour_all[i],minute_all[i])
    data_all.append(data_temp)
data_all = np.array(data_all)

magnitude_all = np.around(np.array(catalog[:,-3], dtype=np.float32), 1)
latitude_all = np.around(np.array(catalog[:,-2], dtype=np.float32), 1)
longitude_all = np.around(np.array(catalog[:,-1], dtype=np.float32), 1)

# dis_all = []
# for i in np.arange(len(catalog)):
#     start_time = time.time()
#     if i % 100000 == 0:
#         print(i, (time.time()-start_time)/60, 'minutes')
#     latitude = np.around(np.array(catalog[i, -2], dtype=np.float32), 3)
#     longitude = np.around(np.array(catalog[i, -1], dtype=np.float32), 3)
#     dis = geodesic((latitude, longitude), (latitude_all[i], longitude_all[i])).km
#     dis_all.append(dis)
# dis_all = np.array(dis_all)
# print('dis_all.shape=', dis_all.shape)
# np.savetxt(file_location+r'\catalog\distance.txt', dis_all, fmt='%f', delimiter=' ')

min_max_mag = 7
for i in np.arange(len(catalog)):
    magnitude = np.around(np.array(catalog[i,-3], dtype=np.float32), 1)
    if magnitude>=min_max_mag:
        year = np.array(float(catalog[i,0]))
        month = np.array(float(catalog[i,1]))
        day = np.array(float(catalog[i,2]))
        hour = np.array(float(catalog[i,3]))
        minute = np.array(float(catalog[i,4]))
        second = np.around(np.array(catalog[i,5], dtype=np.float32), 2)
        latitude = np.around(np.array(catalog[i,-2], dtype=np.float32), 3)
        longitude = np.around(np.array(catalog[i,-1], dtype=np.float32), 3)

        now = datetime(year,month,day,hour,minute,second)
        before = now - relativedelta(days=10)

        year_before = before.year
        month_before = before.month
        day_before = before.day
        hour_before = before.hour
        minute_before = before.minute
        second_before = before.second
        # print(year, year_before,
        #     month, month_before,
        #     day, day_before,
        #     hour, hour_before,
        #     minute, minute_before,
        #     second, second_before)
        
        mag_local = magnitude_all[np.where(
            (data_all<=now) & (data_all>=before) )]
        # print(mag_local)

        lat_local = latitude_all[np.where(
            (data_all<=now) & (data_all>=before) )]
        long_local = longitude_all[np.where(
            (data_all<=now) & (data_all>=before) )]
        
        for j in np.arange(len(mag_local)):
            if (lat_local[j]-latitude)**2 + (long_local[j]-longitude)**2 < 0.5**2:
                fig = plt.figure(figsize=(6, 5))
                print(f'{now} & {latitude:.3f}N & {longitude:.3f}W & {mag_local[-1]:.1f} \\')
                # plt.subplot(1,1,1) 
                # plt.plot(np.arange(len(mag_local)), mag_local, 
                #     'dodgerblue', linewidth=2, marker='o',
                #     markersize=4, markerfacecolor='white', alpha=0.75)
                # plt.xlabel('Sample Index', fontproperties=times, fontsize=18)
                # plt.ylabel('Magnitude', fontproperties=times, fontsize=18)
                # plt.title(f'M>={min_max_mag} (10 days, 50KM) \n' + \
                #     f' time: {now} \n {latitude:.3f}N-{longitude:.3f}W'.format(\
                #     ), fontproperties=times, fontsize=20, color='r')
                # plt.tick_params(labelsize=16)
                # plt.grid(True, linestyle='--')
                # ax = plt.gca()
                # ax.yaxis.set_major_locator(plt.MultipleLocator(1))
                # plt.tight_layout()
                # plt.show()
                break
