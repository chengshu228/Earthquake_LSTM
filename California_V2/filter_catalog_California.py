import numpy as np
import time

import config
data_location = config.data_location
min_year,max_year = config.min_year,config.max_year 
min_lat,max_lat = config.min_lat,config.max_lat 
min_lon,max_lon = config.min_lon,config.max_lon
min_mag = config.min_mag 

start_time = time.time()

catalog = np.genfromtxt(data_location+\
    f'\merge_catalog_{min_year}_{max_year}.txt',delimiter=' ')
print('\tcatalog\n\t原始地震目录({}-{}):'.format(
    min_year, max_year), catalog.shape)

catalog = np.delete(catalog, 
    list(np.argwhere(catalog[:,0]<min_year)) 
    + list(np.argwhere(catalog[:,0]>max_year))
    + list(np.argwhere(catalog[:,-2]>max_lat)) 
    + list(np.argwhere(catalog[:,-2]<min_lat))
    + list(np.argwhere(catalog[:, -1]>max_lon)) 
    + list(np.argwhere(catalog[:, -1]<min_lon))
    + list(np.argwhere(catalog[:,-3]<min_mag))
    , 
    axis=0).reshape(-1, catalog.shape[1])

print('\t处理后的地震目录:', catalog.shape) 
print('\n\tNS纬度范围:[{}, {}]'.format(min(catalog[:,-2]), max(catalog[:,-2])), 
    '\n\tEW经度范围:[{}, {}]'.format(min(catalog[:,-1]), max(catalog[:,-1])),
    '\n\t震级范围:[{}, {}]'.format(min(catalog[:,-3]), max(catalog[:,-3])))
    
np.savetxt(data_location+\
    f'\\filter-min_year{min_year}-min_year{max_year}-'+\
    f'min_lat{min_lat}-max_lat{max_lat}-'+\
    f'min_lon{min_lon}-max_lon{max_lon}-'+\
    f'min_mag{min_mag}' +\
    '.txt',
    catalog, fmt='%f', delimiter=' ')

print('\t', (time.time()-start_time)/60, 'minutes')

# 831906

# # 将地震目录中的面波震级转化为里氏震级,并统一震级 
# # ML=(MS+1.08)/1.13  ML=(1.17MB+0.67)/1.13
# for i in np.arange(len(data)):
#     # if data[i, -2]=='ML':
#     #     data[i, -1] = np.around((data[i, -1]+1.08)/1.13, 1)
#     # elif data[i, -2]=='mb':
#     #     data[i, -1] = np.around((1.17*data[i, -1]+0.67)/1.13, 1)
#     # else:
#     #     data[i, -1] = data[i, -1]

