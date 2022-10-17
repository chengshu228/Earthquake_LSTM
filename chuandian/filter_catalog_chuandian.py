import config

np = config.np
pd = config.pd
tf = config.tf
min_year = config.min_year
max_year = config.max_year 
min_latitude = config.min_latitude 
max_latitude = config.max_latitude 
min_longitude = config.min_longitude
max_longitude = config.max_longitude
min_magnitude = config.min_magnitude 

time = config.time
start_time = time.time()

file_location = config.file_location

data1 = pd.read_csv(file_location+r'\catalog\1970-2020.txt', delimiter=' ', header=None)
data1 = np.array(data1)
print('catalog\n\t原始地震目录(1970-2020): ', data1.shape)
catalog1 = np.delete(data1, list(np.argwhere(data1[:, 0]<min_year)) 
    + list(np.argwhere(data1[:, 0]>max_year))
    + list(np.argwhere(data1[:, 6]>max_latitude)) 
    + list(np.argwhere(data1[:, 6]<min_latitude))
    + list(np.argwhere(data1[:, 7]>max_longitude)) 
    + list(np.argwhere(data1[:, 7]<min_longitude))
    + list(np.argwhere(data1[:, -1]<min_magnitude)), 
    axis=0).reshape(-1, data1.shape[1])
catalog1 = np.delete(catalog1, 9, axis = 1)
catalog1 = np.delete(catalog1, 8, axis = 1)

data2 = pd.read_csv(file_location + r'\catalog\2020-2021.txt', delimiter=' ', header=None)
data2 = np.array(data2)
print('catalog\n\t原始地震目录: ', data2.shape)
catalog2 = np.delete(data2, list(np.argwhere(data2[:, 0]<min_year)) 
    + list(np.argwhere(data2[:, 0]>max_year))
    + list(np.argwhere(data2[:, 6]>max_latitude)) 
    + list(np.argwhere(data2[:, 6]<min_latitude))
    + list(np.argwhere(data2[:, 7]>max_longitude)) 
    + list(np.argwhere(data2[:, 7]<min_longitude))
    + list(np.argwhere(data2[:, -1]<min_magnitude)), 
    axis=0).reshape(-1, data2.shape[1])

catalog = np.concatenate((catalog1, catalog2), axis=0)
print('\t处理后的地震目录: ', catalog.shape, 
    '\tNS纬度范围: ', max(catalog[:,6]), min(catalog[:,6]), 
    '\tEW经度范围: ', max(catalog[:,7]), min(catalog[:,7]),
    '\t震级范围: ', max(catalog[:,-1]), min(catalog[:,-1]))

np.savetxt(file_location+r'\catalog\filter_catalog.txt', catalog, fmt='%f', delimiter=' ')
print((time.time()-start_time)/60, 'minutes')


# # 将地震目录中的面波震级转化为里氏震级,并统一震级 
# # ML=(MS+1.08)/1.13  ML=(1.17MB+0.67)/1.13
# for i in np.arange(len(data)):
#     # if data[i, -2]=='ML':
#     #     data[i, -1] = np.around((data[i, -1]+1.08)/1.13, 1)
#     # elif data[i, -2]=='mb':
#     #     data[i, -1] = np.around((1.17*data[i, -1]+0.67)/1.13, 1)
#     # else:
#     #     data[i, -1] = data[i, -1]

